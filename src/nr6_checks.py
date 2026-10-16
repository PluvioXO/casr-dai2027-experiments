"""Meaningful CPU/CUDA plumbing checks. These do not measure pretrained efficacy."""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import copy,json,tempfile,time
import torch
import nr5_rl as v5
from nr5_rl_checks import fixture,scripted_items,run_checks as v5_checks
from nr4_model import NativeSlots
from nr6_train import Config,conditional_kl,all_position_batch,preservation_items,train_recipe,extended_update
from nr6_io import Workspace,atomic_copy
from nr6_probes import parse_work,first_digit,prompt_variant,implied_hops,prefix_kl_values

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def run_checks(out=None):
    started=time.perf_counter();previous_threads=torch.get_num_threads();torch.set_num_threads(1)
    report={}
    try:
        report['v5']=v5_checks()
        base,tok,examples=fixture()
        cfg=Config(max_steps=2,prompts_per_update=1,group_size=4,generation_batch=4,
            max_prompt_tokens=2048,max_new_tokens=4,eval_every=1,save_every=1,eval_batch=2,
            lr=1e-3,min_lr=1e-5,max_mean_kl=100,max_sequence_kl=100,checkpointing=True,
            kl_positions=2,head_chunk=8,kl_head_chunk=7,clip_grad=1e6,snapshot_every=0,audit_steps=[1,2])
        def new_module(device='cpu'):
            return NativeSlots(copy.deepcopy(base).to(device),'mlp',seed=17,n_layers=2,mlp_neurons=3)
        items=scripted_items(tok,[examples[0]]*4)
        module=new_module()
        with torch.no_grad():
            for p in module.parameters():p.normal_(std=.01)
        batch=all_position_batch(items,tok,module,cfg)
        assert len(batch['kl_indices'])==len(batch['positions'])
        with module.route(1),module.gate(v5.MASKS['AB']):
            value=conditional_kl(module,batch,v5.MASKS['A'],cfg)
            value.backward()
        actual=[p.grad.clone() if p.grad is not None else None for p in module.parameters(1)]
        for p in module.parameters():p.grad=None
        with torch.no_grad(),module.gate(v5.MASKS['A']):teacher=v5._chosen_log_probs(module,batch)
        with module.route(1),module.gate(v5.MASKS['AB']):
            student=v5._chosen_log_probs(module,batch)
            expected=v5._per_sequence(v5._forward_kl(teacher,student),batch).mean();expected.backward()
        assert torch.allclose(value,expected,atol=1e-7,rtol=1e-5)
        assert all(torch.allclose(a,p.grad,atol=2e-6,rtol=1e-4) for a,p in zip(actual,module.parameters(1)))
        assert all(p.grad is None for p in module.parameters(0))
        values,changed=prefix_kl_values(module,batch,v5.MASKS['A'],v5.MASKS['AB'],7)
        assert abs(sum(values)/len(values)-float(expected.detach()))<1e-6
        report['chunked_full_position_kl_matches_dense_value_and_gradient']=True
        module.close()

        def fake_generate(module,tok,group,cfg,mask='AB',sample=False):
            torch.rand(3,device=module.device)  # Ensure auxiliary RNG really gets restored.
            return scripted_items(tok,group)
        module=new_module()
        mix_cfg=replace(cfg,preservation_mode='all_mixture')
        rng=torch.get_rng_state().clone()
        with patch.object(v5,'_generate',side_effect=fake_generate) as call:
            mixed,stats=preservation_items(module,tok,items,mix_cfg,31,1)
        assert torch.equal(rng,torch.get_rng_state())
        assert len(mixed)==len(items) and stats['extra_rollout_responses']==3
        assert [c.args[4] for c in call.call_args_list]==['A','B','none']
        report['mixture_mask_counts_and_independent_rng']=True
        module.close()

        for device in (['cpu','cuda'] if torch.cuda.is_available() else ['cpu']):
            module=new_module(device)
            with torch.no_grad():
                for p in module.parameters():p.normal_(std=.01)
            before=[[p.detach().clone() for p in module.parameters(k)] for k in (0,1)]
            opts=[torch.optim.AdamW(module.parameters(k),lr=cfg.lr,weight_decay=0) for k in (0,1,0,1)]
            result=extended_update(module,tok,items,[1,-1,1,-1],replace(cfg,preservation_mode='all_ab'),opts,0,'routed_preserve',items)
            assert result['accepted'] and all(any(not torch.equal(p,b) for p,b in zip(module.parameters(k),before[k])) for k in (0,1))
            assert module.assert_frozen_base()
            params=module.state_dict();states=copy.deepcopy([o.state_dict() for o in opts])
            rejected=extended_update(module,tok,items,[1,-1,1,-1],replace(cfg,preservation_mode='all_ab',
                max_mean_kl=1e-20,max_sequence_kl=1e-20,backtracks=0),opts,0,'routed_preserve',items)
            assert not rejected['accepted'] and equal(params,module.state_dict()) and equal(states,[o.state_dict() for o in opts])
            module.close();report[f'extended_update_and_atomic_rejection_{device}']=True

        with tempfile.TemporaryDirectory(prefix='nr6_checks_') as name:
            root=Path(name);data={'train':examples,'dev':examples}
            m1=new_module();m2=new_module()
            with patch.object(v5,'_generate',side_effect=fake_generate):
                v5.train_arm(m1,tok,data,cfg,root/'v5','mlp_routed_preserve',31,'routed_preserve')
                train_recipe(m2,tok,data,cfg,root/'v6','routed_preserve',31)
            a=v5._load(root/'v5/latest.pt');b=v5._load(root/'v6/latest.pt')
            assert equal(a['deltas'],b['deltas']) and equal(a['optimizers'],b['optimizers']) and equal(a['rng'],b['rng'])
            assert (root/'v6/update_audits/00001.json').exists()
            report['v5_baseline_updates_unchanged_by_v6_auditing']=True
            m1.close();m2.close()

            # Uninterrupted mixture versus disconnect after a committed checkpoint,
            # restored into a completely different local directory.
            m1=new_module();m2=new_module();m3=new_module()
            with patch.object(v5,'_generate',side_effect=fake_generate):
                train_recipe(m1,tok,data,mix_cfg,root/'full','routed_preserve_mixture',31)
            expected=v5._load(root/'full/latest.pt')
            ws=Workspace(root/'local1',root/'durable',{'test':'nr6-resume'})
            interrupted=[False]
            def save_stop():
                ws.save_training('job',mix_cfg)
                if v5._load(ws.path('job/latest.pt'))['step']==1 and not interrupted[0]:
                    interrupted[0]=True;raise RuntimeError('Intentional disconnect')
            try:
                with patch.object(v5,'_generate',side_effect=fake_generate):
                    train_recipe(m2,tok,data,mix_cfg,ws.path('job'),'routed_preserve_mixture',31,save_stop)
            except RuntimeError as error:assert str(error)=='Intentional disconnect'
            else:raise AssertionError('Expected disconnect')
            ws2=Workspace(root/'local2',root/'durable',{'test':'nr6-resume'})
            assert ws2.restore_training('job',mix_cfg)
            with patch.object(v5,'_generate',side_effect=fake_generate):
                train_recipe(m3,tok,data,mix_cfg,ws2.path('job'),'routed_preserve_mixture',31,
                             lambda:ws2.save_training('job',mix_cfg))
            resumed=v5._load(ws2.path('job/latest.pt'))
            assert equal(expected['deltas'],resumed['deltas'])
            assert equal(expected['optimizers'],resumed['optimizers']) and equal(expected['rng'],resumed['rng'])
            with patch.object(Path,'rglob',side_effect=AssertionError('Recursive scan')):
                assert ws2.export().exists()
            source=root/'bytes';target=root/'target';source.write_bytes(b'new');target.write_bytes(b'old')
            def partial(src,dst):Path(dst).write_bytes(b'partial');raise OSError('copy interrupted')
            with patch('nr6_io.shutil.copyfile',side_effect=partial):
                try:atomic_copy(source,target)
                except OSError:pass
                else:raise AssertionError('Expected failed copy')
            assert target.read_bytes()==b'old'
            report['fresh_local_resume_restores_rng_adam_deltas_and_best']=True
            report['no_recursive_export_and_interrupted_copy_preserves_old_file']=True
            for m in (m1,m2,m3):m.close()
        e=examples[1]
        assert parse_work(' 1 -> 2 -> 3\nAnswer: 3\n')['states']==[1,2,3]
        assert parse_work(' I think 1 plus 2\nAnswer: 3') is None
        assert first_digit(' 3\nAnswer:')=='3' and first_digit(' 13') is None
        assert implied_hops(e,'3')==2
        for variant in ('standard','count_hint'):
            assert prompt_variant(e,variant)==prompt_variant(dict(e,answer='POISON',gold_trace=['POISON']),variant)
        report['own_prefix_parser_and_target_blind_count_hint']=True
        report.update(all_passed=True,seconds=time.perf_counter()-started,
            cuda_tested=torch.cuda.is_available(),
            scope='Tiny random models and scripted rewards validate mechanics only. No pretrained learning or A100 runtime claim.')
        if out is not None:v5.atomic_json(out,report)
        return report
    finally:torch.set_num_threads(previous_threads)
