"""Offline checks of RL plumbing; synthetic tokens/rewards are never efficacy evidence."""
from pathlib import Path
from unittest import mock
import copy, json, tempfile
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.decoders import Fuse
import nr5_rl as rl
from nr4_model import NativeSlots
from nr4_model_checks import run_checks as model_checks
from nr5_tasks import format_prompt
from rlq_tasks import _example, _arithmetic_problem, _pointer_problem


def fixture():
    examples = [_example(0, 1, _arithmetic_problem(1, [('add', 2)])),
                _example(1, 2, _pointer_problem(1, {i:(i+1)%10 for i in range(10)}, 2))]
    alphabet = sorted(set(''.join(format_prompt(e) for e in examples) + '0123456789\n ->Answer:'))
    vocab = {'<pad>':0, '<eos>':1, '<unk>':2} | {c:i+3 for i,c in enumerate(alphabet)}
    backend = Tokenizer(WordLevel(vocab, unk_token='<unk>'))
    backend.pre_tokenizer = Split('', behavior='isolated'); backend.decoder = Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>', eos_token='<eos>',
                                  unk_token='<unk>', padding_side='left')
    torch.manual_seed(1349)
    config = Qwen2Config(vocab_size=len(tok), hidden_size=16, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048, attention_dropout=0.,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    config._attn_implementation = 'sdpa'
    model = Qwen2ForCausalLM(config).float().eval()
    return model, tok, examples


def scripted_items(tok, examples):
    items = []
    for i,e in enumerate(examples):
        answer = e['answer'] if i%2 == 0 else str((int(e['answer'])+1)%10)
        text = ' ' + ' -> '.join(map(str, e['gold_trace'])) + '\nAnswer: ' + answer + '\n'
        ids = tok(text, add_special_tokens=False).input_ids
        assert tok.decode(ids, skip_special_tokens=True) == text
        items.append(dict(example=e, prompt_ids=tok(format_prompt(e), add_special_tokens=False).input_ids,
            generated_ids=ids, completion=text, answer=answer, valid=True,
            correct=answer==e['answer'], truncated=False))
    return items


def run_checks(outdir=None):
    before_threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        report = {}
        base,tok,examples = fixture()
        cfg = rl.Config(max_steps=2, prompts_per_update=1, group_size=2, generation_batch=2,
            max_prompt_tokens=2048, max_new_tokens=3, eval_every=1, save_every=1,
            eval_batch=2, lr=1e-3, min_lr=1e-5, max_mean_kl=100., max_sequence_kl=100.,
            checkpointing=True, kl_positions=2, head_chunk=8, clip_grad=1e6, snapshot_every=1)
        # Teacher-forced probabilities must match the native causal-LM head exactly.
        module = NativeSlots(copy.deepcopy(base), 'attn_mlp', n_layers=2, mlp_neurons=3)
        items = scripted_items(tok, [examples[0]]*2)
        batch = rl._training_batch(items, tok.pad_token_id, module.device, cfg)
        with torch.no_grad(), module.gate((True,True)):
            seq, _ = rl._scores(module,batch,cfg)
            logits = module.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                position_ids=batch['position_ids'], use_cache=False).logits.float().log_softmax(-1)
            p=batch['positions']; picked=logits[p[:,0],p[:,1],batch['targets']]
            expected=torch.zeros(2).index_add(0,batch['owners'],picked)
            assert torch.allclose(seq,expected,atol=1e-5,rtol=1e-5)
        module.close()
        report['completion_logprob_alignment'] = True
        results, recorded = {}, {}
        for rule in rl.RULES:
            module = NativeSlots(copy.deepcopy(base),'attn_mlp',n_layers=2,mlp_neurons=3)
            torch.manual_seed(456)
            with torch.no_grad():
                for k in (0,1):
                    for parameter in module.parameters(k):
                        parameter.normal_(mean=0.,std=.01)
            snapshots = [[p.detach().clone() for p in module.parameters(k)] for k in (0,1)]
            roles = (0,1,0,1)
            opts = [torch.optim.AdamW(module.parameters(k),lr=cfg.lr,weight_decay=0.) for k in roles]
            captured = {}
            for role, optimizer in enumerate(opts):
                original_step = optimizer.step
                def record_step(closure=None, role=role, optimizer=optimizer, original_step=original_step):
                    captured[role] = [None if p.grad is None else p.grad.detach().clone()
                                      for group in optimizer.param_groups for p in group['params']]
                    return original_step(closure)
                optimizer.step = record_step
            metrics = rl.update(module,tok,items,[1.,-1.],cfg,opts,0,rule)
            changed=[any(not torch.equal(p,b) for p,b in zip(module.parameters(k),snapshots[k])) for k in (0,1)]
            assert metrics['accepted'] and changed[0]
            assert changed[1] == (rule!='routed'), (rule,changed)
            assert module.assert_frozen_base()
            results[rule] = dict(accepted=True,changed_slots=changed,
                                preservation_kl=metrics.get('preservation_kl',0.),
                                neutral_kl=metrics.get('neutral_kl',0.))
            recorded[rule] = captured
            assert set(captured) == ({0,1,3} if rule=='unrouted_preserve' else
                                    {0,3} if rule=='routed_preserve' else
                                    {0,1} if rule=='unrouted' else {0}), (rule, list(captured))
            module.close()
        report['reward_and_preservation_gradient_ownership'] = results
        # Same pre-update state => same reward gradients across preservation toggle;
        # the foreign preservation gradient also matches its routed counterpart.
        def same_grads(a,b):
            return all((x is None and y is None) or
                       (x is not None and y is not None and torch.equal(x,y)) for x,y in zip(a,b))
        for role in (0,1):
            assert same_grads(recorded['unrouted'][role], recorded['unrouted_preserve'][role])
        assert same_grads(recorded['routed_preserve'][3], recorded['unrouted_preserve'][3])
        assert not same_grads(recorded['unrouted_preserve'][1], recorded['unrouted_preserve'][3])
        report['foreign_reward_and_preservation_gradients_never_mixed'] = True

        def equal_nested(a,b):
            if torch.is_tensor(a): return torch.equal(a,b)
            if isinstance(a,dict): return a.keys()==b.keys() and all(equal_nested(a[k],b[k]) for k in a)
            if isinstance(a,(tuple,list)): return len(a)==len(b) and all(equal_nested(x,y) for x,y in zip(a,b))
            return a==b
        module=NativeSlots(copy.deepcopy(base),'attn_mlp',n_layers=2,mlp_neurons=3)
        with torch.no_grad():
            for p in module.parameters(): p.normal_(std=.01)
        opts=[torch.optim.AdamW(module.parameters(k),lr=cfg.lr,weight_decay=0.) for k in (0,1,0,1)]
        assert rl.update(module,tok,items,[1.,-1.],cfg,opts,0,'unrouted_preserve')['accepted']
        saved_params=module.state_dict()
        saved_states=copy.deepcopy([o.state_dict() for o in opts])
        reject_cfg=copy.deepcopy(cfg); reject_cfg.max_mean_kl=1e-20; reject_cfg.max_sequence_kl=1e-20
        reject_cfg.backtracks=0
        rejected=rl.update(module,tok,items,[1.,-1.],reject_cfg,opts,1,'unrouted_preserve')
        assert not rejected['accepted'], rejected
        assert equal_nested(saved_params,module.state_dict())
        assert equal_nested(saved_states,[o.state_dict() for o in opts])
        module.close()
        report['rejected_update_restores_parameters_and_all_adam_roles']=True

        # A pooled near-ceiling result cannot hide an inadequate depth stratum.
        dev=[]
        for task in (0,1):
            for depth in (1,2):
                for mask in rl.MASKS:
                    for i in range(10):
                        dev.append(dict(task_id=task,difficulty=depth,mask=mask,
                                        correct=(i<2 if mask=='none' else depth==1)))
        summary=rl._dev_summary(dev,cfg)
        assert summary['task_adequacy']==[False,False]
        assert len(summary['strata'])==4 and summary['selection_key'][0]==0
        pool=[dict(difficulty=d,id=str(d)) for d in (1,2,3)]
        depth_draws=[]; rng=rl.random.Random(17)
        for step in range(3):
            depth_draws += [e['difficulty'] for e in rl._sample_prompts(pool,step,cfg,rng)]
        assert [depth_draws.count(d) for d in (1,2,3)]==[1,1,1]
        report['stratified_development_and_uniform_depth_sampling']=True

        def fake_generate(module,tok,group,cfg,mask='AB',sample=False):
            return scripted_items(tok,group)

        # A saved mid-run interruption must resume to the same tensors/Adam step.
        with tempfile.TemporaryDirectory(prefix='nr5_smoke_') as name:
            root=Path(name); data={'train':examples,'dev':examples}
            m1=NativeSlots(copy.deepcopy(base),'attn_mlp',n_layers=2,mlp_neurons=3)
            with mock.patch.object(rl,'_generate',side_effect=fake_generate):
                status1=rl.train_arm(m1,tok,data,cfg,root/'uninterrupted','attn_mlp_unrouted_preserve',17,'unrouted_preserve')
            reference=rl._load(root/'uninterrupted/latest.pt')
            assert (root/'uninterrupted/snapshots/step_00001.pt').exists()
            assert status1['complete'] and status1['updates_with_movement']>0
            m2=NativeSlots(copy.deepcopy(base),'attn_mlp',n_layers=2,mlp_neurons=3)
            interrupted=[False]
            def stop_after_first_step():
                saved=rl._load(root/'resumed/latest.pt')
                if saved['step']==1 and not interrupted[0]:
                    interrupted[0]=True
                    raise RuntimeError('Synthetic interruption after committed checkpoint')
            try:
                with mock.patch.object(rl,'_generate',side_effect=fake_generate):
                    rl.train_arm(m2,tok,data,cfg,root/'resumed','attn_mlp_unrouted_preserve',17,'unrouted_preserve',on_checkpoint=stop_after_first_step)
            except RuntimeError as error:
                assert str(error).startswith('Synthetic interruption')
            else:
                raise AssertionError('No simulated interruption')
            with mock.patch.object(rl,'_generate',side_effect=fake_generate):
                status2=rl.train_arm(m2,tok,data,cfg,root/'resumed','attn_mlp_unrouted_preserve',17,'unrouted_preserve')
            resumed=rl._load(root/'resumed/latest.pt')
            assert all(torch.equal(reference['deltas'][k],resumed['deltas'][k]) for k in reference['deltas'])
            assert status1['selected_delta_hash']==status2['selected_delta_hash']
            assert equal_nested(reference['optimizers'],resumed['optimizers'])
            with mock.patch.object(rl,'_generate',side_effect=AssertionError('Completed arm trained twice')):
                rl.train_arm(m2,tok,data,cfg,root/'resumed','attn_mlp_unrouted_preserve',17,'unrouted_preserve')
            m1.close();m2.close()
        report['safe_checkpoint_interruption_resume_and_completed_skip'] = True
        # Exercise real generate/stopping on random logits; there is no accuracy claim.
        module=NativeSlots(copy.deepcopy(base),'attn',n_layers=2,mlp_neurons=3)
        generated=rl.predict(module,tok,examples,'AB',cfg,'smoke','attn_routed',17)
        assert len(generated)==2 and all(type(r['correct']) is bool for r in generated)
        module.close()
        report['actual_generation_smoke'] = True
        report['all_passed']=True
        report['scope']='Offline tiny random models, synthetic rewards, and mechanics only. No pretrained-model learning, A100 memory, or efficacy result.'
        if outdir is not None:
            out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
            rl.atomic_json(out/'checks.json',report)
        return report
    finally:
        torch.set_num_threads(before_threads)
