"""V6 experiment runner: explicit I/O, two endpoints, and read-only V5 diagnostics."""
from dataclasses import asdict, replace
from pathlib import Path
from collections import defaultdict
import gc, hashlib, json, random, time, traceback
import pandas as pd
import torch
from nr4_model import NativeSlots
import nr5_rl as v5
from nr5_data import make_data
from nr5_tasks import format_prompt, pointer_map_key
from nr6_train import RECIPES, Config, train_recipe
from nr6_io import Workspace, digest, atomic_copy
from nr6_probes import fresh_pointer_examples, stopping_audit, coverage_audit, sampling_audit, summarize_stopping

MODEL_ID='Qwen/Qwen2.5-0.5B'
REVISION='060db6499f32faf8b98477b0a26969ef7d8b9987'
SOURCE_FILES=['rlq_tasks.py','nr4_model.py','nr4_model_checks.py','nr5_tasks.py',
    'nr5_data.py','nr5_data_checks.py','nr5_rl.py','nr5_rl_checks.py',
    'nr6_train.py','nr6_io.py','nr6_probes.py','nr6_run.py','nr6_checks.py']

def prepare(local,durable,cfg,seeds,recipes,families=12,smoke=False):
    assert seeds and len(set(seeds))==len(seeds) and all(isinstance(s,int) for s in seeds)
    assert recipes and len(set(recipes))==len(recipes) and set(recipes)<=set(RECIPES)
    cfg.validate()
    for recipe in recipes:replace(cfg,preservation_mode=RECIPES[recipe][1]).validate()
    data=make_data('mixed','smoke' if smoke else 'pilot')
    new_probes=fresh_pointer_examples(data,families)
    identity=dict(experiment='native-gradient-routing-v6-mechanisms-20261012',
        model_id=MODEL_ID,revision=REVISION,config=asdict(cfg),seeds=list(seeds),recipes=list(recipes),
        recipe_definitions=RECIPES,profile='smoke' if smoke else 'pilot',
        data_hash=digest(data),probe_hash=digest(new_probes),families=families,
        endpoints=['final','modularity_selected'],
        source_hashes={n:hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in SOURCE_FILES})
    # JSON normalization prevents tuple/list identity differences on resume.
    identity=json.loads(json.dumps(identity))
    ws=Workspace(local,durable,identity)
    for name in SOURCE_FILES:
        p=ws.path('source/'+name);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(Path(name).read_bytes())
        if not (ws.durable/'source'/name).exists():ws.publish('source/'+name)
        else:ws.artifacts.add('source/'+name)
    ws.write_json('dataset.json',data);ws.write_json('fresh_pointer_probes.json',new_probes)
    jobs=[]
    for seed in seeds:
        ordered=list(recipes);random.Random(seed+85391).shuffle(ordered)
        jobs.extend(dict(seed=seed,recipe=r,job=f'mixed/seed_{seed}/mlp_{r}') for r in ordered)
    ws.write_json('schedule.json',jobs)
    print('Ready. Planned fits:',len(jobs),'updates each:',cfg.max_steps,flush=True)
    print('Primary endpoint: final step. Secondary: original V5 development-selected checkpoint.',flush=True)
    return ws,data,new_probes,jobs

def load_model():
    from transformers import AutoTokenizer,AutoModelForCausalLM
    assert torch.cuda.is_available(),'Select an A100 GPU, or choose smoke mode.'
    assert torch.cuda.is_bf16_supported(),'This experiment uses BF16; choose an A100/L4 or another BF16 GPU.'
    torch.backends.cuda.matmul.allow_tf32=True
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=REVISION,padding_side='left',trust_remote_code=False)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=REVISION,
        torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,use_safetensors=True,
        attn_implementation='sdpa',trust_remote_code=False).to('cuda').eval()
    return model,tok

@torch.no_grad()
def last_logits(model,tok,examples):
    model.eval();result=[]
    for e in examples:
        enc=tok(format_prompt(e),add_special_tokens=False,return_tensors='pt',return_token_type_ids=False).to(next(model.parameters()).device)
        h=model.model(**enc,use_cache=False).last_hidden_state
        result.append(model.lm_head(h[:,-1]).float().cpu())
    return result

def evaluate(module,tok,data,cfg,ws,job,recipe,seed,endpoint,step):
    identity=dict(delta=module.delta_hash(),runtime=ws.runtime,data=digest({s:data[s] for s in ('test','depth_probe')}),
                  endpoint=endpoint,step=step,recipe=recipe,seed=seed)
    rel=f'{job}/evaluation/{endpoint}_{digest(identity)[:16]}.json'
    old=ws.pull(rel);cache=json.loads(old.read_text()) if old else dict(identity=identity,rows=[])
    assert cache['identity']==identity
    rows=cache['rows'];keys={(r['split'],r['mask'],r['id']) for r in rows}
    expected={(s,m,e['id']) for s in ('test','depth_probe') for m in v5.MASKS for e in data[s]}
    assert keys<=expected and len(keys)==len(rows)
    for split in ('test','depth_probe'):
        for mask in v5.MASKS:
            missing=[e for e in data[split] if (split,mask,e['id']) not in keys]
            for start in range(0,len(missing),cfg.eval_batch):
                group=missing[start:start+cfg.eval_batch]
                fresh=v5.predict(module,tok,group,mask,cfg,split,'mlp_'+recipe,seed)
                for r in fresh:r.update(recipe=recipe,endpoint=endpoint,checkpoint_step=step,runtime_key=ws.runtime_key)
                rows.extend(fresh)
                if ((start//cfg.eval_batch+1)%10==0) or start+len(group)==len(missing):
                    ws.write_json(rel,dict(identity=identity,rows=rows))
                    print(f'{recipe}/{endpoint}: {split}/{mask} {start+len(group)}/{len(missing)} new examples',flush=True)
    assert {(r['split'],r['mask'],r['id']) for r in rows}==expected
    ws.write_json(f'{job}/predictions_{endpoint}.json',dict(identity=identity,rows=rows))
    return rows

def metrics_table(rows):
    result=[]
    for split,task,depth in sorted({(r['split'],r['task_id'],r['difficulty']) for r in rows}):
        group=[r for r in rows if (r['split'],r['task_id'],r['difficulty'])==(split,task,depth)]
        masks={m:{r['id']:r for r in group if r['mask']==m} for m in v5.MASKS}
        ids=set(masks['none']);assert ids and all(set(x)==ids for x in masks.values())
        owner,foreign=('A','B') if task==0 else ('B','A')
        def changes(a,b):
            return dict(lost=sum(masks[a][i]['correct'] and not masks[b][i]['correct'] for i in ids),
                rescued=sum(not masks[a][i]['correct'] and masks[b][i]['correct'] for i in ids),
                changed=sum(masks[a][i]['answer']!=masks[b][i]['answer'] for i in ids))
        row=dict(split=split,task_id=task,depth=depth,n=len(ids),
            **{m+'_correct':sum(r['correct'] for r in masks[m].values()) for m in masks})
        row.update({'composition_'+k:v for k,v in changes(owner,'AB').items()})
        row.update({'foreign_'+k:v for k,v in changes('none',foreign).items()})
        row['owner_correct']=row[owner+'_correct'];row['foreign_correct']=row[foreign+'_correct']
        result.append(row)
    return result

def run_jobs(ws,data,probes,cfg,jobs,max_fits=0,diagnose_new=True,count_hint=True):
    finished_count=0
    for spec in jobs:
        job,recipe,seed=spec['job'],spec['recipe'],spec['seed']
        completed=ws.pull(f'{job}/complete.json')
        done = json.loads(completed.read_text()) if completed else {}
        needs_probes = diagnose_new and recipe in ('routed_preserve','routed_preserve_all','routed_preserve_mixture')
        if done.get('runtime')==ws.runtime and (not needs_probes or
                (done.get('mechanisms_complete') and done.get('count_hint')==count_hint)):
            print('Reusing completed fit and evaluations:',job,flush=True);continue
        if max_fits and finished_count>=max_fits:
            print('Session fit limit reached. Run this cell again with the same settings to continue.',flush=True);break
        fit_cfg=replace(cfg,preservation_mode=RECIPES[recipe][1])
        ws.restore_training(job,fit_cfg)
        model=module=None;started=time.perf_counter()
        try:
            print('Loading pinned pretrained model for',job,flush=True)
            model,tok=load_model();torch.cuda.reset_peak_memory_stats()
            witnesses=[next(e for e in data['dev'] if e['task_id']==t and e['difficulty']==d)
                       for t in (0,1) for d in (1,2,3)]
            raw=last_logits(model,tok,witnesses)
            module=NativeSlots(model,'mlp',seed=seed,n_layers=8,mlp_neurons=32)
            ws.write_json(f'{job}/slot_manifest.json',module.manifest())
            base_hash=module.base_hash()
            status=train_recipe(module,tok,data,fit_cfg,ws.path(job),recipe,seed,
                on_checkpoint=lambda:ws.save_training(job,fit_cfg))
            assert status['updates_with_movement']>0, 'No parameter update moved; do not count this as a learning result.'
            ws.publish(f'{job}/status.json')
            best=v5._load(ws.path(f'{job}/best.pt'));latest=v5._load(ws.path(f'{job}/latest.pt'))
            v5.atomic_torch(ws.path(f'{job}/final.pt'),dict(signature=latest['signature'],deltas=latest['deltas'],step=latest['step']))
            ws.publish(f'{job}/best.pt',evidence=False);ws.publish(f'{job}/final.pt',evidence=False)
            endpoint_rows={};audits={}
            for endpoint,saved in [('final',latest),('modularity_selected',best)]:
                module.load_state_dict(saved['deltas'])
                with module.gate(v5.MASKS['none']):off=last_logits(model,tok,witnesses)
                error=max(float((a-b).abs().max()) for a,b in zip(raw,off))
                assert error==0 and module.base_hash()==base_hash
                audits[endpoint]=dict(base_hash=base_hash,delta_hash=module.delta_hash(),
                    all_off_max_logit_diff=error,probe_count=len(witnesses),step=saved['step'],runtime=ws.runtime)
                if endpoint=='modularity_selected' and saved['step']==latest['step']:
                    assert audits['final']['delta_hash']==module.delta_hash()
                    copied=[dict(r,endpoint=endpoint) for r in endpoint_rows['final']]
                    identity=dict(endpoint=endpoint,step=saved['step'],delta=module.delta_hash(),runtime=ws.runtime,
                        recipe=recipe,seed=seed,reused_identical_final=True)
                    ws.write_json(f'{job}/predictions_{endpoint}.json',dict(identity=identity,rows=copied))
                    endpoint_rows[endpoint]=copied
                else:
                    endpoint_rows[endpoint]=evaluate(module,tok,data,fit_cfg,ws,job,recipe,seed,endpoint,saved['step'])
            ws.write_json(f'{job}/checkpoint_audits.json',audits)
            tables={end:metrics_table(rows) for end,rows in endpoint_rows.items()}
            ws.write_json(f'{job}/metrics.json',tables)
            # New diagnostics always use final-step weights, never choose a checkpoint.
            module.load_state_dict(latest['deltas'])
            if diagnose_new and recipe in ('routed_preserve','routed_preserve_all','routed_preserve_mixture'):
                namespace=digest(dict(delta=module.delta_hash(),runtime=ws.runtime,probes=digest(probes),count_hint=count_hint))[:16]
                rel=f'{job}/mechanisms/{namespace}'
                stops=stopping_audit(module,tok,probes,fit_cfg,ws,rel+'/stopping.json',count_hint)
                coverage_examples=witnesses+[e for e in probes if e['difficulty'] in (4,5)][:8]
                coverage_audit(module,tok,coverage_examples,fit_cfg,ws,rel+'/coverage.json')
                ws.write_json(f'{job}/mechanism_summary.json',dict(namespace=namespace,stopping=summarize_stopping(stops)))
            ws.write_json(f'{job}/complete.json',dict(runtime=ws.runtime,recipe=recipe,seed=seed,
                final_step=latest['step'],selected_step=best['step'],training_complete=True,
                mechanisms_complete=needs_probes,count_hint=count_hint,
                seconds_this_session=time.perf_counter()-started,
                peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30))
            finished_count+=1
            report(ws,jobs)
        except BaseException as error:
            ws.write_json(f'{job}/interruption.json',dict(type=type(error).__name__,message=str(error),
                traceback=traceback.format_exc(),runtime=ws.runtime,time=time.time()))
            raise
        finally:
            if module is not None:module.close()
            del module,model
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    return report(ws,jobs)

def report(ws,jobs):
    tables,training,roles,complete=[],[],[],[]
    for spec in jobs:
        job=spec['job'];done=ws.pull(f'{job}/complete.json')
        if not done or json.loads(done.read_text())['runtime']!=ws.runtime:continue
        complete.append(job)
        p=ws.pull(f'{job}/metrics.json');metrics=json.loads(p.read_text())
        for endpoint,rows in metrics.items():
            tables.extend(dict(r,endpoint=endpoint,recipe=spec['recipe'],seed=spec['seed']) for r in rows)
        hist=ws.pull(f'{job}/history.jsonl')
        if hist:
            training.extend(dict(json.loads(l),recipe=spec['recipe'],seed=spec['seed']) for l in hist.read_text().splitlines())
        for s in ws.identity['config']['audit_steps']:
            p=ws.pull(f'{job}/update_audits/{s:05d}.json')
            if not p:continue
            a=json.loads(p.read_text());previous=a['before']
            events=[e for e in a['events'] if a['accepted'] and e['attempt']==a['accepted_attempt']]
            for event in events:
                for key,value in event['nll'].items():
                    task,mask=key.split('/')
                    roles.append(dict(recipe=spec['recipe'],seed=spec['seed'],step=s,update_task=a['task_id'],
                        role=event['role'],probe_task=int(task),mask=mask,nll_before=previous[key],nll_after=value,
                        nll_change=value-previous[key]))
                previous=event['nll']
    files={'per_depth.csv':tables,'training_steps.csv':training,'accepted_role_effects.csv':roles}
    for name,rows in files.items():
        p=ws.path('report/'+name);p.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).to_csv(p,index=False);ws.publish('report/'+name)
    contrasts=[]
    lookup={(r['seed'],r['recipe'],r['endpoint'],r['task_id'],r['depth']):r for r in tables}
    for r in tables:
        if r['recipe'] not in ('routed_preserve_all','routed_preserve_mixture'):continue
        comparator='routed_preserve' if r['recipe']=='routed_preserve_all' else 'routed_preserve_all'
        key=(r['seed'],comparator,r['endpoint'],r['task_id'],r['depth'])
        if key not in lookup:continue
        base=lookup[key]
        contrasts.append(dict(seed=r['seed'],endpoint=r['endpoint'],task_id=r['task_id'],depth=r['depth'],
            recipe=r['recipe'],comparator=comparator,n=r['n'],
            AB_correct_difference=r['AB_correct']-base['AB_correct'],
            owner_correct_difference=r['owner_correct']-base['owner_correct'],
            composition_lost_difference=r['composition_lost']-base['composition_lost'],
            foreign_changed_difference=r['foreign_changed']-base['foreign_changed']))
    p=ws.path('report/coverage_comparisons.csv');pd.DataFrame(contrasts).to_csv(p,index=False);ws.publish('report/coverage_comparisons.csv')
    summary=dict(planned=len(jobs),complete=len(complete),completed_jobs=complete,
        interpretation='Descriptive counts and paired within-seed contrasts; no success/failure verdict or multiplicity-adjusted significance claim.',
        primary='final step',secondary='V5 modularity-selected checkpoint',
        caution='Same V5 task distribution; fresh fit seeds are replications, not an untouched task domain. More preservation coverage costs extra compute.')
    ws.write_json('report/status.json',summary)
    return summary

def audit_v5_logs(v5root,ws):
    root=Path(v5root);rows=[]
    for regime in ('fixed','mixed'):
        for rule in v5.RULES:
            p=root/regime/'seed_17'/('mlp_'+rule)
            if not (p/'dev_history.json').exists():continue
            hist=json.loads((p/'dev_history.json').read_text())
            status=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {}
            for h in hist:
                rows.append(dict(regime=regime,rule=rule,step=h['step'],selected=h['step']==status.get('selected_step'),
                    counts=h['counts'],selection_key=h['selection_key']))
            if (p/'history.jsonl').exists():
                training=[json.loads(l) for l in (p/'history.jsonl').read_text().splitlines()]
                ws.write_json(f'v5_logs/{regime}_{rule}_training.json',training)
    ws.write_json('v5_logs/development_trajectory.json',rows)
    return rows

def diagnose_v5(v5root,ws,probes,cfg,count_hint=True):
    """Read only explicit V5 paths; no recursive Drive discovery or writes to V5."""
    root=Path(v5root);manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['model_id']==MODEL_ID and manifest['revision']==REVISION
    assert manifest['source_hashes']['nr4_model.py']==hashlib.sha256(Path('nr4_model.py').read_bytes()).hexdigest(), 'Native slot code differs.'
    audit_v5_logs(root,ws)
    jobs=[('mixed','routed_preserve','best.pt','stopping'),
          ('fixed','routed','best.pt','stopping'),
          ('fixed','unrouted','latest.pt','sampling')]
    for regime,rule,filename,kind in jobs:
        source=root/regime/'seed_17'/('mlp_'+rule)/filename
        if not source.exists():
            print('Skipping unavailable V5 tensor checkpoint:',source,flush=True);continue
        done_rel=f'v5_diagnostics/{regime}_{rule}/current.json'
        print('Staging saved V5 checkpoint:',source,flush=True)
        staged=ws.path(f'imports/{regime}_{rule}_{filename}');atomic_copy(source,staged)
        saved=v5._load(staged);signature=saved['signature'];layout=signature['manifest']
        source_data=json.loads((root/'datasets'/f'{regime}.json').read_text())
        old_maps={pointer_map_key(e) for split in ('train','dev','test','depth_probe','diagnostic')
                  for e in source_data[split] if e['task_id']==1}
        assert not old_maps.intersection(pointer_map_key(e) for e in probes), 'Diagnostic maps overlap the source run.'
        identity=dict(source_tensor_shapes_hash=digest({k:list(v.shape) for k,v in saved['deltas'].items()}),
                      source_step=saved['step'],source_file=filename,source_run=str(root))
        model=module=None
        try:
            model,tok=load_model()
            module=NativeSlots(model,layout['architecture'],seed=layout['seed'],n_layers=len(layout['layers']),
                mlp_neurons=layout['requested_mixed_mlp_neurons'])
            assert module.manifest()==layout and module.base_hash()==signature['base_hash']
            module.load_state_dict(saved['deltas'])
            identity.update(delta_hash=module.delta_hash(),runtime=ws.runtime,probes=digest(probes),count_hint=count_hint)
            namespace=digest(identity)[:16];prefix=f'v5_diagnostics/{regime}_{rule}/{namespace}'
            if kind=='stopping':
                stops=stopping_audit(module,tok,probes,cfg,ws,prefix+'/stopping.json',count_hint)
                witnesses=[next(e for e in source_data['dev'] if e['task_id']==t and e['difficulty']==d)
                           for t in (0,1) for d in (1,2,3)]
                coverage_audit(module,tok,witnesses+[e for e in probes if e['difficulty'] in (4,5)][:8],cfg,ws,prefix+'/coverage.json')
                ws.write_json(prefix+'/summary.json',dict(identity=identity,stopping=summarize_stopping(stops)))
            else:
                examples=sorted([e for e in source_data['train'] if e['task_id']==1],key=lambda e:e['id'])
                random.Random(20261012).shuffle(examples)
                sampling_audit(module,tok,examples[:12],cfg,ws,prefix+'/sampling.json')
            ws.write_json(done_rel,dict(namespace=namespace,identity=identity))
            assert module.delta_hash()==identity['delta_hash']
        finally:
            if module is not None:module.close()
            del module,model,saved
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
