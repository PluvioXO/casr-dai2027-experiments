"""Read-only, explicitly assisted stopping tests and intervention-state KL audits."""
from collections import Counter
from dataclasses import replace
import re, random
import torch
import nr5_rl as v5
from nr5_tasks import format_prompt, pointer_parts, pointer_map_key, build_pointer_counterfactuals, extract_answer
from nr6_io import digest

def fresh_pointer_examples(data, families=12):
    excluded = {pointer_map_key(e) for s in ('train','dev','test','depth_probe','diagnostic')
                for e in data[s] if e['task_id'] == 1}
    built = build_pointer_counterfactuals(seed=20261012, nfamilies=families,
        depths=(1,2,3,4,5,6), excluded_maps=excluded)
    return [dict(e, answer=built['labels'][e['id']]['answer'],
                 gold_trace=built['labels'][e['id']]['gold_trace']) for e in built['examples']]

def parse_work(completion):
    """Use the model's own text; never fill or repair a missing state."""
    match = re.search(r'\n[ \t]*Answer:', completion, flags=re.I)
    if not match: return None
    raw = completion[:match.start()].rstrip()
    work = re.sub(r'^\s*Work:\s*', '', raw).strip()
    if not re.fullmatch(r'[0-9](?:\s*(?:->|→)\s*[0-9])+', work): return None
    return dict(raw=raw, states=[int(c) for c in re.findall('[0-9]', work)])

def first_digit(text):
    match = re.match(r'^\s*([0-9])(?=\D|$)', text)
    return match.group(1) if match else None

def implied_hops(example, answer):
    if answer is None: return None
    start, mapping = pointer_parts(example); state = start
    for k in range(10):
        if str(state) == answer: return k
        state = mapping[state]
    raise AssertionError('Expected a ten-cycle and a single-digit answer')

def prompt_variant(e, variant):
    prompt = format_prompt(e)
    if variant == 'standard': return prompt
    if variant != 'count_hint': raise ValueError(variant)
    # Uses only the public requested number, not an answer or intermediate state.
    k = int(re.search(r'Follow exactly ([0-9]+) links\.', e['problem'])[1])
    hint = f'Count links, not states: {k} links require {k+1} states including the starting digit.\n'
    return hint + prompt

@torch.no_grad()
def generate_text(module, tok, prompts, cfg, mask, max_new_tokens=None, stop_answer=True):
    module.model.eval()
    enc = tok(prompts, padding=True, add_special_tokens=False, return_tensors='pt',
              return_token_type_ids=False).to(module.device)
    width = enc.input_ids.shape[1]
    assert width <= cfg.max_prompt_tokens, 'Diagnostic prompt too long; never truncate.'
    stop = v5.AnswerStop(tok, width, len(prompts))
    kwargs = dict(max_new_tokens=max_new_tokens or cfg.max_new_tokens, do_sample=False,
        temperature=None, top_p=None, top_k=None, use_cache=True,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    if stop_answer: kwargs['stopping_criteria'] = v5.StoppingCriteriaList([stop])
    with module.gate(v5.MASKS[mask]): out = module.model.generate(**enc, **kwargs)
    rows = []
    for i in range(len(prompts)):
        n = stop.lengths[i] if stop_answer else None
        n = n or out.shape[1]-width
        ids = out[i, width:width+n].detach().cpu().tolist()
        rows.append(dict(completion=tok.decode(ids, skip_special_tokens=True), tokens=len(ids),
            truncated=stop_answer and stop.lengths[i] is None))
    return rows

def stopping_audit(module, tok, examples, cfg, ws, rel, count_hint=True):
    identity = dict(delta=module.delta_hash(), runtime=ws.runtime, data=digest(examples), count_hint=count_hint)
    loaded = ws.pull(rel)
    cache = __import__('json').loads(loaded.read_text()) if loaded else dict(identity=identity, rows=[])
    assert cache['identity'] == identity, 'Diagnostic cache identity mismatch'
    rows = cache['rows']; seen = {(r['id'],r['mask'],r['variant']) for r in rows}
    for mask in ('B','AB'):
        for variant in (('standard','count_hint') if count_hint else ('standard',)):
            missing = [e for e in examples if (e['id'],mask,variant) not in seen]
            for start in range(0,len(missing),cfg.eval_batch):
                group = missing[start:start+cfg.eval_batch]
                outputs = generate_text(module,tok,[prompt_variant(e,variant) for e in group],cfg,mask)
                pending, forced_prompts = [], []
                for e, output in zip(group,outputs):
                    parsed = parse_work(output['completion']); answer = extract_answer(output['completion'])
                    row = dict(id=e['id'],family_id=e['family_id'],depth=e['difficulty'],mask=mask,variant=variant,
                        answer=answer,gold=e['answer'],correct=answer==e['answer'],
                        implied_hops=implied_hops(e,answer),**output, trace_parseable=parsed is not None,
                        observed_hops=None,correct_prefix=False,premature_correct_prefix=False,
                        forced_next_digit=None,forced_next_correct=None,forced_text=None,
                        supplied_correct_state=False)
                    if parsed:
                        states = parsed['states']; _, mapping = pointer_parts(e)
                        h = len(states)-1
                        prefix_correct = h < len(e['gold_trace']) and states == e['gold_trace'][:len(states)]
                        row.update(observed_hops=h,correct_prefix=prefix_correct,
                            premature_correct_prefix=bool(prefix_correct and h<e['difficulty']),
                            local_next_gold=str(mapping[states[-1]]))
                        # Force only an arrow at its natural answer boundary. The entire
                        # preceding trace was generated by this same mask and prompt.
                        forced_prompts.append(prompt_variant(e,variant)+parsed['raw']+' ->')
                        pending.append(row)
                    rows.append(row)
                if pending:
                    extra = generate_text(module,tok,forced_prompts,cfg,mask,max_new_tokens=8,stop_answer=False)
                    for row, output in zip(pending,extra):
                        digit = first_digit(output['completion'])
                        row.update(forced_next_digit=digit,forced_next_correct=digit==row['local_next_gold'],
                                   forced_text=output['completion'])
                ws.write_json(rel,dict(identity=identity,rows=rows))
                print(f'Stopping audit {mask}/{variant}: {min(start+len(group),len(missing))}/{len(missing)} new cases',flush=True)
    assert module.delta_hash() == identity['delta']
    return rows

@torch.no_grad()
def prefix_kl_values(module, batch, teacher_mask, student_mask, chunk=32):
    p = batch['positions'][batch['kl_indices']]
    module.model.eval()
    with module.gate(teacher_mask): ht = v5._hidden(module,batch)[p[:,0],p[:,1]]
    with module.gate(student_mask): hs = v5._hidden(module,batch)[p[:,0],p[:,1]]
    values, changed = [], []
    for start in range(0,len(p),chunk):
        lt = module.model.lm_head(ht[start:start+chunk]).float().log_softmax(-1)
        ls = module.model.lm_head(hs[start:start+chunk]).float().log_softmax(-1)
        values.extend(v5._forward_kl(lt,ls).cpu().tolist())
        changed.extend((lt.argmax(-1)!=ls.argmax(-1)).cpu().tolist())
    return values, changed

def coverage_audit(module,tok,examples,cfg,ws,rel):
    identity = dict(delta=module.delta_hash(),runtime=ws.runtime,data=digest(examples))
    loaded=ws.pull(rel)
    cache=__import__('json').loads(loaded.read_text()) if loaded else dict(identity=identity,rows=[])
    assert cache['identity']==identity
    rows=cache['rows'];seen={(r['id'],r['prefix_mask'],r['comparison'],r['position_scope']) for r in rows}
    for task in (0,1):
        owner,foreign=('A','B') if task==0 else ('B','A')
        group=[e for e in examples if e['task_id']==task]
        for source in v5.MASKS:
            missing=[e for e in group if (e['id'],source,'composition','all') not in seen]
            for start in range(0,len(missing),cfg.generation_batch):
                subset=missing[start:start+cfg.generation_batch]
                items=v5._generate(module,tok,subset,cfg,source,sample=False)
                batch=v5._training_batch(items,tok.pad_token_id,module.device,
                    replace(cfg,kl_positions=max(len(i['generated_ids']) for i in items)))
                old=v5._training_batch(items,tok.pad_token_id,module.device,cfg)
                four=set(old['kl_indices'].cpu().tolist())
                owners=batch['owners'].cpu().tolist()
                for name,tmask,smask in [('composition',owner,'AB'),('neutrality','none',foreign)]:
                    values,changed=prefix_kl_values(module,batch,v5.MASKS[tmask],v5.MASKS[smask],cfg.kl_head_chunk)
                    for i,e in enumerate(subset):
                        for scope in ('all','v5_four'):
                            ix=[j for j,o in enumerate(owners) if o==i and (scope=='all' or j in four)]
                            rows.append(dict(id=e['id'],task_id=task,depth=e['difficulty'],prefix_mask=source,
                                comparison=name,teacher_mask=tmask,student_mask=smask,position_scope=scope,
                                n_positions=len(ix),mean_kl=sum(values[j] for j in ix)/len(ix),
                                max_token_kl=max(values[j] for j in ix),
                                greedy_token_change=sum(changed[j] for j in ix)/len(ix),
                                prefix_completion=items[i]['completion']))
                ws.write_json(rel,dict(identity=identity,rows=rows))
                print(f'Coverage audit task {task}, prefixes from {source}: {start+len(subset)}/{len(missing)}',flush=True)
    assert module.delta_hash()==identity['delta']
    return rows

def sampling_audit(module,tok,examples,cfg,ws,rel,draws=16):
    assert draws==16
    identity=dict(delta=module.delta_hash(),runtime=ws.runtime,data=digest(examples),draws=draws)
    loaded=ws.pull(rel)
    cache=__import__('json').loads(loaded.read_text()) if loaded else dict(identity=identity,rows=[])
    assert cache['identity']==identity
    rows=cache['rows'];seen={r['id'] for r in rows}
    devices=[module.device.index or 0] if module.device.type=='cuda' else []
    for e in examples:
        if e['id'] in seen:continue
        sampled=[]
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(digest({'id':e['id'],'probe':20261012})[:12],16))
            for start in range(0,draws,cfg.generation_batch):
                sampled.extend(v5._generate(module,tok,[e]*min(cfg.generation_batch,draws-start),cfg,'AB',sample=True))
        rewards=[int(i['correct']) for i in sampled]
        rows.append(dict(id=e['id'],task_id=e['task_id'],depth=e['difficulty'],rewards=rewards,
            successes=sum(rewards),draws=draws,
            diverse_groups={str(g):sum(min(rewards[j:j+g])!=max(rewards[j:j+g])
                for j in range(0,draws,g)) for g in (2,4,8,16)},
            completions=[i['completion'] for i in sampled],
            scope='Frozen-policy sampling diagnostic; no temperature change or learning.'))
        ws.write_json(rel,dict(identity=identity,rows=rows))
    assert module.delta_hash()==identity['delta']
    return rows

def summarize_stopping(rows):
    result=[]
    for mask,variant,depth in sorted({(r['mask'],r['variant'],r['depth']) for r in rows}):
        q=[r for r in rows if (r['mask'],r['variant'],r['depth'])==(mask,variant,depth)]
        premature=[r for r in q if r['premature_correct_prefix']]
        result.append(dict(mask=mask,variant=variant,depth=depth,n=len(q),correct=sum(r['correct'] for r in q),
            trace_parseable=sum(r['trace_parseable'] for r in q),premature_correct_prefix=len(premature),
            forced_next_correct_given_premature=sum(r['forced_next_correct'] for r in premature),
            implied_hops=dict(Counter(str(r['implied_hops']) for r in q))))
    return result
