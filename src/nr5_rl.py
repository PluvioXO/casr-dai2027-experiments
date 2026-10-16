"""Native-slot RL: terminal-reward RLOO, conservative updates, resumable pilots.

Both slots participate in every training rollout. Routing restricts parameter
updates, never the forward computation or gradient transport through the base.
No test or depth-probe example is used for learning or checkpoint selection.
The regularizer is exact conditional forward KL at sampled prefix states, averaged
within each sequence and then across sequences. Prefixes are held fixed in that
gradient; this is a local conditional-KL surrogate, not an unbiased derivative of
full trajectory KL. It is not a sampled-token k3 approximation.
Step backtracking tests old||new at those same sampled states; it is an empirical
local check, not a global policy trust-region or state-distribution guarantee.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from transformers import StoppingCriteria, StoppingCriteriaList

from nr5_tasks import format_prompt, extract_answer

MASKS = {'none': (False, False), 'A': (True, False), 'B': (False, True), 'AB': (True, True)}
RULES = ('unrouted', 'routed', 'unrouted_preserve', 'routed_preserve')


@dataclass
class Config:
    max_steps: int = 320
    lr: float = 1e-4
    min_lr: float = 1e-6
    prompts_per_update: int = 4
    group_size: int = 4
    generation_batch: int = 8
    max_new_tokens: int = 64
    max_prompt_tokens: int = 512
    eval_every: int = 80
    save_every: int = 5
    eval_batch: int = 8
    reference_kl: float = 0.02
    max_mean_kl: float = 0.02
    max_sequence_kl: float = 0.1
    clip_grad: float = 1.
    backtracks: int = 3
    checkpointing: bool = True
    kl_positions: int = 4
    head_chunk: int = 8
    preservation_coefficient: float = 1.0
    neutral_coefficient: float = 1.0
    stratify_depth: bool = True
    snapshot_every: int = 0
    selection_gain: float = 0.15
    selection_ceiling: float = 0.90

    def validate(self):
        for name in ('max_steps', 'prompts_per_update', 'generation_batch', 'max_new_tokens',
                     'max_prompt_tokens', 'eval_every', 'save_every', 'eval_batch', 'kl_positions', 'head_chunk'):
            assert isinstance(getattr(self, name), int) and getattr(self, name) > 0, name
        assert self.group_size >= 2 and self.backtracks >= 0
        assert 0 < self.min_lr <= self.lr and self.clip_grad > 0
        assert self.reference_kl >= 0 and self.max_mean_kl > 0 and self.max_sequence_kl > 0
        assert self.preservation_coefficient >= 0 and self.neutral_coefficient >= 0
        assert 0 <= self.selection_gain <= 1 and 0 < self.selection_ceiling <= 1
        assert isinstance(self.snapshot_every, int) and self.snapshot_every >= 0
        return self


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(v) for v in value)
    if isinstance(value, list):
        return [_cpu(v) for v in value]
    return value


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(_cpu(value), tmp)
    os.replace(tmp, path)


def _load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class AnswerStop(StoppingCriteria):
    """Stop only after a complete valid Answer line, or EOS; never on a digit prefix."""
    def __init__(self, tok, start, count):
        self.tok, self.start = tok, start
        self.lengths = [None] * count

    def __call__(self, ids, scores, **kwargs):
        eos = self.tok.eos_token_id
        for i, row in enumerate(ids):
            if self.lengths[i] is None:
                text = self.tok.decode(row[self.start:], skip_special_tokens=True)
                if row[-1].item() == eos or (text.endswith('\n') and extract_answer(text) is not None):
                    self.lengths[i] = len(row) - self.start
        return torch.tensor([n is not None for n in self.lengths], device=ids.device)


def _trace(text, gold):
    work = re.split(r'\n[ \t]*Answer:', text, maxsplit=1, flags=re.I)[0].strip()
    work = re.sub(r'^Work:\s*', '', work, flags=re.I)
    strict = bool(re.fullmatch(r'[0-9](?:\s*(?:->|→)\s*[0-9])+', work))
    digits = [int(x) for x in re.findall(r'[0-9]', work)] if strict else []
    return {'trace_hops': len(digits) - 1 if strict else None,
            'first_hop_correct': bool(strict and len(digits) >= 2 and digits[:2] == list(gold)[:2]),
            'trace_correct': bool(strict and digits == list(gold))}


def _generate(module, tok, examples, cfg, mask='AB', sample=False):
    assert tok.padding_side == 'left', 'Generation must use left padding.'
    module.model.eval()
    enc = tok([format_prompt(e) for e in examples], padding=True, add_special_tokens=False,
              return_tensors='pt', return_token_type_ids=False).to(module.device)
    width = enc.input_ids.shape[1]
    assert width <= cfg.max_prompt_tokens, 'Prompt exceeds limit; never truncate a task.'
    stop = AnswerStop(tok, width, len(examples))
    kw = dict(max_new_tokens=cfg.max_new_tokens, do_sample=sample, use_cache=True,
              repetition_penalty=1., eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
              stopping_criteria=StoppingCriteriaList([stop]))
    if sample:
        kw.update(temperature=1., top_p=1., top_k=0)
    else:
        kw.update(temperature=None, top_p=None, top_k=None)
    with torch.no_grad(), module.gate(MASKS[mask]):
        outputs = module.model.generate(**enc, **kw)
    result = []
    for i, ex in enumerate(examples):
        n = stop.lengths[i] or (outputs.shape[1] - width)
        generated = outputs[i, width:width+n].detach().cpu().tolist()
        prompt = enc.input_ids[i][enc.attention_mask[i].bool()].detach().cpu().tolist()
        text = tok.decode(generated, skip_special_tokens=True)
        answer = extract_answer(text)
        result.append(dict(example=ex, prompt_ids=prompt, generated_ids=generated,
                           completion=text, answer=answer, valid=answer is not None,
                           correct=answer == str(ex['answer']), truncated=stop.lengths[i] is None))
    return result


def predict(module, tok, examples, mask_name, cfg, split, arm, seed):
    """Greedy, fresh-cache evaluation; gates stay fixed for the entire completion."""
    assert mask_name in MASKS
    rows = []
    for start in range(0, len(examples), cfg.eval_batch):
        for item in _generate(module, tok, examples[start:start+cfg.eval_batch], cfg, mask_name):
            ex = item['example']
            row = {k: item[k] for k in ('answer', 'correct', 'valid', 'completion', 'truncated')}
            row.update(arm=arm, seed=int(seed), split=split, task_id=int(ex['task_id']), id=ex['id'],
                       difficulty=int(ex['difficulty']), mask=mask_name, gold=str(ex['answer']),
                       tokens=len(item['generated_ids']))
            row.update(_trace(item['completion'], ex['gold_trace']))
            rows.append(row)
    return rows


def _training_batch(items, pad, device, cfg):
    """Pack sampled completions without changing any sampled token or prompt."""
    lengths = [len(i['prompt_ids']) + len(i['generated_ids']) for i in items]
    assert all(i['generated_ids'] for i in items)
    width = max(lengths)
    ids = torch.full((len(items), width), pad, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    positions, owners, targets, kl_indices = [], [], [], []
    for i, item in enumerate(items):
        shift = width - lengths[i]
        sequence = item['prompt_ids'] + item['generated_ids']
        ids[i, shift:] = torch.tensor(sequence, device=device)
        mask[i, shift:] = 1
        first = shift + len(item['prompt_ids']) - 1
        n = len(item['generated_ids'])
        # Fixed evenly spaced positions depend on token count, never reward.
        chosen = set(np.linspace(0, n-1, min(n, cfg.kl_positions), dtype=int).tolist())
        for j, target in enumerate(item['generated_ids']):
            if j in chosen:
                kl_indices.append(len(positions))
            positions.append((i, first+j)); owners.append(i); targets.append(target)
    return dict(input_ids=ids, attention_mask=mask,
                position_ids=(mask.cumsum(-1)-1).clamp_min(0),
                positions=torch.tensor(positions, dtype=torch.long, device=device),
                owners=torch.tensor(owners, dtype=torch.long, device=device),
                targets=torch.tensor(targets, dtype=torch.long, device=device),
                kl_indices=torch.tensor(kl_indices, dtype=torch.long, device=device),
                count=len(items))


def _hidden(module, batch):
    return module.model.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                              position_ids=batch['position_ids'], use_cache=False).last_hidden_state


def _chosen_log_probs(module, batch):
    """Small full-vocabulary distributions at the predeclared sampled positions."""
    h = _hidden(module, batch)
    p = batch['positions'][batch['kl_indices']]
    selected = h[p[:, 0], p[:, 1]]
    return torch.log_softmax(module.model.lm_head(selected).float(), dim=-1)


def _scores(module, batch, cfg):
    h = _hidden(module, batch)
    p = batch['positions']
    selected = h[p[:, 0], p[:, 1]]
    logps = []
    def token_score(hidden, targets):
        log_dist = torch.log_softmax(module.model.lm_head(hidden).float(), dim=-1)
        return log_dist.gather(-1, targets[:, None]).squeeze(-1)
    # Checkpointing saves hidden inputs, not an entire response×vocabulary graph.
    for start in range(0, len(p), cfg.head_chunk):
        inputs = selected[start:start+cfg.head_chunk]
        targets = batch['targets'][start:start+cfg.head_chunk]
        logps.append(checkpoint(token_score, inputs, targets, use_reentrant=False)
                     if torch.is_grad_enabled() else token_score(inputs, targets))
    token_logp = torch.cat(logps)
    sequence_logp = torch.zeros(batch['count'], device=h.device).index_add(0, batch['owners'], token_logp)
    kl_hidden = selected[batch['kl_indices']]
    kl_logp = torch.log_softmax(module.model.lm_head(kl_hidden).float(), dim=-1)
    return sequence_logp, kl_logp


def _forward_kl(logp, logq):
    # At exactly identical distributions the true gradient is zero. Explicitly
    # preserve that fact: Adam can amplify float32 normalization roundoff into
    # a spurious first update of an otherwise untouched foreign slot.
    if torch.equal(logp.detach(), logq.detach()):
        return (logp - logq).sum(-1) * 0.
    return (logp.exp() * (logp - logq)).sum(-1).clamp_min(0.)


def _per_sequence(values, batch):
    owners = batch['owners'][batch['kl_indices']]
    sums = torch.zeros(batch['count'], device=values.device).index_add(0, owners, values)
    counts = torch.bincount(owners, minlength=batch['count']).clamp_min(1)
    return sums / counts


def _rng_state(prompt_rng):
    nr = np.random.get_state()
    return dict(python=random.getstate(), prompts=prompt_rng.getstate(),
                numpy=(nr[0], nr[1].tolist(), int(nr[2]), int(nr[3]), float(nr[4])),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(value, prompt_rng):
    random.setstate(value['python']); prompt_rng.setstate(value['prompts'])
    nr = value['numpy']; np.random.set_state((nr[0], np.asarray(nr[1], dtype=np.uint32), nr[2], nr[3], nr[4]))
    torch.set_rng_state(value['torch'].cpu())
    if value['cuda']:
        assert torch.cuda.is_available() and len(value['cuda']) == torch.cuda.device_count()
        torch.cuda.set_rng_state_all([x.cpu() for x in value['cuda']])


def _enable_training(module, cfg):
    module.model.train()
    # No stochastic dropout is permitted: old/new probability comparisons must match.
    for child in module.model.modules():
        if isinstance(child, torch.nn.Dropout):
            assert child.p == 0, 'This pilot assumes the pretrained model has zero dropout.'
    if cfg.checkpointing:
        module.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})


def _routing_name(routing):
    if isinstance(routing, bool):
        return 'routed' if routing else 'unrouted'
    assert routing in RULES, routing
    return routing


def update(module, tok, items, advantages, cfg, optimizers, task, routing):
    """One attempted step with atomic rollback and genuinely separate role gradients.

    All four rules use reward-A, reward-B, preserve-A, preserve-B Adam optimizers.
    In unrouted_preserve a foreign parameter has two different gradients computed
    at the same pre-update state. Each optimizer receives ONLY its role's saved
    gradient; gradients must not be accumulated and reused for both Adam steps.
    Role gradients are jointly clipped, then reward steps precede preservation
    steps. Separate moments are part of the optimizer recipe, not an inference
    claim that this equals Adam on a summed objective.
    """
    routing = _routing_name(routing)
    preserve = routing.endswith('_preserve')
    routed = routing.startswith('routed')
    reward_slots = [task] if routed else [0, 1]
    optimizer_ids = reward_slots + ([2 + (1-task)] if preserve else [])
    active_ids = sorted({k % 2 for k in optimizer_ids})
    assert len(optimizers) == 4, 'All arms require four Adam role histories.'
    slot_parameters = [list(module.parameters(k)) for k in (0, 1)]
    active = [p for k in active_ids for p in slot_parameters[k]]
    role_gradients = {}

    def clear_gradients():
        # Several optimizers share parameters; clearing once per parameter is safe.
        for k in (0, 1):
            for p in slot_parameters[k]:
                p.grad = None

    def capture_role(role):
        role_gradients[role] = [None if p.grad is None else p.grad.detach().clone()
                                for p in slot_parameters[role % 2]]

    clear_gradients()
    batch = _training_batch(items, tok.pad_token_id, module.device, cfg)
    module.model.eval()
    with torch.no_grad(), module.gate(MASKS['none']):
        reference_logp = _chosen_log_probs(module, batch).detach()
    with torch.no_grad(), module.gate(MASKS['AB']):
        old_logp = _chosen_log_probs(module, batch).detach()
    if preserve:
        with torch.no_grad(), module.gate((task == 0, task == 1)):
            owner_teacher = _chosen_log_probs(module, batch).detach()
    _enable_training(module, cfg)
    with module.gate(MASKS['AB']), module.route(task if routed else None):
        sequence_logp, current_logp = _scores(module, batch, cfg)
        advantages = torch.as_tensor(advantages, device=module.device, dtype=torch.float32)
        policy_loss = -(advantages * sequence_logp).mean()
        reference_kl = _per_sequence(_forward_kl(current_logp, reference_logp), batch).mean()
        loss = policy_loss + cfg.reference_kl * reference_kl
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite policy loss; no optimizer step was taken.')
        # Checkpoint recomputation must see the same gate and route.
        loss.backward()
    for role in reward_slots:
        capture_role(role)
    metrics = dict(policy_loss=float(policy_loss.detach()), reference_kl=float(reference_kl.detach()),
                   reward_loss=float(loss.detach()), preservation_kl=0., neutral_kl=0.,
                   accepted=False, accepted_lr=None, backtrack_attempts=0,
                   step_kl_mean=None, step_kl_max_sequence=None, delta_max_move=0.,
                   slot_max_move=[0., 0.], active_optimizer_slots=active_ids,
                   active_optimizer_indices=optimizer_ids,
                   optimizer_role_order='reward then preservation; isolated pre-update gradients')
    del sequence_logp, current_logp, policy_loss, reference_kl, loss
    clear_gradients()
    if preserve:
        with module.route(1-task), module.gate(MASKS['AB']):
            student = _chosen_log_probs(module, batch)
            comp = _per_sequence(_forward_kl(owner_teacher, student), batch).mean()
            if not torch.isfinite(comp):
                raise FloatingPointError('Nonfinite composition-preservation loss.')
            (cfg.preservation_coefficient * comp).backward()
            metrics['preservation_kl'] = float(comp.detach())
        del student, comp, owner_teacher
        with module.route(1-task), module.gate((task != 0, task != 1)):
            student = _chosen_log_probs(module, batch)
            neutral = _per_sequence(_forward_kl(reference_logp, student), batch).mean()
            if not torch.isfinite(neutral):
                raise FloatingPointError('Nonfinite foreign-slot neutrality loss.')
            (cfg.neutral_coefficient * neutral).backward()
            metrics['neutral_kl'] = float(neutral.detach())
        capture_role(2+(1-task))
        del student, neutral
    del reference_logp
    clear_gradients()
    metrics['loss'] = (metrics['reward_loss'] + cfg.preservation_coefficient * metrics['preservation_kl']
                       + cfg.neutral_coefficient * metrics['neutral_kl'])
    role_norms = [0.] * 4
    for role, gradients in role_gradients.items():
        if any(not torch.isfinite(g).all().item() for g in gradients if g is not None):
            raise FloatingPointError('Nonfinite role gradient; no optimizer step was taken.')
        role_norms[role] = math.sqrt(sum(float(g.float().square().sum()) for g in gradients if g is not None))
    metrics['role_gradient_norms'] = role_norms
    metrics['slot_gradient_norms'] = [math.hypot(role_norms[k], role_norms[k+2]) for k in (0,1)]
    grad_norm = math.sqrt(sum(x*x for x in role_norms))
    if not math.isfinite(grad_norm):
        raise FloatingPointError('Nonfinite total gradient; no optimizer step was taken.')
    metrics['grad_norm'] = grad_norm
    scale = min(1., cfg.clip_grad / (grad_norm + 1e-6))
    for gradients in role_gradients.values():
        for g in gradients:
            if g is not None:
                g.mul_(scale)
    original = [p.detach().clone() for p in active]
    original_opt = [copy.deepcopy(optimizers[k].state_dict()) for k in optimizer_ids]

    def restore():
        with torch.no_grad():
            for p, before in zip(active, original):
                p.copy_(before)
        for k, state in zip(optimizer_ids, original_opt):
            optimizers[k].load_state_dict(copy.deepcopy(state))

    module.model.eval()
    try:
        for attempt in range(cfg.backtracks+1):
            restore()
            lr = max(cfg.min_lr, cfg.lr * 0.5**attempt)
            for role in optimizer_ids:
                clear_gradients()
                for p, g in zip(slot_parameters[role % 2], role_gradients[role]):
                    p.grad = g
                for group in optimizers[role].param_groups:
                    group['lr'] = lr
                if role_norms[role] > 0:
                    optimizers[role].step()
            clear_gradients()
            finite = all(torch.isfinite(p).all().item() for p in active)
            metrics['backtrack_attempts'] = attempt + 1
            if finite:
                with torch.no_grad(), module.gate(MASKS['AB']):
                    new_logp = _chosen_log_probs(module, batch)
                    per_seq = _per_sequence(_forward_kl(old_logp, new_logp), batch)
                    mean, largest = float(per_seq.mean()), float(per_seq.max())
                if math.isfinite(mean) and math.isfinite(largest):
                    metrics.update(step_kl_mean=mean, step_kl_max_sequence=largest)
                    if mean <= cfg.max_mean_kl and largest <= cfg.max_sequence_kl:
                        moves = [float((p.detach()-b).abs().max()) for p,b in zip(active,original)]
                        metrics.update(accepted=True, accepted_lr=lr, delta_max_move=max(moves, default=0.))
                        offset = 0
                        for k in active_ids:
                            count = len(slot_parameters[k])
                            metrics['slot_max_move'][k] = max(moves[offset:offset+count], default=0.)
                            offset += count
                        break
        if not metrics['accepted']:
            restore()
    except BaseException:
        restore()
        raise
    finally:
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                group['lr'] = cfg.lr
        clear_gradients()
    return metrics

def _dev_summary(rows, cfg=None):
    """Predeclared development-only, depth-stratified checkpoint ranking.

    Adequacy means accuracy >= min(selection_ceiling, base + selection_gain).
    This is a ranking device, NOT a claim of learning at a ceiling-level base.
    Reporting must separately evaluate actual gains and removal/composition.
    """
    cfg = cfg or Config()
    accuracy, counts = {}, {}
    for task in (0, 1):
        for mask in MASKS:
            selected = [r for r in rows if r['task_id'] == task and r['mask'] == mask]
            assert selected, (task, mask)
            accuracy[f'{task}/{mask}'] = sum(r['correct'] for r in selected) / len(selected)
            counts[f'{task}/{mask}'] = [sum(r['correct'] for r in selected), len(selected)]
    owner_gain, ab_gain, leakage = [], [], []
    strata = []
    for task, owner, other in ((0, 'A', 'B'), (1, 'B', 'A')):
        base = accuracy[f'{task}/none']
        owner_gain.append(accuracy[f'{task}/{owner}']-base)
        ab_gain.append(accuracy[f'{task}/AB']-base)
        leakage.append(abs(accuracy[f'{task}/{other}']-base))
        depths = sorted({r['difficulty'] for r in rows if r['task_id'] == task})
        for depth in depths:
            acc, ns = {}, {}
            for mask in MASKS:
                selected = [r for r in rows if r['task_id'] == task and r['difficulty'] == depth and r['mask'] == mask]
                assert selected, (task,depth,mask)
                acc[mask] = sum(r['correct'] for r in selected)/len(selected)
                ns[mask] = len(selected)
            assert len(set(ns.values())) == 1, 'Unpaired development masks.'
            target = min(cfg.selection_ceiling, acc['none']+cfg.selection_gain)
            strata.append(dict(task_id=task, difficulty=depth, n=ns['none'], accuracy=acc,
                target=target, owner_adequate=acc[owner] >= target-1e-12,
                AB_adequate=acc['AB'] >= target-1e-12,
                owner_gain=acc[owner]-acc['none'], AB_gain=acc['AB']-acc['none'],
                owner_margin=acc[owner]-target, AB_margin=acc['AB']-target,
                nonowner_abs_change=abs(acc[other]-acc['none'])))
    task_adequacy = [all(s['owner_adequate'] and s['AB_adequate'] for s in strata if s['task_id']==task)
                     for task in (0,1)]
    key = [sum(task_adequacy), sum(s['owner_adequate'] and s['AB_adequate'] for s in strata),
           sum(s['AB_adequate'] for s in strata), sum(s['owner_adequate'] for s in strata),
           min(s['AB_margin'] for s in strata), sum(s['AB_margin'] for s in strata)/len(strata),
           min(s['owner_margin'] for s in strata), sum(s['owner_margin'] for s in strata)/len(strata),
           -sum(s['nonowner_abs_change'] for s in strata)/len(strata)]
    return {'accuracy': accuracy, 'counts': counts, 'owner_gains': owner_gain, 'AB_gains': ab_gain,
            'nonowner_abs_changes': leakage, 'strata':strata, 'task_adequacy':task_adequacy,
            'selection_key': key,
            'selection_rule':'lexicographic task/stratum adequacy; worst then macro AB and owner margins; neutrality; earlier tie'}


def _sample_prompts(train_set, task_iteration, cfg, rng):
    if not cfg.stratify_depth:
        return [rng.choice(train_set) for _ in range(cfg.prompts_per_update)]
    by_depth = {}
    for ex in train_set:
        by_depth.setdefault(int(ex['difficulty']), []).append(ex)
    depths = sorted(by_depth)
    # Exact long-run uniformity even when #prompts is not divisible by #depths.
    chosen = [depths[(task_iteration*cfg.prompts_per_update+j) % len(depths)]
              for j in range(cfg.prompts_per_update)]
    rng.shuffle(chosen)
    return [rng.choice(by_depth[d]) for d in chosen]

def train_arm(module, tok, data, cfg, run_dir, arm, seed, routing, on_checkpoint=None):
    """Train/resume one arm and leave its dev-selected checkpoint loaded.

    latest.pt is the resume point; best.pt is selection-only. An interrupted dev
    evaluation is rerun from a saved pending marker before any new update.
    History beyond the last committed checkpoint is discarded on restart.
    """
    cfg.validate()
    routing = _routing_name(routing)
    root = Path(run_dir); root.mkdir(parents=True, exist_ok=True)
    assert all(any(e['task_id'] == task for e in data['train']) for task in (0, 1))
    assert all(any(e['task_id'] == task for e in data['dev']) for task in (0, 1))
    signature = dict(config=asdict(cfg), arm=arm, seed=int(seed), routing=routing,
                     manifest=module.manifest(), base_hash=module.base_hash(),
                     train_dev_hash=_digest({'train': data['train'], 'dev': data['dev']}),
                     trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    config_path = root/'config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text()) == signature, 'Existing arm differs: use a new run folder.'
    else:
        atomic_json(config_path, signature)
    optimizer_slots = (0, 1, 0, 1)
    optimizers = [torch.optim.AdamW(module.parameters(task), lr=cfg.lr, weight_decay=0.) for task in optimizer_slots]
    rng = random.Random(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    history, dev_history = [], []
    step, best_step, best_key, pending_dev = 0, 0, None, True
    if (root/'latest.pt').exists():
        saved = _load(root/'latest.pt')
        assert saved['signature'] == signature
        module.load_state_dict(saved['deltas'])
        for optimizer, state in zip(optimizers, saved['optimizers']):
            optimizer.load_state_dict(state)
        _restore_rng(saved['rng'], rng)
        step, best_step, best_key = saved['step'], saved['best_step'], saved['best_key']
        pending_dev = saved['pending_dev']
        history, dev_history = saved['history'], saved['dev_history']
    else:
        module.zero_()
    def journal():
        tmp = root/'history.jsonl.tmp'
        tmp.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in history))
        os.replace(tmp, root/'history.jsonl')
        atomic_json(root/'dev_history.json', dev_history)
    def save():
        atomic_torch(root/'latest.pt', dict(signature=signature, deltas=module.state_dict(),
                     optimizers=[o.state_dict() for o in optimizers], rng=_rng_state(rng),
                     step=step, best_step=best_step, best_key=best_key, pending_dev=pending_dev,
                     history=history, dev_history=dev_history))
        journal()
        if on_checkpoint is not None:
            on_checkpoint()
    journal()
    status_path = root/'status.json'
    if status_path.exists() and json.loads(status_path.read_text()).get('complete'):
        status = json.loads(status_path.read_text())
        selected = _load(root/'best.pt'); assert selected['signature'] == signature
        module.load_state_dict(selected['deltas'])
        assert module.base_hash() == signature['base_hash']
        return status
    train_sets = [[e for e in data['train'] if e['task_id'] == task] for task in (0, 1)]
    save()
    while True:
        if pending_dev:
            rows = []
            for mask in MASKS:
                rows.extend(predict(module, tok, data['dev'], mask, cfg, 'dev', arm, seed))
            summary = _dev_summary(rows, cfg)
            summary['effective_deltas'] = module.effective_stats()
            # Save per-example evidence for every development decision.
            atomic_json(root/'dev_predictions'/f'step_{step:05d}.json', rows)
            key = summary['selection_key']
            if best_key is None or tuple(key) > tuple(best_key):
                best_key, best_step = key, step
                atomic_torch(root/'best.pt', dict(signature=signature, deltas=module.state_dict(),
                                                step=step, selection_key=key))
            dev_history = [r for r in dev_history if r['step'] != step]
            dev_history.append(dict(step=step, **summary))
            pending_dev = False
            save()
            print(f'{arm} seed {seed}: dev step {step}, owner gains {summary["owner_gains"]}, AB gains {summary["AB_gains"]}', flush=True)
        if step >= cfg.max_steps:
            break
        started = time.perf_counter()
        task = step % 2
        prompts = _sample_prompts(train_sets[task], step//2, cfg, rng)
        expanded = [ex for ex in prompts for _ in range(cfg.group_size)]
        items = []
        for start in range(0, len(expanded), cfg.generation_batch):
            items.extend(_generate(module, tok, expanded[start:start+cfg.generation_batch], cfg, 'AB', sample=True))
        rewards = np.asarray([float(i['correct']) for i in items]).reshape(-1, cfg.group_size)
        advantages = rewards - (rewards.sum(1, keepdims=True)-rewards)/(cfg.group_size-1)
        metrics = update(module, tok, items, advantages.reshape(-1), cfg, optimizers, task, routing)
        step += 1
        row = dict(step=step, task_id=task, reward=float(rewards.mean()),
                   valid_fraction=sum(i['valid'] for i in items)/len(items),
                   truncated_fraction=sum(i['truncated'] for i in items)/len(items),
                   reward_diverse_groups=int(((rewards.max(1)-rewards.min(1)) > 0).sum()),
                   groups=cfg.prompts_per_update,
                   prompt_depths=[int(ex['difficulty']) for ex in prompts],
                   reward_by_depth={str(d):float(rewards[[j for j,ex in enumerate(prompts) if int(ex['difficulty'])==d]].mean())
                       for d in sorted({int(ex['difficulty']) for ex in prompts})},
                   response_tokens=sum(len(i['generated_ids']) for i in items),
                   seconds=time.perf_counter()-started, **metrics)
        history.append(row)
        if cfg.snapshot_every and step % cfg.snapshot_every == 0:
            atomic_torch(root/'snapshots'/f'step_{step:05d}.pt',
                         dict(signature=signature, deltas=module.state_dict(), step=step))
        # A pending-dev checkpoint is committed before costly evaluation starts.
        pending_dev = step % cfg.eval_every == 0 or step == cfg.max_steps
        if step % cfg.save_every == 0 or pending_dev:
            save()
            print(f'{arm} seed {seed}: {step}/{cfg.max_steps}, task {task}, reward {row["reward"]:.3f}, accepted {metrics["accepted"]}', flush=True)
    assert module.base_hash() == signature['base_hash'], 'Frozen base changed.'
    last_hash = module.delta_hash()
    selected = _load(root/'best.pt'); module.load_state_dict(selected['deltas'])
    status = dict(complete=True, arm=arm, seed=int(seed), routing=routing,
                  steps=step, selected_step=best_step, selection_key=best_key,
                  accepted_updates=sum(r['accepted'] for r in history),
                  updates_with_movement=sum(r['delta_max_move'] > 0 for r in history),
                  reward_diverse_updates=sum(r['reward_diverse_groups'] > 0 for r in history),
                  selected_delta_hash=module.delta_hash(), latest_delta_hash=last_hash,
                  base_hash_unchanged=True, no_test_selection=True,
                  effective_deltas=module.effective_stats(),
                  gradient_routing=dict(reward_slots='owner only' if routing.startswith('routed') else 'both',
                                        preservation='foreign slot only' if routing.endswith('_preserve') else 'none',
                                        optimizer_policy='4 separate role Adam histories, isolated gradients, joint role clipping'),
                  preservation_scope='Fixed sampled prefixes, not a global guarantee.')
    atomic_json(status_path, status)
    return status
