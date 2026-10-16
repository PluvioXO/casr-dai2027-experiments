"""V6 controlled preservation extensions; the four V5 baselines call V5 unchanged."""
from dataclasses import dataclass, field, replace
from pathlib import Path
from unittest.mock import patch
import copy, hashlib, json, math, time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
import nr5_rl as v5
from nr5_rl import MASKS, atomic_json
from nr5_tasks import format_prompt

RECIPES = {
    'unrouted': ('unrouted', 'v5'),
    'routed': ('routed', 'v5'),
    'unrouted_preserve': ('unrouted_preserve', 'v5'),
    'routed_preserve': ('routed_preserve', 'v5'),
    'routed_preserve_all': ('routed_preserve', 'all_ab'),
    'routed_preserve_mixture': ('routed_preserve', 'all_mixture'),
}

@dataclass
class Config(v5.Config):
    preservation_mode: str = 'v5'
    kl_head_chunk: int = 32
    audit_steps: list = field(default_factory=lambda: [79, 80, 159, 160, 319, 320])

    def validate(self):
        super().validate()
        assert self.preservation_mode in ('v5', 'all_ab', 'all_mixture')
        assert self.kl_head_chunk > 0
        assert all(isinstance(s, int) and s > 0 for s in self.audit_steps)
        if self.preservation_mode == 'all_mixture':
            assert self.group_size == 4, 'Mixture has exactly one trajectory per mask per prompt.'
        return self

def all_position_batch(items, tok, module, cfg):
    n = max(len(i['generated_ids']) for i in items)
    return v5._training_batch(items, tok.pad_token_id, module.device, replace(cfg, kl_positions=n))

def preservation_items(module, tok, items, cfg, seed, step):
    """16 preservation trajectories in every arm; mixture adds 12 rollouts.

    AB uses one of the original four draws per prompt. A, B, and none each
    generate one new draw per prompt. Extra sampling has its own RNG stream.
    All reward trajectories remain AB and every reward sample is on-policy.
    """
    if cfg.preservation_mode != 'all_mixture':
        return items, dict(extra_rollout_responses=0, extra_rollout_tokens=0,
                           preservation_trajectory_masks={'AB': len(items)})
    assert len(items) % 4 == 0
    prompts = [items[j]['example'] for j in range(0, len(items), 4)]
    for j in range(0, len(items), 4):
        assert len({i['example']['id'] for i in items[j:j+4]}) == 1
    result = [items[j] for j in range(0, len(items), 4)]
    extra = []
    devices = [module.device.index or 0] if module.device.type == 'cuda' else []
    key = int(hashlib.sha256(f'nr6-preserve:{seed}:{step}'.encode()).hexdigest()[:15], 16)
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(key)
        if devices: torch.cuda.manual_seed_all(key)
        for mask in ('A', 'B', 'none'):
            for start in range(0, len(prompts), cfg.generation_batch):
                extra.extend(v5._generate(module, tok, prompts[start:start+cfg.generation_batch], cfg, mask, sample=True))
    result.extend(extra)
    assert len(result) == len(items)
    return result, dict(extra_rollout_responses=len(extra),
        extra_rollout_tokens=sum(len(i['generated_ids']) for i in extra),
        preservation_trajectory_masks={m: len(prompts) for m in MASKS})

def conditional_kl(module, batch, teacher_mask, cfg):
    """Exact conditional teacher||student KL, averaged per sequence.

    The caller supplies the student gate/route and keeps it through backward.
    Head checkpointing bounds memory: no response-length × vocabulary graph is
    retained. Teachers are detached, including when teacher and student share
    the same physical model. This remains a conditional surrogate, not full
    trajectory KL or an unbiased derivative of occupancy-dependent KL.
    """
    p = batch['positions'][batch['kl_indices']]
    with torch.no_grad(), module.gate(teacher_mask):
        ht = v5._hidden(module, batch)[p[:, 0], p[:, 1]].detach()
    hs = v5._hidden(module, batch)[p[:, 0], p[:, 1]]
    def chunk_loss(s, t):
        with torch.no_grad():
            logt = torch.log_softmax(module.model.lm_head(t).float(), dim=-1)
        logs = torch.log_softmax(module.model.lm_head(s).float(), dim=-1)
        return v5._forward_kl(logt, logs)
    values = []
    for start in range(0, len(p), cfg.kl_head_chunk):
        args = (hs[start:start+cfg.kl_head_chunk], ht[start:start+cfg.kl_head_chunk])
        values.append(checkpoint(chunk_loss, *args, use_reentrant=False)
                      if torch.is_grad_enabled() else chunk_loss(*args))
    return v5._per_sequence(torch.cat(values), batch).mean()

def gold_probe_items(tok, examples):
    """Privileged fixed traces ONLY for measuring update effects, never fitting."""
    out = []
    for e in examples:
        text = ' ' + ' -> '.join(map(str, e['gold_trace'])) + '\nAnswer: ' + e['answer'] + '\n'
        out.append(dict(example=e, prompt_ids=tok(format_prompt(e), add_special_tokens=False).input_ids,
            generated_ids=tok(text, add_special_tokens=False).input_ids, completion=text,
            answer=e['answer'], valid=True, correct=True, truncated=False))
    return out

@torch.no_grad()
def probe_nll(module, tok, examples, cfg):
    """Mean gold-completion token NLL on fixed development probes, by task/mask."""
    module.model.eval()
    out = {}
    for task in (0, 1):
        items = gold_probe_items(tok, [e for e in examples if e['task_id'] == task])
        if not items: continue
        batch = v5._training_batch(items, tok.pad_token_id, module.device, cfg)
        lengths = torch.tensor([len(i['generated_ids']) for i in items], device=module.device)
        for mask in ('A', 'B', 'AB'):
            with module.gate(MASKS[mask]):
                seq, _ = v5._scores(module, batch, cfg)
            out[f'{task}/{mask}'] = float((-seq / lengths).mean())
    return out

def train_recipe(module, tok, data, cfg, root, recipe, seed, on_checkpoint=None):
    """Use V5 sampling, optimizer, selection, and resumption; change only stated loss coverage."""
    rule, mode = RECIPES[recipe]
    assert cfg.preservation_mode == mode
    root = Path(root)
    next_step = 1
    if (root/'latest.pt').exists(): next_step = int(v5._load(root/'latest.pt')['step']) + 1
    context = {'step': next_step}
    probes = [sorted((e for e in data['dev'] if e['task_id'] == t and e['difficulty'] == d),
                     key=lambda e:e['id'])[0]
              for t in (0,1) for d in sorted({e['difficulty'] for e in data['dev'] if e['task_id']==t})]
    original_update = v5.update

    def measured_update(module, tok, items, advantages, cfg, optimizers, task, routing):
        step = context['step']; context['step'] += 1
        pitems, sampling = preservation_items(module, tok, items, cfg, seed, step)
        audit = step in cfg.audit_steps
        events, originals = [], []
        before = probe_nll(module, tok, probes, cfg) if audit else None
        tracker = {'first_role': None, 'attempt': 0}
        if audit:
            for role, opt in enumerate(optimizers):
                original = opt.step; originals.append((opt, original))
                def wrapped(*args, role=role, original=original, **kwargs):
                    if tracker['first_role'] is None: tracker['first_role'] = role
                    if role == tracker['first_role']: tracker['attempt'] += 1
                    result = original(*args, **kwargs)
                    events.append(dict(attempt=tracker['attempt'], role=role,
                        nll=probe_nll(module, tok, probes, cfg)))
                    return result
                opt.step = wrapped
        try:
            if mode == 'v5':
                metrics = original_update(module, tok, items, advantages, cfg, optimizers, task, routing)
            else:
                metrics = extended_update(module, tok, items, advantages, cfg, optimizers, task, routing, pitems)
        finally:
            for opt, original in originals: opt.step = original
        metrics.update(sampling)
        metrics['preservation_mode'] = mode
        if audit:
            record = dict(step=step, task_id=task, recipe=recipe, before=before, events=events,
                after=probe_nll(module, tok, probes, cfg), accepted=metrics['accepted'],
                accepted_attempt=metrics['backtrack_attempts'] if metrics['accepted'] else None,
                scope='Read-only gold-trace NLL probes; not reward or selection. Failed backtrack trials are not accepted updates.')
            atomic_json(root/'update_audits'/f'{step:05d}.json', record)
        return metrics
    with patch.object(v5, 'update', side_effect=measured_update):
        return v5.train_arm(module, tok, data, cfg, root, 'mlp_'+recipe, seed, rule, on_checkpoint)

# The extended update below is generated from V5's verified role-isolated update.
# Reward gradients, four Adam histories, clipping and AB step backtracking remain unchanged.

def extended_update(module, tok, items, advantages, cfg, optimizers, task, routing, pitems):
    """One attempted step with atomic rollback and genuinely separate role gradients.

    All four rules use reward-A, reward-B, preserve-A, preserve-B Adam optimizers.
    In unrouted_preserve a foreign parameter has two different gradients computed
    at the same pre-update state. Each optimizer receives ONLY its role's saved
    gradient; gradients must not be accumulated and reused for both Adam steps.
    Role gradients are jointly clipped, then reward steps precede preservation
    steps. Separate moments are part of the optimizer recipe, not an inference
    claim that this equals Adam on a summed objective.
    """
    routing = v5._routing_name(routing)
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
    batch = v5._training_batch(items, tok.pad_token_id, module.device, cfg)
    module.model.eval()
    with torch.no_grad(), module.gate(MASKS['none']):
        reference_logp = v5._chosen_log_probs(module, batch).detach()
    with torch.no_grad(), module.gate(MASKS['AB']):
        old_logp = v5._chosen_log_probs(module, batch).detach()
    v5._enable_training(module, cfg)
    with module.gate(MASKS['AB']), module.route(task if routed else None):
        sequence_logp, current_logp = v5._scores(module, batch, cfg)
        advantages = torch.as_tensor(advantages, device=module.device, dtype=torch.float32)
        policy_loss = -(advantages * sequence_logp).mean()
        reference_kl = v5._per_sequence(v5._forward_kl(current_logp, reference_logp), batch).mean()
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
        pbatch = all_position_batch(pitems, tok, module, cfg)
        with module.route(1-task), module.gate(MASKS['AB']):
            comp = conditional_kl(module, pbatch, (task == 0, task == 1), cfg)
            if not torch.isfinite(comp): raise FloatingPointError('Nonfinite composition KL')
            (cfg.preservation_coefficient * comp).backward()
            metrics['preservation_kl'] = float(comp.detach())
        del comp
        with module.route(1-task), module.gate((task != 0, task != 1)):
            neutral = conditional_kl(module, pbatch, MASKS['none'], cfg)
            if not torch.isfinite(neutral): raise FloatingPointError('Nonfinite neutrality KL')
            (cfg.neutral_coefficient * neutral).backward()
            metrics['neutral_kl'] = float(neutral.detach())
        capture_role(2+(1-task))
        metrics['preservation_positions'] = len(pbatch['kl_indices'])
        del neutral, pbatch
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
                    new_logp = v5._chosen_log_probs(module, batch)
                    per_seq = v5._per_sequence(v5._forward_kl(old_logp, new_logp), batch)
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
