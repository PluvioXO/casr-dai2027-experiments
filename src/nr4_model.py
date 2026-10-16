"""Native parameter slots for checkpoint-anchored, two-task RL experiments.

No layers or hidden dimensions are added to the transformer. Each trainable
FP32 delta replaces a prescribed row/column slice of an existing weight with
``(checkpoint_slice.float() + delta).to(checkpoint_dtype)``. The parametrization
is an implementation of weight replacement, not a parallel activation branch.

Two disjoint parameter supports are selected before training. All-on inference
still permits functional interaction through shared activations. In particular,
in grouped-query attention a selected KV head also serves frozen query heads in
its group. Parameter disjointness is not functional isolation.
"""

from contextlib import contextmanager
import hashlib
import json
import random

import torch
from torch import nn
from torch.nn.utils import parametrize


def _tensor_hash(named_tensors):
    """Hash exact tensor bytes, including BF16, without a floating-point cast."""
    h = hashlib.sha256()
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        t = tensor.detach().contiguous().cpu()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("ascii"))
        h.update(str(t.dtype).encode("ascii"))
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class _Control:
    def __init__(self):
        self.mask = (True, True)
        self.closed = False


class _NativeSlice(nn.Module):
    """Two sparse, non-overlapping deltas on one native weight tensor."""

    def __init__(self, original, axis, indices, control):
        super().__init__()
        self.axis = int(axis)
        self.control = control
        self.deltas = nn.ParameterList()
        assert original.ndim == 2 and self.axis in (0, 1)
        assert not set(indices[0]).intersection(indices[1]), "Slot overlap"
        for slot in range(2):
            ix = torch.tensor(indices[slot], dtype=torch.long, device=original.device)
            self.register_buffer(f"index_{slot}", ix, persistent=False)
            shape = list(original.shape)
            shape[self.axis] = len(indices[slot])
            self.deltas.append(nn.Parameter(torch.zeros(
                shape, dtype=torch.float32, device=original.device)))

    def forward(self, original):
        if not any(self.control.mask):
            return original
        result = original
        for slot, enabled in enumerate(self.control.mask):
            if enabled:
                ix = getattr(self, f"index_{slot}")
                replacement = (original.index_select(self.axis, ix).float()
                               + self.deltas[slot]).to(original.dtype)
                result = result.index_copy(self.axis, ix, replacement)
        return result


class NativeSlots:
    """Own two sets of native checkpoint-relative weight increments.

    ``gate((a, b))`` switches increments in the forward pass. ``route(task)``
    makes only task 0 or 1's deltas trainable; ``route(None)`` enables both.
    Routing does NOT detach activations or block gradient transport through
    frozen weights. Use a separate optimizer for each slot and do not call
    ``step`` on an inactive slot's optimizer (Adam momentum can otherwise move
    an inactive slot).

    ``mlp``, ``attn``, and ``attn_mlp`` use the same layers, slot assignment seed,
    and approximately equal per-slot scalar budgets when native dimensions
    permit. Attention-only extends the mixed arm's full query-head slice with
    additional query/output dimensions in the same KV group. This implementation
    expects Qwen-style gated feedforward layers and at least two KV heads.
    """

    def __init__(self, model, architecture="mlp", seed=17, n_layers=8,
                 mlp_neurons=32):
        if architecture not in {"mlp", "attn", "attn_mlp"}:
            raise ValueError("architecture must be 'mlp', 'attn', or 'attn_mlp'")
        if int(n_layers) < 1 or int(mlp_neurons) < 1:
            raise ValueError("Layer and neuron counts must be positive")
        if any(parametrize.is_parametrized(m) for m in model.modules()):
            raise ValueError("Start with an unparametrized pretrained model")
        self.model = model
        self.architecture = architecture
        self.seed = int(seed)
        self._control = _Control()
        self._attachments = []
        self._named_deltas = []
        self._base_parameters = list(model.named_parameters())
        for _, p in self._base_parameters:
            p.requires_grad_(False)
            p.grad = None
        self._original_hash = _tensor_hash(self._base_parameters)

        try:
            layers = model.model.layers
            config = model.config
            width = int(config.hidden_size)
            q_heads = int(config.num_attention_heads)
            kv_heads = int(getattr(config, "num_key_value_heads", q_heads))
            head_dim = int(getattr(config, "head_dim", None) or width // q_heads)
            intermediate = int(config.intermediate_size)
        except (AttributeError, TypeError) as error:
            raise ValueError("Expected Qwen-style model.model.layers/config") from error
        if kv_heads < 2 or q_heads % kv_heads:
            raise ValueError("Need >=2 KV groups and query heads divisible by KV heads")
        if intermediate < 2:
            raise ValueError("Need at least two native MLP neurons")
        group_size = q_heads // kv_heads
        chosen_count = min(int(n_layers), len(layers))
        if not chosen_count:
            raise ValueError("Model has no transformer layers")
        layer_ids = ([len(layers) // 2] if chosen_count == 1 else
                     [i * (len(layers) - 1) // (chosen_count - 1)
                      for i in range(chosen_count)])

        # Mixed uses three H*n MLP slices and four H*head_dim attention slices.
        # Leave enough neurons for the matched MLP-only arm on tiny smoke models.
        attention_neuron_equivalent = 4 * head_dim / 3
        match_room = max(1, intermediate // 2 - round(attention_neuron_equivalent))
        mixed_n = min(int(mlp_neurons), match_room)
        mlp_only_n = min(intermediate // 2, round(mixed_n + attention_neuron_equivalent))
        n = {"mlp": mlp_only_n, "attn": 0, "attn_mlp": mixed_n}[architecture]
        # Attention-only reallocates the mixed arm's MLP scalar budget to
        # additional Q rows and matching O columns within the same KV group.
        attn_q_dims_target = head_dim + 1.5 * mixed_n
        attn_q_dims = min(group_size * head_dim, round(attn_q_dims_target))
        mixed_count = chosen_count * width * (3 * mixed_n + 4 * head_dim)
        mlp_count = chosen_count * width * 3 * mlp_only_n
        attn_count = chosen_count * width * (2 * attn_q_dims + 2 * head_dim)
        self._plan = {
            "architecture": architecture, "seed": self.seed,
            "layers": layer_ids, "hidden_size": width,
            "intermediate_size": intermediate, "query_heads": q_heads,
            "kv_heads": kv_heads, "head_dim": head_dim,
            "query_heads_per_kv_group": group_size,
            "requested_mixed_mlp_neurons": int(mlp_neurons),
            "mixed_mlp_neurons": mixed_n, "mlp_only_neurons": mlp_only_n,
            "selected_mlp_neurons_per_slot_per_layer": n,
            "attention_only_query_dimensions_per_slot_per_layer": attn_q_dims,
            "attention_only_query_dimensions_target": attn_q_dims_target,
            "attention_only_group_capacity_limited": group_size * head_dim < round(attn_q_dims_target),
            "per_slot_comparator_counts": {"mlp": mlp_count, "attn": attn_count,
                                           "attn_mlp": mixed_count},
            "mlp_vs_mixed_relative_count_difference": (mlp_count - mixed_count) / mixed_count,
            "attn_vs_mixed_relative_count_difference": (attn_count - mixed_count) / mixed_count,
            "architecture_preserved": True,
            "frozen": "Checkpoint weights, biases, norms, embeddings, unembedding",
            "gqa_note": ("Each selected KV head is shared by every query head in its "
                         "group. Mixed updates one full query head and matching output "
                         "columns. Attention-only also updates selected query/output "
                         "dimensions from other heads in that same group. Unselected "
                         "query parameters remain frozen but read updated keys and values."),
            "attention_only_note": ("The attention-only slot is a native parameter slice, "
                                    "not exclusively a set of whole attention heads. Its extra "
                                    "Q rows and matching O columns stay within its own KV group."),
            "restoration": "Gates disable checkpoint-relative deltas, not base neurons or heads",
            "functional_isolation_guaranteed": False,
            "layout": [],
        }

        try:
            for layer_id in layer_ids:
                layer = layers[layer_id]
                rng = random.Random(self.seed * 1000003 + layer_id * 9176 + 101)
                neuron_order = list(range(intermediate))
                rng.shuffle(neuron_order)
                # Half partitions ensure cross-architecture slots never overlap.
                half = intermediate // 2
                neurons = [sorted(neuron_order[:half][:n]),
                           sorted(neuron_order[half:][:n])]
                kv_order = list(range(kv_heads))
                rng.shuffle(kv_order)
                kv_selected = kv_order[:2]
                q_selected = [kv * group_size + rng.randrange(group_size)
                              for kv in kv_selected]
                q_indices = [list(range(h * head_dim, (h + 1) * head_dim))
                             for h in q_selected]
                if architecture == "attn":
                    for slot in range(2):
                        group_head_start = kv_selected[slot] * group_size
                        other_heads = [h for h in range(group_head_start, group_head_start + group_size)
                                       if h != q_selected[slot]]
                        rng.shuffle(other_heads)
                        other = []
                        for head in other_heads:
                            dimensions = list(range(head * head_dim, (head + 1) * head_dim))
                            rng.shuffle(dimensions)
                            other.extend(dimensions)
                        q_indices[slot] = sorted(q_indices[slot] + other[:attn_q_dims - head_dim])
                self._plan["layout"].append({
                    "layer": layer_id, "mlp_neuron_indices": neurons,
                    "kv_head_indices": kv_selected if architecture != "mlp" else [],
                    "primary_query_head_indices": q_selected if architecture != "mlp" else [],
                    "query_head_indices": q_selected if architecture != "mlp" else [],
                    "touched_query_head_indices": [sorted({j // head_dim for j in ix})
                                                   for ix in q_indices] if architecture != "mlp" else [[], []],
                    "query_dimension_indices": q_indices if architecture != "mlp" else [[], []],
                })
                prefix = f"model.layers.{layer_id}"
                if architecture != "attn":
                    for proj, axis, shape in [
                        ("gate_proj", 0, (intermediate, width)),
                        ("up_proj", 0, (intermediate, width)),
                        ("down_proj", 1, (width, intermediate)),
                    ]:
                        linear = getattr(layer.mlp, proj)
                        self._attach(linear, axis, neurons, shape,
                                     f"{prefix}.mlp.{proj}.weight")
                if architecture != "mlp":
                    kv_indices = [list(range(h * head_dim, (h + 1) * head_dim))
                                  for h in kv_selected]
                    for proj, axis, indices, shape in [
                        ("q_proj", 0, q_indices, (q_heads * head_dim, width)),
                        ("k_proj", 0, kv_indices, (kv_heads * head_dim, width)),
                        ("v_proj", 0, kv_indices, (kv_heads * head_dim, width)),
                        ("o_proj", 1, q_indices, (width, q_heads * head_dim)),
                    ]:
                        self._attach(getattr(layer.self_attn, proj), axis, indices,
                                     shape, f"{prefix}.self_attn.{proj}.weight")
        except Exception:
            self.close()
            raise
        self._plan["trainable_scalars_per_slot"] = [
            sum(p.numel() for p in self.parameters(slot)) for slot in range(2)]
        expected = {"mlp": mlp_count, "attn": attn_count, "attn_mlp": mixed_count}[architecture]
        assert self._plan["trainable_scalars_per_slot"] == [expected, expected]
        self._plan["base_hash"] = self._original_hash
        self.model.eval()

    def _attach(self, linear, axis, indices, expected_shape, name):
        if not isinstance(linear, nn.Linear) or tuple(linear.weight.shape) != expected_shape:
            raise ValueError(f"Unexpected native projection for {name}; expected {expected_shape}")
        adapter = _NativeSlice(linear.weight, axis, indices, self._control)
        parametrize.register_parametrization(linear, "weight", adapter)
        self._attachments.append((linear, adapter))
        for slot, parameter in enumerate(adapter.deltas):
            self._named_deltas.append((f"{name}.slot_{slot}", parameter, slot))

    @property
    def mask(self):
        return self._control.mask

    @property
    def device(self):
        return self._base_parameters[0][1].device

    def _check_open(self):
        if self._control.closed:
            raise RuntimeError("NativeSlots has been closed")

    @contextmanager
    def gate(self, mask):
        self._check_open()
        if len(mask) != 2 or any(v not in (0, 1, False, True) for v in mask):
            raise ValueError("Use two binary forward gates")
        previous = self._control.mask
        self._control.mask = tuple(bool(v) for v in mask)
        try:
            yield self
        finally:
            self._control.mask = previous

    @contextmanager
    def route(self, task=None):
        self._check_open()
        if task not in (None, 0, 1):
            raise ValueError("Task must be 0, 1, or None for unrouted updates")
        previous = [p.requires_grad for _, p, _ in self._named_deltas]
        for _, p, slot in self._named_deltas:
            p.requires_grad_(task is None or slot == task)
        try:
            yield self
        finally:
            for (_, p, _), active in zip(self._named_deltas, previous):
                p.requires_grad_(active)

    def parameters(self, task=None):
        self._check_open()
        if task not in (None, 0, 1):
            raise ValueError("Task must be 0, 1, or None")
        return [p for _, p, slot in self._named_deltas if task is None or slot == task]

    def state_dict(self):
        self._check_open()
        return {name: p.detach().clone() for name, p, _ in self._named_deltas}

    def load_state_dict(self, state):
        self._check_open()
        expected = {name: p for name, p, _ in self._named_deltas}
        if set(state) != set(expected):
            raise ValueError("Delta checkpoint keys do not match the selected layout")
        # Validate before mutating anything. The caller must also compare manifests.
        for name, value in state.items():
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                    or value.shape != expected[name].shape or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid FP32 delta tensor: {name}")
        with torch.no_grad():
            for name, value in state.items():
                expected[name].copy_(value.to(expected[name].device))

    def base_hash(self):
        return _tensor_hash(self._base_parameters)

    def delta_hash(self):
        self._check_open()
        return _tensor_hash([(name, p) for name, p, _ in self._named_deltas])

    @torch.no_grad()
    def effective_stats(self):
        """Compare FP32 master deltas with native-dtype effective weight changes.

        This intentionally measures each slot as enabled, irrespective of the
        current forward gates. Small FP32 increments may round away at BF16
        weights; their optimizer state remains available to accumulate updates.
        """
        self._check_open()
        totals = [0, 0]
        accum = torch.zeros(2, 5, dtype=torch.float64, device=self.device)
        for linear, adapter in self._attachments:
            original = linear.parametrizations.weight.original
            for slot in range(2):
                raw = adapter.deltas[slot].detach()
                indices = getattr(adapter, f"index_{slot}")
                base = original.index_select(adapter.axis, indices)
                effective = (base.float() + raw).to(base.dtype).float() - base.float()
                accum[slot, 0] += raw.double().square().sum()
                accum[slot, 1] = torch.maximum(accum[slot, 1], raw.abs().max().double())
                accum[slot, 2] += effective.double().square().sum()
                accum[slot, 3] = torch.maximum(accum[slot, 3], effective.abs().max().double())
                accum[slot, 4] += torch.count_nonzero(effective)
                totals[slot] += raw.numel()
        values = accum.cpu().tolist()
        return [{"task_id": slot, "raw_delta_l2": row[0] ** 0.5,
                 "raw_delta_max": row[1], "effective_delta_l2": row[2] ** 0.5,
                 "effective_delta_max": row[3], "effective_nonzero_count": int(row[4]),
                 "total": totals[slot], "effective_nonzero_fraction": row[4] / totals[slot]}
                for slot, row in enumerate(values)]

    def manifest(self):
        self._check_open()
        # A JSON roundtrip returns independent nested lists and verifies serializability.
        return json.loads(json.dumps(self._plan))

    def zero_(self):
        self._check_open()
        with torch.no_grad():
            for _, p, _ in self._named_deltas:
                p.zero_()
                p.grad = None
        return self

    def assert_frozen_base(self):
        self._check_open()
        if any(p.requires_grad for _, p in self._base_parameters):
            raise AssertionError("A checkpoint parameter became trainable")
        if self.base_hash() != self._original_hash:
            raise AssertionError("Frozen checkpoint parameters changed")
        return True

    def close(self):
        """Remove all parametrizations and restore the original checkpoint weights."""
        if self._control.closed:
            return
        for linear, _ in reversed(self._attachments):
            parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)
        self._control.closed = True
