"""Offline mechanics checks. These synthetic tests provide no task-performance evidence."""
import copy
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from nr4_model import NativeSlots


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=True)
        self.k_proj = nn.Linear(16, 8, bias=True)
        self.v_proj = nn.Linear(16, 8, bias=True)
        self.o_proj = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, 4, 4).transpose(1, 2)
        k = self.k_proj(x).view(b, t, 2, 4).transpose(1, 2).repeat_interleave(2, 1)
        v = self.v_proj(x).view(b, t, 2, 4).transpose(1, 2).repeat_interleave(2, 1)
        weights = q.float() @ k.float().transpose(-2, -1) / 2
        causal = torch.ones(t, t, dtype=torch.bool, device=x.device).triu(1)
        weights = weights.masked_fill(causal, -torch.inf).softmax(-1).to(x.dtype)
        return self.o_proj((weights @ v).transpose(1, 2).reshape(b, t, 16))


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(16, 64, bias=False)
        self.up_proj = nn.Linear(16, 64, bias=False)
        self.down_proj = nn.Linear(64, 16, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()
        self.n1, self.n2 = nn.LayerNorm(16), nn.LayerNorm(16)

    def forward(self, x):
        x = x + self.self_attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class _TinyNativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16, intermediate_size=64,
                                      num_attention_heads=4, num_key_value_heads=2)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block() for _ in range(3)])
        self.embed = nn.Embedding(23, 16)
        self.head = nn.Linear(16, 23, bias=False)

    def forward(self, tokens):
        x = self.embed(tokens)
        for layer in self.model.layers:
            x = layer(x)
        return self.head(x)


def _one_check(architecture, dtype, device):
    torch.manual_seed(31415)
    model = _TinyNativeModel().to(device=device, dtype=dtype).eval()
    tokens = torch.tensor([[1, 3, 7, 2], [6, 8, 1, 4]], device=device)
    with torch.no_grad():
        reference = model(tokens).clone()
    slots = NativeSlots(model, architecture, n_layers=2, mlp_neurons=5)
    manifest = slots.manifest()
    assert manifest["layers"] == [0, 2]
    assert manifest["trainable_scalars_per_slot"] == [manifest["per_slot_comparator_counts"][architecture]] * 2
    for layout in manifest["layout"]:
        if architecture != "mlp":
            d, group = manifest["head_dim"], manifest["query_heads_per_kv_group"]
            for slot in range(2):
                kv, q = layout["kv_head_indices"][slot], layout["primary_query_head_indices"][slot]
                indices = layout["query_dimension_indices"][slot]
                assert set(range(q * d, (q + 1) * d)).issubset(indices)
                assert all(kv * group * d <= j < (kv + 1) * group * d for j in indices)
                assert len(indices) == (manifest["attention_only_query_dimensions_per_slot_per_layer"]
                                        if architecture == "attn" else d)
                assert len(indices) == len(set(indices))
        if architecture == "attn":
            assert layout["mlp_neuron_indices"] == [[], []]
            assert all(".self_attn." in key for key in slots.state_dict())
    assert all(p.dtype == torch.float32 for p in slots.parameters())
    assert all(s["effective_delta_l2"] == s["raw_delta_l2"] == 0
               and s["effective_nonzero_count"] == 0 for s in slots.effective_stats())
    with torch.no_grad():
        assert torch.equal(model(tokens), reference), "Zero increments changed base logits"
        for slot in range(2):
            for p in slots.parameters(slot):
                p.fill_(0.05 if slot == 0 else -0.03)
        for linear, adapter in slots._attachments:
            original = linear.parametrizations.weight.original
            mask_a = torch.zeros_like(original, dtype=torch.bool)
            mask_b = torch.zeros_like(original, dtype=torch.bool)
            mask_a.index_fill_(adapter.axis, adapter.index_0, True)
            mask_b.index_fill_(adapter.axis, adapter.index_1, True)
            assert not (mask_a & mask_b).any(), "Slot supports overlap"
            for mask, support in [((1, 0), mask_a), ((0, 1), mask_b),
                                  ((1, 1), mask_a | mask_b), ((0, 0), mask_a & mask_b)]:
                with slots.gate(mask):
                    actual = linear.weight.detach()
                    assert torch.equal(actual[~support], original[~support])
                    expected = original
                    for bank in range(2):
                        if mask[bank]:
                            ix = getattr(adapter, f"index_{bank}")
                            values = (original.index_select(adapter.axis, ix).float()
                                      + adapter.deltas[bank]).to(dtype)
                            expected = expected.index_copy(adapter.axis, ix, values)
                    assert torch.equal(actual, expected), "Incorrect native slice replacement"
        with slots.gate((0, 0)):
            assert torch.equal(model(tokens), reference), "All-off logits differ from base"
    before_mask = slots.mask
    try:
        with slots.gate((0, 1)):
            raise RuntimeError("Deliberate nested-context exception")
    except RuntimeError:
        pass
    assert slots.mask == before_mask
    before_flags = [p.requires_grad for p in slots.parameters()]
    try:
        with slots.route(0):
            raise RuntimeError("Deliberate routing exception")
    except RuntimeError:
        pass
    assert before_flags == [p.requires_grad for p in slots.parameters()]

    # Both increments participate in inference, but only one slot learns. The
    # loss is downstream of a completely frozen middle block.
    optimizers = [torch.optim.AdamW(slots.parameters(k), lr=0.001, weight_decay=0.01)
                  for k in range(2)]
    original_base_hash = slots.base_hash()
    for task in (1, 0, None):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        before = [p.detach().clone() for p in slots.parameters()]
        before_by_slot = [[p.detach().clone() for p in slots.parameters(k)] for k in range(2)]
        with slots.route(task):
            loss = model(tokens).float().square().mean()
            loss.backward()
            active = [0, 1] if task is None else [task]
            for bank in range(2):
                params = slots.parameters(bank)
                if bank in active:
                    assert all(p.grad is not None for p in params)
                    assert params[0].grad.abs().sum() > 0, "No gradient to earliest native delta"
                else:
                    assert all(p.grad is None for p in params), "Off-task gradients leaked"
            for bank in active:
                optimizers[bank].step()
        for bank in range(2):
            changed = any(not torch.equal(old, new) for old, new
                          in zip(before_by_slot[bank], slots.parameters(bank)))
            assert changed == (task is None or bank == task), "Incorrect optimizer isolation"
        assert any(not torch.equal(old, new) for old, new in zip(before, slots.parameters()))
    slots.assert_frozen_base()
    assert slots.base_hash() == original_base_hash

    saved, digest = slots.state_dict(), slots.delta_hash()
    assert all(s["effective_delta_l2"] > 0 and s["raw_delta_l2"] > 0
               and 0 < s["effective_nonzero_fraction"] <= 1 for s in slots.effective_stats())
    slots.zero_()
    slots.load_state_dict(saved)
    assert slots.delta_hash() == digest
    malformed = {k: v.clone() for k, v in saved.items()}
    next(iter(malformed.values())).flatten()[0] = float("nan")
    try:
        slots.load_state_dict(malformed)
        raise AssertionError("Non-finite delta checkpoint accepted")
    except ValueError:
        pass
    assert slots.delta_hash() == digest, "Failed load partially mutated deltas"
    with slots.gate((0, 0)), torch.no_grad():
        assert torch.equal(model(tokens), reference), "Training damaged base restoration"
    slots.close()
    with torch.no_grad():
        assert torch.equal(model(tokens), reference), "Close did not restore original weights"
    assert all(not p.requires_grad for p in model.parameters())
    return {"architecture": architecture, "dtype": str(dtype), "device": str(device),
            "per_slot_parameters": manifest["trainable_scalars_per_slot"],
            "base_restoration_bitwise": True, "disjoint_native_supports": True,
            "routing_and_optimizer_isolation": True, "frozen_block_gradient_transport": True,
            "safe_state_loading": True, "context_restoration": True, "close_restores_base": True}


def run_checks(device="cpu", include_bfloat16=True):
    """Run synthetic native-weight mechanics tests; never downloads a model."""
    device = torch.device(device)
    tests = [("mlp", torch.float32), ("attn", torch.float32), ("attn_mlp", torch.float32)]
    if include_bfloat16:
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 checks require a BF16-capable GPU")
        tests.extend([("attn", torch.bfloat16), ("attn_mlp", torch.bfloat16)])
    result = [_one_check(architecture, dtype, device) for architecture, dtype in tests]
    return {"status": "passed", "synthetic_only": True, "checks": result}


if __name__ == "__main__":
    import json
    torch.set_num_threads(1)
    print(json.dumps(run_checks(), indent=2))
