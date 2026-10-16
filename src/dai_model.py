"""Two switchable LoRA adapters with the same interface as the native slots."""
import math
import torch
from torch import nn
from torch.nn.utils import parametrize
from nr4_model import NativeSlots


class _LoRAWeight(nn.Module):
    def __init__(self, original, control, rank, seed):
        super().__init__()
        self.control = control
        self.left = nn.ParameterList()
        self.right = nn.ParameterList()
        self.initial_right = []
        generator = torch.Generator(device=original.device).manual_seed(seed)
        for _ in range(2):
            a = torch.empty(rank, original.shape[1], device=original.device, dtype=torch.float32)
            a.uniform_(-1 / math.sqrt(original.shape[1]), 1 / math.sqrt(original.shape[1]),
                       generator=generator)
            self.right.append(nn.Parameter(a))
            self.left.append(nn.Parameter(torch.zeros(original.shape[0], rank,
                                                      device=original.device, dtype=torch.float32)))
            self.initial_right.append(a.detach().clone())

    def forward(self, original):
        result = original
        for slot, enabled in enumerate(self.control.mask):
            if enabled:
                # alpha = rank, hence the LoRA multiplier is one.
                result = (result.float() + self.left[slot] @ self.right[slot]).to(original.dtype)
        return result


class LoRASlots(NativeSlots):
    def __init__(self, model, architecture="mlp", seed=17, n_layers=8, mlp_neurons=32, rank=18):
        if architecture != "mlp" or rank < 1:
            raise ValueError("This matched study uses MLP LoRA with a positive rank")
        super().__init__(model, architecture, seed, n_layers, mlp_neurons)
        native_count = self._plan["trainable_scalars_per_slot"][0]
        layers = self._plan["layers"]
        old_attachments = self._attachments
        self._attachments, self._named_deltas = [], []
        try:
            for index, (linear, _) in enumerate(old_attachments):
                parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)
                adapter = _LoRAWeight(linear.weight, self._control, rank, seed + 1009 * index)
                parametrize.register_parametrization(linear, "weight", adapter)
                self._attachments.append((linear, adapter))
                name = f"model.layers.{layers[index // 3]}.mlp.{('gate_proj','up_proj','down_proj')[index % 3]}"
                for slot in range(2):
                    self._named_deltas.extend([
                        (f"{name}.slot_{slot}.A", adapter.right[slot], slot),
                        (f"{name}.slot_{slot}.B", adapter.left[slot], slot)])
        except BaseException:
            for linear, _ in old_attachments:
                if parametrize.is_parametrized(linear):
                    parametrize.remove_parametrizations(linear, "weight", leave_parametrized=False)
            self._control.closed = True
            raise
        counts = [sum(p.numel() for p in self.parameters(k)) for k in range(2)]
        self.architecture = "lora"
        self._plan = dict(architecture="lora", seed=seed, layers=layers, rank=rank,
                          alpha=rank, dropout=0, trainable_scalars_per_slot=counts,
                          native_comparator_scalars_per_slot=native_count,
                          target_modules=["gate_proj", "up_proj", "down_proj"],
                          implementation="FP32 low-rank weight parametrization; alpha/rank=1",
                          architecture_preserved=True, base_hash=self._original_hash)

    def zero_(self):
        # Zeroing both factors would make LoRA untrainable. Restore A, zero B.
        with torch.no_grad():
            for _, adapter in self._attachments:
                for slot in range(2):
                    adapter.right[slot].copy_(adapter.initial_right[slot])
                    adapter.left[slot].zero_()
            for p in self.parameters():
                p.grad = None
        return self

    @torch.no_grad()
    def effective_stats(self):
        output = []
        for slot in range(2):
            squared, maximum, nonzero, total = 0., 0., 0, 0
            for _, adapter in self._attachments:
                delta = adapter.left[slot] @ adapter.right[slot]
                squared += float(delta.double().square().sum())
                maximum = max(maximum, float(delta.abs().max()))
                nonzero += int(torch.count_nonzero(delta))
                total += delta.numel()
            output.append(dict(task_id=slot, raw_delta_l2=squared ** .5, raw_delta_max=maximum,
                               effective_delta_l2=squared ** .5, effective_delta_max=maximum,
                               effective_nonzero_count=nonzero, total=total,
                               effective_nonzero_fraction=nonzero / total,
                               factor_scalars=sum(p.numel() for p in self.parameters(slot))))
        return output
