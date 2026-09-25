"""LoRA parameters, attachment through Comfy's weight-function hook, candidate
states, and export in the one format verified against the evaluator's
`LoraLoader` at its pinned commit:

    diffusion_model.{module}.lora_up.weight    [out, r]
    diffusion_model.{module}.lora_down.weight  [r, in]
    diffusion_model.{module}.alpha             scalar = scale * r

Comfy computes  W += (alpha / r) * (up @ down)  in fp32 then casts, which is
exactly what the wrapper below computes during training.
"""

import math
import re

import torch
from safetensors.torch import save_file


class LoraWrapper:
    """Weight function installed via ModelPatcher.add_weight_wrapper.

    Comfy calls `f(weight)` inside cast_bias_weight with autograd enabled and
    `f.to(device)` when moving modules; gradients flow to `up`/`down`.
    """

    def __init__(self, scale: float):
        self.up = None
        self.down = None
        self.scale = scale
        self.enabled = True

    def __call__(self, w):
        if not self.enabled or self.up is None:
            return w
        delta = (self.up.float() @ self.down.float()) * self.scale
        return (w.float() + delta.reshape(w.shape)).to(w.dtype)

    def to(self, device):
        return self


def select_targets(diffusion_model, include=None, exclude=None):
    inc = re.compile(include) if include else None
    exc = re.compile(exclude) if exclude else None
    out = []
    for name, m in diffusion_model.named_modules():
        if not (isinstance(m, torch.nn.Linear) or m.__class__.__name__ == "Linear"):
            continue
        w = getattr(m, "weight", None)
        if w is None or w.dim() != 2:
            continue
        if inc and not inc.search(name):
            continue
        if exc and exc.search(name):
            continue
        out.append((name, m))
    return out


class Lora:
    def __init__(self, targets, rank: int, alpha: float, device, seed: int):
        self.rank, self.alpha, self.device = rank, float(alpha), device
        self.scale = self.alpha / rank
        self.shapes = [(name, tuple(m.weight.shape)) for name, m in targets]
        self.names = [n for n, _ in self.shapes]
        self.wrappers = {n: LoraWrapper(self.scale) for n in self.names}
        self._init = None          # warm-start state (architecture v6 L1); every reinit starts from it
        self.reinit(seed)

    def init_from_file(self, path):
        """Warm start from a LoRA file in our export layout. Fail-safe: every target must be present
        with the same rank and shapes, else nothing changes and (False, reason) is returned."""
        try:
            sd = load_file(str(path), device="cpu")
        except Exception as e:  # noqa: BLE001
            return False, f"cannot read {path}: {e}"
        st = {}
        for name, (out_f, in_f) in self.shapes:
            k = f"diffusion_model.{name}"
            up, down = sd.get(f"{k}.lora_up.weight"), sd.get(f"{k}.lora_down.weight")
            if up is None or down is None:
                return False, f"target {name} missing in {path}"
            if tuple(up.shape) != (out_f, self.rank) or tuple(down.shape) != (self.rank, in_f):
                return False, f"shape mismatch at {name}: up {tuple(up.shape)} down {tuple(down.shape)} vs rank {self.rank}"
            st[name] = (up.float(), down.float())
        extra = sum(1 for k in sd if k.endswith(".lora_up.weight")) - len(st)
        self._init = st
        self.load_state(st)
        return True, f"{len(st)} targets from {path}" + (f" ({extra} extra targets in the file ignored)" if extra else "")

    def reinit(self, seed: int):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.params, self.up_params, self.down_params = [], [], []
        for name, (out_f, in_f) in self.shapes:
            down = torch.nn.Parameter((torch.randn(self.rank, in_f, generator=g) / math.sqrt(in_f)).to(self.device))
            up = torch.nn.Parameter(torch.zeros(out_f, self.rank, device=self.device))
            self.params += [up, down]
            self.up_params.append(up)
            self.down_params.append(down)
            w = self.wrappers[name]
            w.up, w.down, w.scale, w.enabled = up, down, self.scale, True
        if getattr(self, "_init", None) is not None:
            self.load_state(self._init)

    def attach(self, patcher):
        for name, w in self.wrappers.items():
            patcher.add_weight_wrapper(f"diffusion_model.{name}.weight", w)

    # --- candidate states ---------------------------------------------------------
    def state(self):
        return {n: (self.wrappers[n].up.detach().clone(), self.wrappers[n].down.detach().clone()) for n in self.names}

    def state_from_flat(self, tensors):
        return {n: (tensors[2 * i], tensors[2 * i + 1]) for i, n in enumerate(self.names)}

    def set_tensors(self, st, scale=None):
        """Temporarily route the wrappers through another state (for scoring a candidate)."""
        for n, (up, down) in st.items():
            w = self.wrappers[n]
            w.up, w.down = up.to(self.device), down.to(self.device)   # candidate states are parked on the CPU
            if scale is not None:
                w.scale = scale

    def restore(self):
        for i, n in enumerate(self.names):
            w = self.wrappers[n]
            w.up, w.down, w.scale = self.params[2 * i], self.params[2 * i + 1], self.scale

    def load_state(self, st):
        with torch.no_grad():
            for i, n in enumerate(self.names):
                self.params[2 * i].copy_(st[n][0])
                self.params[2 * i + 1].copy_(st[n][1])

    def set_enabled(self, flag: bool):
        for w in self.wrappers.values():
            w.enabled = flag


def soup_state(states, weights=None):
    """Average several LoRA states by rank concatenation (exact, no approximation).

    up_i are scaled by the member weights and concatenated along rank; downs are
    concatenated. The merged delta is sum_i w_i * up_i @ down_i, and `LoraLoader`
    reads it as a single rank-(n*r) LoRA with alpha = scale * (n*r).
    """
    n = len(states)
    weights = weights or [1.0 / n] * n
    out = {}
    for name in states[0]:
        ups = [s[name][0] * w for s, w in zip(states, weights)]
        downs = [s[name][1] for s in states]
        out[name] = (torch.cat(ups, 1), torch.cat(downs, 0))
    return out


def export_state_dict(state, scale: float):
    sd = {}
    for name, (up, down) in state.items():
        k = f"diffusion_model.{name}"
        sd[f"{k}.lora_up.weight"] = up.float().cpu().contiguous()
        sd[f"{k}.lora_down.weight"] = down.float().cpu().contiguous()
        sd[f"{k}.alpha"] = torch.tensor(float(scale * down.shape[0]))
    return sd


def save_lora(path, state, scale: float, metadata=None):
    save_file(export_state_dict(state, scale), str(path), metadata=metadata or {})


def enable_checkpointing(diffusion_model, min_len=4, stride=1):
    """Wrap every block of every long ModuleList in torch.utils.checkpoint
    (non-reentrant) so activations fit next to a 9-13B bf16 model on one H100."""
    import torch.utils.checkpoint as cp

    wrapped = 0
    for _, m in diffusion_model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) >= min_len:
            for i, blk in enumerate(m):
                if i % stride or isinstance(blk, torch.nn.ModuleList) or getattr(blk, "_crown_ckpt", False):
                    continue
                orig = blk.forward

                def fwd(*a, _orig=orig, **kw):
                    if torch.is_grad_enabled():
                        return cp.checkpoint(_orig, *a, use_reentrant=False, **kw)
                    return _orig(*a, **kw)

                blk.forward = fwd
                blk._crown_ckpt = True
                wrapped += 1
    return wrapped
