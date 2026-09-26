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
from safetensors.torch import load_file, save_file


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
        if w.dtype in (torch.bfloat16, torch.float16):
            # the evaluator merges in fp16 on sm80+ (lora_compute_dtype) and casts back (v6.1 F4)
            return (w.to(torch.float16) + delta.reshape(w.shape).to(torch.float16)).to(w.dtype)
        return (w.float() + delta.reshape(w.shape)).to(w.dtype)

    def to(self, device):
        return self


class _MergedLoraLinear(torch.autograd.Function):
    """Training forward/backward of one LoRA-targeted Comfy Linear under --fast-lora.

    Forward: exactly the weight-wrapper path, y = F.linear(x, W', b) with W' produced by Comfy's
    cast_bias_weight and this module's LoraWrapper (the evaluator's merge numerics), so the values are
    bitwise those of the default path. Backward: dL/dx = dy W' with W' merged again (it is never kept
    alive between forward and backward), and the LoRA gradients through the rank-r activation path,
        dL/dup = scale * dy^T (x down^T),        dL/ddown = scale * (dy up)^T x,
    which is the straight-through gradient of the merge (dL/dW' = dy^T x, contracted with down and up by
    associativity) without ever forming the out x in weight gradient. Per Linear this costs one forward
    and one input-gradient GEMM plus rank-r products, where the default path also runs the full
    weight-gradient GEMM; it keeps only the input activation, so a whole model's activations can fit
    without gradient checkpointing."""

    @staticmethod
    def forward(ctx, x, up, down, module, scale):
        import comfy.ops as ops

        weight, bias, stream = ops.cast_bias_weight(module, x, offloadable=True)
        y = torch.nn.functional.linear(x, weight, bias)
        ops.uncast_bias_weight(module, weight, bias, stream)
        ctx.module, ctx.scale = module, scale
        ctx.save_for_backward(x, up, down)
        return y

    @staticmethod
    def backward(ctx, dy):
        import comfy.ops as ops

        x, up, down = ctx.saved_tensors
        dx = d_up = d_down = None
        if ctx.needs_input_grad[0]:
            weight, bias, stream = ops.cast_bias_weight(ctx.module, x, offloadable=True)
            dx = torch.matmul(dy, weight.to(dy.dtype))
            ops.uncast_bias_weight(ctx.module, weight, bias, stream)
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            x2 = x.reshape(-1, x.shape[-1]).float()
            dy2 = dy.reshape(-1, dy.shape[-1]).float()
            if ctx.needs_input_grad[1]:
                d_up = (dy2.t() @ (x2 @ down.float().t())).mul_(ctx.scale).to(up.dtype)
            if ctx.needs_input_grad[2]:
                d_down = ((dy2 @ up.float()).t() @ x2).mul_(ctx.scale).to(down.dtype)
        return dx, d_up, d_down, None, None


def _fast_forward(module, orig):
    """Replacement for one module's forward_comfy_cast_weights: the merged-forward / rank-r-gradient
    Function while the LoRA trains, the untouched Comfy path otherwise (scoring, frozen LoRA)."""
    def forward_comfy_cast_weights(input, *args, **kwargs):
        w = module._crown_wrapper
        if (args or kwargs or not torch.is_grad_enabled() or not w.enabled or w.up is None
                or not (w.up.requires_grad or w.down.requires_grad)):
            return orig(input, *args, **kwargs)
        return _MergedLoraLinear.apply(input, w.up, w.down, module, w.scale)
    return forward_comfy_cast_weights


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
        self.modules = {name: m for name, m in targets}
        self.wrappers = {n: LoraWrapper(self.scale) for n in self.names}
        self.reinit(seed)

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

    def attach(self, patcher):
        for name, w in self.wrappers.items():
            patcher.add_weight_wrapper(f"diffusion_model.{name}.weight", w)

    def enable_fast_path(self):
        """--fast-lora: route every targeted Linear whose op class uses Comfy's standard cast-then-linear
        forward through _MergedLoraLinear; any other op class keeps the weight-wrapper path (fail closed).
        Returns the number of routed modules."""
        import comfy.ops as ops

        std = {ops.disable_weight_init.Linear.forward_comfy_cast_weights, ops.fp8_ops.Linear.forward_comfy_cast_weights}
        n = 0
        for name, m in self.modules.items():
            if getattr(type(m), "forward_comfy_cast_weights", None) not in std or getattr(m, "_crown_fast", False):
                continue
            m._crown_wrapper = self.wrappers[name]
            m.forward_comfy_cast_weights = _fast_forward(m, m.forward_comfy_cast_weights)
            m._crown_fast = True
            n += 1
        return n

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
