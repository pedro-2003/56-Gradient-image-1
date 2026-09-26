"""Runtime gate for --fast-lora on the REAL Comfy ops (CPU is enough; no model weights needed).

For each Comfy Linear op class the trainer can meet (bf16 disable_weight_init, fp8 manual_cast, fp8
fp8_ops), a LoraWrapper is installed as the module's weight function exactly as ModelPatcher does, and
the same input / LoRA state go through (A) the default weight-wrapper path and (B) the fast path:
  * forward values must be bitwise equal (both run cast_bias_weight + the wrapper's merge + F.linear);
  * dL/dx must agree to bf16 rounding; dL/dup and dL/ddown to the rounding of the default path's full
    bf16 weight gradient (the fast path contracts in fp32, so it is the more precise of the two);
  * without grad (scoring) the fast path must fall through to the untouched Comfy forward;
  * a two-Linear chain checks that gradients propagate through routed modules.

    CUDA_VISIBLE_DEVICES= python tools/check_fast_lora.py --comfy-root /workspace/crown-env/ComfyUI
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / max(float(b.norm()), 1e-30))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comfy-root", default=os.environ.get("COMFY_ROOT", "/workspace/crown-env/ComfyUI"))
    a = p.parse_args()
    from crown import comfy_boot

    sys.path.insert(0, os.path.abspath(a.comfy_root))
    import comfy.cli_args as cli

    cli.args.cpu = True          # Comfy needs its CPU mode set before model_management imports without a GPU
    comfy_boot.boot(a.comfy_root)
    import torch

    import comfy.ops as ops
    from crown.lora import Lora

    torch.manual_seed(0)
    results = []

    def make(cls, wdtype, in_f, out_f):
        m = cls.Linear(in_f, out_f, bias=True, dtype=wdtype)
        with torch.no_grad():
            m.weight = torch.nn.Parameter((torch.randn(out_f, in_f) * 0.05).to(wdtype), requires_grad=False)
            m.bias = torch.nn.Parameter((torch.randn(out_f) * 0.01).to(torch.bfloat16 if wdtype != torch.bfloat16 else wdtype),
                                        requires_grad=False)
        return m

    def lora_for(mods, rank=4, alpha=2.0, seed=1):
        lo = Lora([(f"m{i}", m) for i, m in enumerate(mods)], rank, alpha, "cpu", seed)
        g = torch.Generator().manual_seed(seed + 7)
        with torch.no_grad():
            for up in lo.up_params:          # a non-zero state so the merge matters
                up.copy_(torch.randn(up.shape, generator=g) * 0.05)
        for i, m in enumerate(mods):
            m.weight_function = [lo.wrappers[f"m{i}"]]    # what ModelPatcher does for add_weight_wrapper
        return lo

    def reference(mods, lo, x):
        """fp64 straight-through gradient of the same function: the merged weight's VALUE (as the default
        path computes it) plus a zero-valued fp64 LoRA term that carries dL/dup, dL/ddown exactly."""
        ups = [p.detach().double().requires_grad_(True) for p in lo.up_params]
        downs = [p.detach().double().requires_grad_(True) for p in lo.down_params]
        h = x.detach().double()
        for i, m in enumerate(mods):
            w = lo.wrappers[f"m{i}"]
            with torch.no_grad():
                wq = ops.cast_bias_weight(m, x.detach(), offloadable=False)[0].double()   # the merged W' value
                b = m.bias.detach().double()
            d = (ups[i] @ downs[i]) * w.scale
            h = torch.nn.functional.linear(h, wq + (d - d.detach()), b)
            if i + 1 < len(mods):
                h = torch.nn.functional.gelu(h)
        (h ** 2).mean().backward()
        return [t for pair in zip(ups, downs) for t in (pair[0].grad, pair[1].grad)]

    def run(mods, lo, x, fast):
        if fast:
            lo.enable_fast_path()
        xx = x.detach().clone().requires_grad_(x.requires_grad)
        h = xx
        for i, m in enumerate(mods):
            h = m(h)
            if i + 1 < len(mods):
                h = torch.nn.functional.gelu(h)
        loss = (h.float() ** 2).mean()
        loss.backward()
        gx = xx.grad.detach().clone() if xx.requires_grad else None
        return h.detach().clone(), gx, [p.grad.detach().clone() for p in lo.params]

    cases = [("bf16 disable_weight_init", ops.disable_weight_init, torch.bfloat16),
             ("fp8 manual_cast", ops.manual_cast, torch.float8_e4m3fn),
             ("fp8 fp8_ops", ops.fp8_ops, torch.float8_e4m3fn)]
    for label, cls, wdt in cases:
        for chain, x_grad in ((1, True), (2, True), (1, False)):
            dims = [(64, 48), (48, 40)][:chain]
            x = torch.randn(2, 37, 64).to(torch.bfloat16)
            x.requires_grad_(x_grad)
            outs = []
            for fast in (False, True):
                torch.manual_seed(5)
                mods = [make(cls, wdt, i, o) for i, o in dims]
                lo = lora_for(mods)
                outs.append((run(mods, lo, x, fast), mods, lo))
            (ya, gxa, gpa), _, _ = outs[0]
            (yb, gxb, gpb), mods_b, lo_b = outs[1]
            ref = reference(outs[0][1], outs[0][2], x)
            err_a = max(rel(g, r) for g, r in zip(gpa, ref))
            err_b = max(rel(g, r) for g, r in zip(gpb, ref))
            routed = sum(1 for m in mods_b if getattr(m, "_crown_fast", False))
            bitwise = bool(torch.equal(ya, yb))
            gx_err = rel(gxb, gxa) if x_grad else 0.0
            gp_err = max(rel(b, a) for a, b in zip(gpa, gpb))
            with torch.no_grad():
                h = x.detach()
                for i, m in enumerate(mods_b):
                    h = m(h)
                    if i + 1 < len(mods_b):
                        h = torch.nn.functional.gelu(h)
            infer_ok = bool(torch.equal(h, ya))
            ok = routed == len(dims) and bitwise and gx_err < 2e-2 and gp_err < 2e-2 and infer_ok and err_b <= err_a * 1.05
            results.append(ok)
            print(f"{'PASS' if ok else 'FAIL'} {label:26} chain {chain} x_grad {x_grad!s:5}: routed {routed}/{len(dims)}, "
                  f"forward bitwise {bitwise}, rel err dx {gx_err:.2e}, LoRA grads fast-vs-default {gp_err:.2e}, "
                  f"vs fp64: default {err_a:.2e} fast {err_b:.2e}, no-grad path {infer_ok}")
    print(f"\n{sum(results)}/{len(results)} fast-LoRA checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
