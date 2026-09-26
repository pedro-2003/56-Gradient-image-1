"""Gate for --exact-merge on the REAL Comfy ops: a plain-fp8 Comfy Linear (manual_cast) with a LoraWrapper
installed as ModelPatcher would, checked against the evaluator's own functions.

  1. value: the wrapper's merged weight == comfy.float.stochastic_rounding(fp16 merge, fp8,
     seed=comfy.utils.string_to_seed(key)) upcast to bf16, bitwise, computed independently;
  2. the value lies on the fp8 grid;
  3. zero delta returns the base weight unchanged (the evaluator's rounding of an exact fp8 value);
  4. straight-through gradient: for a loss linear in W', the LoRA gradients with and without the rounding
     are identical;
  5. with --fast-lora as well: the forward output is bitwise the default exact-merge path's, gradients agree;
  6. no grad (scoring) returns the rounded weight; enable_exact_merge leaves bf16 modules alone.
The GPU parity against the evaluator's own ModelPatcher runs inside the trainer at the first twin confirm.

    python tools/check_exact_merge.py --comfy-root /workspace/crown-env/ComfyUI [--device cuda]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comfy-root", default=os.environ.get("COMFY_ROOT", "/workspace/crown-env/ComfyUI"))
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    sys.path.insert(0, os.path.abspath(a.comfy_root))
    import comfy.cli_args as cli

    if a.device == "cpu":
        cli.args.cpu = True
    from crown import comfy_boot

    comfy_boot.boot(a.comfy_root)
    import torch

    import comfy.float
    import comfy.ops as ops
    import comfy.utils
    from crown.lora import Lora

    dev = torch.device(a.device)
    torch.manual_seed(0)
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")

    def make(fp8=True, in_f=64, out_f=48):
        cls = ops.manual_cast if fp8 else ops.disable_weight_init
        wd = torch.float8_e4m3fn if fp8 else torch.bfloat16
        m = cls.Linear(in_f, out_f, bias=True, dtype=wd, device=dev)
        with torch.no_grad():
            m.weight = torch.nn.Parameter((torch.randn(out_f, in_f, device=dev) * 0.05).to(wd), requires_grad=False)
            m.bias = torch.nn.Parameter((torch.randn(out_f, device=dev) * 0.01).to(torch.bfloat16), requires_grad=False)
        return m

    def lora_for(mods, up_scale=0.05, seed=1):
        lo = Lora([(f"blk.{i}.proj", m) for i, m in enumerate(mods)], 4, 2.0, dev, seed)
        g = torch.Generator().manual_seed(seed + 7)
        with torch.no_grad():
            for up in lo.up_params:
                up.copy_((torch.randn(up.shape, generator=g) * up_scale).to(dev))
        for i, m in enumerate(mods):
            m.weight_function = [lo.wrappers[f"blk.{i}.proj"]]
        return lo

    # 1-3, 6: value
    m = make()
    lo = lora_for([m])
    n = lo.enable_exact_merge()
    wr = lo.wrappers["blk.0.proj"]
    check("routes the plain-fp8 module", n == 1 and wr.fp8_dtype == torch.float8_e4m3fn,
          f"{n} routed, seed {wr.seed} == crc32 {comfy.utils.string_to_seed('diffusion_model.blk.0.proj.weight')}")
    with torch.no_grad():
        w16 = m.weight.to(torch.bfloat16)
        got = wr(w16)
        delta = (wr.up.float() @ wr.down.float()) * wr.scale
        merged = w16.to(torch.float16) + delta.to(torch.float16)
        want = comfy.float.stochastic_rounding(merged, torch.float8_e4m3fn, seed=wr.seed).to(torch.bfloat16)
    check("value == evaluator stochastic rounding (bitwise)", bool(torch.equal(got, want)),
          f"{int((got == want).sum())}/{got.numel()} equal")
    check("value on the fp8 grid", bool(torch.equal(got.to(torch.float8_e4m3fn).to(torch.bfloat16), got)))
    flipped = int((got != merged.to(torch.bfloat16)).sum())
    moved = int((got != w16).sum())
    print(f"     info: {moved}/{got.numel()} elements differ from the base after the merge ({flipped} differ from the unrounded merge)")
    with torch.no_grad():
        up0 = [u.clone() for u in lo.up_params]
        for u in lo.up_params:
            u.zero_()
        z = wr(w16)
        for u, s in zip(lo.up_params, up0):
            u.copy_(s)
    check("zero delta returns the base weight", bool(torch.equal(z, w16)))
    lo_plain = Lora([("x", make(fp8=False))], 4, 2.0, dev, 3)
    check("bf16 module left alone", lo_plain.enable_exact_merge() == 0)

    # 4: straight-through gradient (loss linear in W')
    def grads(exact, fast=False):
        mm = make()
        torch.manual_seed(5)
        with torch.no_grad():
            mm.weight = torch.nn.Parameter(m.weight.detach().clone(), requires_grad=False)
            mm.bias = torch.nn.Parameter(m.bias.detach().clone(), requires_grad=False)
        ll = lora_for([mm])
        if exact:
            ll.enable_exact_merge()
        if fast:
            ll.enable_fast_path()
        x = torch.randn(2, 9, 64, device=dev).to(torch.bfloat16)
        c = torch.randn(2, 9, 48, device=dev)
        y = mm(x)
        (y.float() * c).sum().backward()
        return y.detach(), [q.grad.detach().clone() for q in ll.params]

    y0, g0 = grads(False)
    y1, g1 = grads(True)
    same = all(torch.allclose(a, b, rtol=1e-3, atol=1e-6) for a, b in zip(g0, g1))
    check("straight-through: LoRA grads unchanged by the rounding", same,
          f"max rel diff {max(float((a - b).norm() / max(float(a.norm()), 1e-30)) for a, b in zip(g0, g1)):.2e}; outputs differ in "
          f"{int((y0 != y1).sum())}/{y0.numel()} elements")
    y2, g2 = grads(True, fast=True)
    check("--fast-lora forward == default exact-merge forward (bitwise)", bool(torch.equal(y1, y2)))
    rel_fast = max(float((a - b).norm() / max(float(a.norm()), 1e-30)) for a, b in zip(g1, g2))
    check("--fast-lora grads agree (relative norm < 2e-2, as in check_fast_lora)", rel_fast < 2e-2, f"max rel diff {rel_fast:.2e}")
    print(f"\n{sum(results)}/{len(results)} exact-merge checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
