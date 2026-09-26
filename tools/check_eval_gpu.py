"""Gate for the evaluator-GPU emulation (crown/philox.py) on a CUDA device, with weights LARGE enough that the
evaluator's GPU (contract.EVAL_GPU, an A100) and this one draw different stochastic-rounding noise:

  1. philox.self_check: this device's own torch.randint reproduced bit for bit with its own SM count;
  2. after comfy_boot.boot() the patched comfy.float.stochastic_rounding uses the evaluator GPU's geometry:
     equal to comfy_kitchen's rounding on randint_u8(EVAL_GPU) and different from Comfy's own (local) draw;
  3. --exact-merge's rounding (lora._evaluator_fp8_round, fused kernel) equals the patched evaluator call, bitwise;
  4. supports_fp8_compute reports the evaluator GPU's answer.

    python tools/check_eval_gpu.py --comfy-root /workspace/crown-env/ComfyUI
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comfy-root", default=os.environ.get("COMFY_ROOT", "/workspace/crown-env/ComfyUI"))
    a = p.parse_args()
    from crown import comfy_boot
    from crown import contract as C

    comfy_boot.boot(a.comfy_root)
    import torch

    import comfy.float as cf
    import comfy.model_management as mm
    import comfy.utils
    from crown.lora import LoraWrapper, _evaluator_fp8_round
    from crown.philox import randint_u8, self_check

    dev = torch.device("cuda")
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)

    ok, msg = self_check(dev)
    check("philox reproduces this GPU's torch.randint", ok, msg)
    g = C.EVAL_GPU
    check("comfy.float.stochastic_rounding patched with the evaluator GPU", getattr(cf, "_crown_eval_sm_count", None) == g["sm_count"],
          f"{comfy_boot._STATE.get('eval_gpu')}")
    torch.manual_seed(0)
    w = (torch.randn(3072, 3072, device=dev) * 0.05).to(torch.float8_e4m3fn)
    delta = torch.randn(3072, 3072, device=dev) * 1e-3
    merged = w.to(torch.float16) + delta.to(torch.float16)
    seed = comfy.utils.string_to_seed("diffusion_model.double_blocks.0.img_mlp.0.weight")
    evaluator = cf.stochastic_rounding(merged, torch.float8_e4m3fn, seed=seed)
    by_hand = cf._ck_stochastic_rounding_fp8(merged, randint_u8((3072, 3072), seed, g["sm_count"], dev, g["threads_per_sm"]), torch.float8_e4m3fn)
    check("patched rounding == comfy_kitchen on the evaluator GPU's noise", bool(torch.equal(evaluator.view(torch.uint8), by_hand.view(torch.uint8))))
    local = cf._crown_orig_stochastic_rounding(merged, torch.float8_e4m3fn, seed=seed)
    diff = float((local.view(torch.uint8) != evaluator.view(torch.uint8)).float().mean())
    check("this GPU's own draw rounds differently (the pattern is GPU-specific)", diff > 0.01, f"{diff * 100:.2f}% of codes differ")
    wr = LoraWrapper(1.0)
    ours = _evaluator_fp8_round(merged, torch.float8_e4m3fn, seed, torch.bfloat16, wr)
    check("--exact-merge rounding == the evaluator GPU's rounding (bitwise)", bool(torch.equal(ours, evaluator.to(torch.bfloat16))),
          f"{int((ours == evaluator.to(torch.bfloat16)).sum())}/{ours.numel()} equal")
    check("supports_fp8_compute follows the evaluator GPU", mm.supports_fp8_compute(dev) == g["fp8_compute"])
    print(f"\n{sum(results)}/{len(results)} evaluator-GPU checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
