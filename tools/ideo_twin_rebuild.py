"""Can the evaluator's ideogram4 base (Comfy-Org ideogram4_fp8_scaled.safetensors, 8.7 GB baked into the
image only for the evaluator twin) be rebuilt from the task cache instead?

Both files hold the same 211 fp8 Linears and 247 bf16 tensors under the same names; the evaluator's file
has ONE scale per tensor (`weight_scale` scalar + a `comfy_quant` marker), the cache ONE scale per row.
Rebuild: dequantise the cache (fp8 x row scale) and cast to e4m3 with scale 1.0, which is how the
evaluator's file was made (its weight_scale is 1.0 on every tensor). Codes can differ only where the
cache's row-grid value and the original weight fall on different sides of an e4m3 rounding boundary.

Reports, per tensor and in total: scale agreement, the share of fp8 codes that match the evaluator's file,
and ||W_rebuilt - W_eval|| / ||W_eval||; bf16 tensors are compared bitwise. With --out, writes the rebuilt
file in the evaluator's exact layout (weight, weight_scale, comfy_quant copied, bias/bf16 tensors).

    python tools/ideo_twin_rebuild.py --eval /opt/crown/assets/ideogram4_fp8_scaled.safetensors \
        --cache /cache/models/gradients-io-tournaments--ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors \
        [--out /tmp/ideogram4_rebuilt.safetensors]
"""

import argparse

import torch
from safetensors import safe_open
from safetensors.torch import save_file

E4M3_MAX = 448.0


def rebuild(w8, row_scale):
    """The evaluator's file stores every quantised Linear as the raw weights cast to e4m3 with
    weight_scale = 1.0 (measured 2026-09-26: scale 1.0 on all 211 tensors, max |code| 0.08-0.44), so the
    rebuild casts the cache's dequantised weights the same way."""
    w = w8.float() * row_scale.float().reshape(-1, *([1] * (w8.dim() - 1)))
    q = w.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, torch.tensor(1.0, dtype=torch.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    fe, fc = safe_open(a.eval, "pt", device="cpu"), safe_open(a.cache, "pt", device="cpu")
    ek, ck = set(fe.keys()), set(fc.keys())
    lin = sorted(k[: -len(".weight")] for k in ek if k.endswith(".weight") and (k[: -len(".weight")] + ".weight_scale") in ek)
    out = {} if a.out else None
    n_el = n_same = 0
    num = den = 0.0
    worst = []
    for name in lin:
        we, se = fe.get_tensor(name + ".weight"), fe.get_tensor(name + ".weight_scale").float()
        q, sr = rebuild(fc.get_tensor(name + ".weight"), fc.get_tensor(name + ".weight_scale"))
        same = int((q.view(torch.uint8) == we.view(torch.uint8)).sum())
        de, dr = we.float() * se, q.float() * sr
        err = float((dr - de).norm() / de.norm())
        n_el += we.numel()
        n_same += same
        num += float((dr - de).norm()) ** 2
        den += float(de.norm()) ** 2
        worst.append((err, name, float(abs(sr / se - 1)), same / we.numel()))
        if out is not None:
            out[name + ".weight"] = q
            out[name + ".weight_scale"] = sr
            out[name + ".comfy_quant"] = fe.get_tensor(name + ".comfy_quant")
            if (name + ".bias") in ek:
                out[name + ".bias"] = fc.get_tensor(name + ".bias") if (name + ".bias") in ck else fe.get_tensor(name + ".bias")
    other = sorted(k for k in ek if not any(k == n + s for n in lin for s in (".weight", ".weight_scale", ".comfy_quant", ".bias")))
    bf_same = 0
    for k in other:
        te, tc = fe.get_tensor(k), fc.get_tensor(k) if k in ck else None
        if tc is not None and te.dtype == tc.dtype and torch.equal(te, tc):
            bf_same += 1
        if out is not None:
            out[k] = tc if tc is not None else te
    worst.sort(reverse=True)
    print(f"fp8 Linears: {len(lin)}; code agreement {n_same / n_el * 100:.3f}% of {n_el / 1e9:.2f} G elements; "
          f"total ||rebuilt - eval|| / ||eval|| = {(num / den) ** 0.5:.3e}")
    print(f"max per-tensor scale deviation {max(w[2] for w in worst):.2e}")
    for err, name, sdev, share in worst[:5]:
        print(f"   worst {name}: rel err {err:.3e}, scale dev {sdev:.1e}, codes equal {share * 100:.2f}%")
    print(f"other tensors (bf16 etc.): {bf_same}/{len(other)} bitwise identical to the evaluator file")
    if out is not None:
        meta = safe_open(a.eval, "pt", device="cpu").metadata()
        save_file(out, a.out, metadata=meta)
        print(f"wrote {a.out} ({len(out)} tensors, metadata {meta})")


if __name__ == "__main__":
    main()
