"""How much does the evaluator's fp8 re-quantisation cost per patched tensor group on ideogram4?

The evaluator merges every LoRA key into the fp8 base (dequantise, add, re-quantise with a fresh
scale and stochastic rounding). Even a ZERO LoRA therefore perturbs each patched tensor. This probe
scores zeroed copies of a real LoRA restricted to key subsets with the validator's own evaluator on
the same images, so the per-group cost of merely carrying keys is measured directly.

    python tools/probe_requant.py --lora <last.safetensors> --base <ideogram4_fp8_scaled.safetensors> \
        --holdout-dir /workspace/cache_pair_ideo/holdout --out /root/probe_requant [--noises 1]
"""
import argparse
import json
import os
import re
import sys

from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parity_check import run_eval  # noqa: E402

GROUPS = {
    "all": r".*",
    "blocks": r"^diffusion_model\.layers\.\d+\.",
    "attention": r"^diffusion_model\.layers\.\d+\.attention\.",
    "feed_forward": r"^diffusion_model\.layers\.\d+\.feed_forward\.",
    "adaln": r"^diffusion_model\.layers\.\d+\.adaln_modulation",
    "non_block": r"^diffusion_model\.(?!layers\.)",
    "one_qkv": r"^diffusion_model\.layers\.0\.attention\.qkv\.",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--holdout-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--family", default="ideogram4")
    p.add_argument("--noises", type=int, default=1)
    p.add_argument("--groups", default=",".join(GROUPS))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sd = load_file(a.lora)
    zero = {k: (v * 0 if k.endswith("lora_up.weight") else v) for k, v in sd.items()}
    results = {}
    for g in a.groups.split(","):
        pat = re.compile(GROUPS[g])
        sub = {k: v for k, v in zero.items() if pat.match(k)}
        n = sum(1 for k in sub if k.endswith(".alpha"))
        if not sub:
            print(f"{g}: no keys"); continue
        path = os.path.join(a.out, f"zero_{g}.safetensors")
        save_file(sub, path)
        r = run_eval(a.family, a.base, path, a.holdout_dir, os.path.join(a.out, f"{g}_n{a.noises}.json"), a.noises)
        results[g] = {"modules": n, "score": r["score"], "per_image": r["per_image"]}
        print(f"{g:>13} modules={n:3d}  zero-LoRA score {r['score']:.6f}")
    ref = results.get("all", {}).get("score")
    for g, r in results.items():
        if ref:
            print(f"{g:>13}: {(r['score'] / ref - 1) * 100:+.3f}% vs all-keys")
    json.dump(results, open(os.path.join(a.out, "requant.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
