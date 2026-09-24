"""Score several LoRAs (and optionally the base model) with the validator's evaluator on one
holdout directory, and print them side by side — the readout of a paired run.

    python tools/score_pair.py --family krea2 --base /cache/models/krea--Krea-2-Raw/raw.safetensors \
        --holdout-dir /workspace/cache_pair/holdout --out /root/pair/krea2 --noises 16 --with-base \
        ours=/workspace/outputs_pair/ours/<task>/local-xxxx/last.safetensors \
        champ=/workspace/outputs_pair/champ/<task>/champ-xxxx/last.safetensors
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from parity_check import pct, run_eval  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--holdout-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--noises", type=int, default=16)
    p.add_argument("--with-base", action="store_true", help="also score a zeroed copy of the first LoRA (= the base model)")
    p.add_argument("--repo-dir", default=None, help="base repo snapshot dir (z-image)")
    p.add_argument("loras", nargs="+", help="name=path")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    entries = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in a.loras]
    if a.with_base:
        from safetensors.torch import load_file, save_file
        sd = load_file(entries[0][1])
        zpath = os.path.join(a.out, "identity.safetensors")
        save_file({k: (v * 0 if k.endswith("lora_up.weight") else v) for k, v in sd.items()}, zpath)
        entries.insert(0, ("base", zpath))
    results = {}
    for name, path in entries:
        if not os.path.exists(path):
            print(f"{name}: MISSING {path}")
            continue
        r = run_eval(a.family, a.base, path, a.holdout_dir, os.path.join(a.out, f"{name}_n{a.noises}.json"), a.noises, a.repo_dir)
        results[name] = r
        print(f"{name:>8}: {r['score']:.6f}  per-image {[round(x, 6) for x in r['per_image']]}")
    base = results.get("base", {}).get("score")
    print("--- evaluator @%d noises on %d held-out images ---" % (a.noises, len(next(iter(results.values()))["per_image"])))
    for name, r in results.items():
        rel = f"{pct(r['score'], base):+.3f}% vs base" if base and name != "base" else ""
        print(f"{name:>8}  {r['score']:.6f}  {rel}")
    names = [n for n in results if n != "base"]
    if len(names) >= 2:
        x, y = names[0], names[1]
        d = pct(results[x]["score"], results[y]["score"])
        wins = sum(1 for i, j in zip(results[x]["per_image"], results[y]["per_image"]) if i < j)
        print(f"{x} vs {y}: {d:+.3f}% (negative = {x} better); {x} wins {wins}/{len(results[x]['per_image'])} images; "
              f"{'BEATS the >1% bar' if d < -1.0 else 'does NOT clear the >1% bar'}")
    json.dump(results, open(os.path.join(a.out, "pair.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
