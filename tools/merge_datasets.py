"""Build the training set of a generator prior (architecture v6 L1): every image/caption pair of
every tournament dataset we hold EXCEPT the excluded ones, laid out as one task directory
(NNN.png/.txt + task.json) that tools/prepare_cache.py turns into a cache zip. Images are
family-agnostic (they are nano-banana-2 / gpt-image-2 outputs); the family only decides which base
model the prior is trained against.

    python tools/merge_datasets.py --datasets research/datasets --family krea2 --exclude 5c1de437 \
        --out research/datasets/prior_krea2_x5c1de437
"""
import argparse
import json
import os
import shutil
import sys

REPO = {
    "krea2": "krea/Krea-2-Raw",
    "ideogram4": "gradients-io-tournaments/ideogram-4-fp8",
    "z-image": "gradients-io-tournaments/Z-Image-Turbo",
    "qwen-image": "gradients-io-tournaments/Qwen-Image",
    "flux": "rayonlabs/FLUX.1-dev",
}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", required=True, help="root holding <short>_<family>/ dataset dirs")
    p.add_argument("--family", required=True, choices=sorted(REPO))
    p.add_argument("--exclude", default="", help="comma-separated dataset shorts to leave out (the leave-one-out target)")
    p.add_argument("--out", required=True)
    p.add_argument("--hours", type=float, default=1.0)
    a = p.parse_args(argv)
    excl = {e.strip() for e in a.exclude.split(",") if e.strip()}
    tid = f"prior-{a.family}-x{'-'.join(sorted(excl)) or 'none'}"
    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    os.makedirs(a.out)
    n, used, skipped = 0, [], []
    for d in sorted(os.listdir(a.datasets)):
        src = os.path.join(a.datasets, d)
        if not os.path.isdir(src) or not os.path.exists(os.path.join(src, "task.json")) or d.startswith("prior_"):
            continue
        short = d.split("_")[0]
        if short in excl:
            skipped.append(d)
            continue
        k = 0
        for f in sorted(os.listdir(src)):
            stem, ext = os.path.splitext(f)
            if ext.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            cap = os.path.join(src, stem + ".txt")
            shutil.copy2(os.path.join(src, f), os.path.join(a.out, f"{n:04d}{ext.lower()}"))
            if os.path.exists(cap):
                shutil.copy2(cap, os.path.join(a.out, f"{n:04d}.txt"))
            n += 1
            k += 1
        used.append((d, k))
    meta = {"id": tid, "model_type": a.family, "base_model_repository": REPO[a.family], "hours_to_complete": a.hours,
            "task_type": "ImageTask", "prior": True, "sources": used, "excluded": skipped, "n_images": n}
    json.dump(meta, open(os.path.join(a.out, "task.json"), "w"), indent=1)
    print(f"{tid}: {n} images from {len(used)} datasets; excluded {skipped}")
    print(tid)
    return 0


if __name__ == "__main__":
    sys.exit(main())
