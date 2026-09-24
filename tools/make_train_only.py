"""Build a second /cache root for PAIRED runs: the same models and hf_cache (symlinked), but the
task zip holds only the training images, and the held-out images live in <out>/holdout for
scoring both trainers' LoRAs with the evaluator afterwards.

    python tools/make_train_only.py --dataset-dir research/datasets/5c1de437_krea2 --task-id <uuid> \
        --cache /workspace/cache --out /workspace/cache_pair [--holdout-frac 0.2 --holdout-min 5]

The split is crown.data.choose_holdout (stratified by aspect and caption length, seeded from the
image digests), so it is reproducible from the dataset alone.
"""
import argparse
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from crown import data  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--cache", required=True, help="the full cache (models/, hf_cache/)")
    p.add_argument("--out", required=True, help="the paired cache root to create")
    p.add_argument("--holdout-frac", type=float, default=0.2)
    p.add_argument("--holdout-min", type=int, default=5)
    a = p.parse_args()

    items = data.load_items(a.dataset_dir)
    hold = set(data.choose_holdout(items, a.holdout_frac, a.holdout_min))
    names_h = [items[i]["name"] for i in sorted(hold)]
    names_t = [it["name"] for i, it in enumerate(items) if i not in hold]

    os.makedirs(os.path.join(a.out, "datasets"), exist_ok=True)
    for sub in ("models", "hf_cache"):
        link = os.path.join(a.out, sub)
        if not os.path.exists(link):
            os.symlink(os.path.join(os.path.abspath(a.cache), sub), link)
    hdir = os.path.join(a.out, "holdout")
    shutil.rmtree(hdir, ignore_errors=True)
    os.makedirs(hdir)

    def stem_files(n):
        s = os.path.splitext(n)[0]
        return [n, s + ".txt"]

    zpath = os.path.join(a.out, "datasets", f"{a.task_id}_tourn.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for n in names_t:
            for f in stem_files(n):
                z.write(os.path.join(a.dataset_dir, f), arcname=f)
    for n in names_h:
        for f in stem_files(n):
            shutil.copy2(os.path.join(a.dataset_dir, f), os.path.join(hdir, f))
    print(f"train {len(names_t)} -> {zpath}")
    print(f"holdout {len(names_h)} -> {hdir}: {names_h}")


if __name__ == "__main__":
    main()
