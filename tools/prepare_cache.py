"""Build a local /cache in the validator's layout for one task, so
tools/local_run.sh can exercise the trainer exactly as production does.

Mirrors trainer/containers/downloader.py (G.O.D): base model snapshot under
models/{model_id with '/'->'--'}, dataset zip under datasets/{task}_tourn.zip,
and the family's text-encoder snapshot under hf_cache/. Needs network and, for
gated repos (krea/Krea-2-Raw, FLUX), an HF token in HF_TOKEN.

    python tools/prepare_cache.py --cache ./cache --task-dir research/datasets/5c1de437_krea2 \
        --model krea/Krea-2-Raw --model-type krea2
"""

import argparse
import json
import os
import re
import zipfile
from pathlib import Path

TEXT_ENCODER_REPO = {"ideogram4": "Qwen/Qwen3-VL-8B-Instruct", "krea2": "Qwen/Qwen3-VL-4B-Instruct"}

# trainer/containers/downloader.py (G.O.D): a FLUX repo with exactly one root .safetensors, no
# model_index.json, no diffusers component dirs, no weight index and no sharding is downloaded as
# that single file only; everything else is a full snapshot.
DIFFUSERS_COMPONENT_DIRS = {"scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "transformer", "unet", "vae"}
WEIGHT_INDEX_SUFFIXES = (".bin.index.json", ".safetensors.index.json")
SHARDED_RE = re.compile(r"-[0-9]{5}-of-[0-9]{5}[.]safetensors$")


def standalone_flux_checkpoint(repo_id: str, model_type: str, token=None):
    """The single root file the validator would keep for this flux repo, or None (full snapshot)."""
    if model_type != "flux":
        return None
    from huggingface_hub import HfApi
    files = [f.rfilename for f in HfApi().model_info(repo_id, token=token, files_metadata=False).siblings]
    roots = [f for f in files if "/" not in f and f.endswith(".safetensors")]
    if len(roots) != 1:
        return None
    if "model_index.json" in files:
        return None
    if any(f.split("/")[0] in DIFFUSERS_COMPONENT_DIRS for f in files if "/" in f):
        return None
    if any(f.endswith(WEIGHT_INDEX_SUFFIXES) for f in files):
        return None
    if SHARDED_RE.search(roots[0]):
        return None
    return roots[0]


def make_task_zip(task_dir: Path, task_id: str, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(task_dir.iterdir()):
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".txt"):
                z.write(f, f.name)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--task-dir", required=True, help="directory with NNN.png/NNN.txt (+ task.json)")
    p.add_argument("--model", required=True, help="HF repo id of the base model (base_model_repository)")
    p.add_argument("--model-type", required=True)
    p.add_argument("--task-id", default=None)
    p.add_argument("--skip-model", action="store_true")
    a = p.parse_args()

    cache = Path(a.cache)
    task_dir = Path(a.task_dir)
    meta = json.load(open(task_dir / "task.json")) if (task_dir / "task.json").exists() else {}
    task_id = a.task_id or meta.get("id") or task_dir.name.split("_")[0]
    z = make_task_zip(task_dir, task_id, cache / "datasets" / f"{task_id}_tourn.zip")
    print("dataset zip:", z)

    if a.skip_model:
        return
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    mdir = cache / "models" / a.model.replace("/", "--")
    single = standalone_flux_checkpoint(a.model, a.model_type, token)
    if single:
        from huggingface_hub import hf_hub_download
        print("standalone FLUX checkpoint (validator rule): downloading only", single, "->", mdir)
        mdir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=a.model, filename=single, local_dir=str(mdir), token=token)
        for f in mdir.iterdir():   # the validator normalises the cache to that one file
            if f.name != single and (f.is_file() or f.is_symlink()):
                f.unlink()
    else:
        print("downloading base model", a.model, "->", mdir)
        snapshot_download(repo_id=a.model, repo_type="model", local_dir=str(mdir), token=token)
    te = TEXT_ENCODER_REPO.get(a.model_type)
    if te:
        tdir = cache / "hf_cache" / te.replace("/", "--")
        print("downloading text encoder", te, "->", tdir)
        snapshot_download(repo_id=te, repo_type="model", local_dir=str(tdir), token=token)
    print("cache ready:", cache)


if __name__ == "__main__":
    main()
