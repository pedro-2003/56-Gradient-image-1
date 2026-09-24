"""G-PARITY instrument: score a local LoRA with the *validator's own evaluator
code* on a local image set, offline.

The evaluator (validator/evaluation/evaluators/diffusion.py) resolves the base
model and each candidate LoRA through HuggingFace. This wrapper runs the same
`evaluate()` with those two resolvers pointed at local files, so the numbers it
prints are exactly what the validator would compute for that LoRA on those
images.

Runs inside the validator's evaluator image, or in our venv via the vendored copy of
the evaluator (tools/evaluator_vendor, verbatim from upstream, imports rewritten):

    python local_evaluate.py --family krea2 --base /path/raw.safetensors \
        --lora /outputs/<task>/<repo>/checkpoints/last.safetensors \
        --dataset /data/holdout_dir_or_zip --out /tmp/eval.json [--strata 16 --noises 16]

Compare `eval_loss` per image against the trainer's summary.json for the same
held-out images: bf16 families must agree to < 0.05%; the gap on ideogram4 /
qwen-image is the parity effect being measured.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--base", required=True, help="local base checkpoint file the evaluator would select")
    p.add_argument("--lora", required=True, help="local last.safetensors")
    p.add_argument("--dataset", required=True, help="directory or zip of held-out images + captions")
    p.add_argument("--out", required=True)
    p.add_argument("--comfy-root", default=os.environ.get("COMFY_ROOT", "/app/validator/evaluation/ComfyUI"))
    p.add_argument("--strata", type=int, default=16)
    p.add_argument("--noises", type=int, default=16)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--repo-dir", default=None,
                   help="local snapshot of the base repo; z-image needs it because the evaluator loads that family's VAE with diffusers from the base repo")
    a = p.parse_args()

    # Prefer the validator's own package when running inside its image; otherwise use the
    # verbatim vendored copy (tools/evaluator_vendor) so this works in the venv, no docker.
    sys.path.insert(0, "/app")
    try:
        import validator.evaluation.image_artifacts as artifacts
        import validator.evaluation.evaluators.diffusion as diffusion
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import evaluator_vendor.image_artifacts as artifacts
        import evaluator_vendor.evaluators.diffusion as diffusion

    root = Path(a.comfy_root)
    folder = "unet" if a.family == "flux" else "diffusion_models"
    (root / "models" / folder).mkdir(parents=True, exist_ok=True)
    (root / "models" / "loras").mkdir(parents=True, exist_ok=True)
    # Comfy resolves "diffusion_models" names through models/unet FIRST, so a stale link left by a
    # flux run would silently replace every other family's base: use per-family names and sweep
    # both folders of every earlier local_* link.
    base_name = f"local_base_{a.family}.safetensors"
    lora_name = f"local_lora_{a.family}.safetensors"
    for d in ("unet", "diffusion_models", "loras"):
        for stale in (root / "models" / d).glob("local_*.safetensors"):
            stale.unlink()
    for src, dst in ((a.base, root / "models" / folder / base_name), (a.lora, root / "models" / "loras" / lora_name)):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(Path(src).resolve())
        except OSError:
            shutil.copy2(src, dst)

    def local_prepare_base(api, task, root_):
        return base_name, {"repo": "local", "revision": "local", "filename": base_name}

    def local_materialize(api, repo, filename, directory):
        if filename is None:                      # the candidate LoRA
            return lora_name, {"repo": repo, "revision": "local", "filename": lora_name}
        return artifacts.materialize_model(api, repo, filename, directory)  # text encoders / VAE still from HF cache

    diffusion.prepare_base = local_prepare_base
    diffusion.materialize_model = local_materialize

    config = {"dataset": a.dataset, "repo": a.repo_dir or "local/base", "family": a.family, "models": ["local/lora"],
              "comfy_root": root, "strata": a.strata, "noises": a.noises, "batch_size": a.batch}
    result = diffusion.evaluate(config, Path(a.out))
    r = result.get("local/lora")
    if isinstance(r, dict):
        t, n = r["eval_loss"]["text_guided_losses"], r["eval_loss"]["no_text_losses"]
        per = [0.5 * x + 0.5 * y for x, y in zip(t, n)]
        print(json.dumps({"score": sum(per) / len(per), "per_image": per, "text": t, "no_text": n}, indent=1))
    else:
        print("EVALUATION FAILED:", r)
        sys.exit(1)


if __name__ == "__main__":
    main()
