"""Validator entrypoint (ops/docker/*.dockerfile ENTRYPOINT).

Receives the validator's CLI (docs/miner.md "Image Trainer"), resolves cache
paths, launches the trainer with the family recipe, and publishes the artifact
at the exact path the uploader reads. Never exits without trying to leave a
loadable LoRA behind.
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time

START = time.time()
sys.path.insert(0, "/app")
from crown import contract as C  # noqa: E402

# Per-family recipe. Structural choices (which layers, rank) — everything
# adaptive (budget, selection, phase 2) is decided at runtime from measurements.
# Gradient checkpointing stays ON for every family until step time and peak memory are
# measured per family; turning it off is a speed optimisation, not a correctness one.
RECIPES = {
    "krea2":      ["--rank", "16", "--lr", "2e-4"],
    "ideogram4":  ["--rank", "16", "--lr", "2e-4"],
    "z-image":    ["--rank", "16", "--lr", "3e-4"],
    "flux":       ["--rank", "16", "--lr", "3e-4"],
    "qwen-image": ["--rank", "8", "--lr", "1e-4"],
}
COMMON = ["--holdout-frac", "0.10", "--holdout-min", "3", "--eval-share", "0.15",
          "--confirm-top", "3", "--confirm-noises", "4", "--ema", "0.99", "--phase2", "--max-members", "2"]


def logs_enabled():
    return os.environ.get("GOD_TRAIN_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}


def say(*a):
    print(*a, flush=True)


def resolve_model_dir(model):
    from crown.assets import resolve_model_dir as r

    return r(model)


def run_trainer(args, family, model_dir, dataset, work, deadline, extra):
    cmd = [sys.executable, "/app/crown/train.py", "--family", family, "--model-dir", model_dir, "--dataset", dataset,
           "--out", work, "--deadline-ts", str(deadline)] + RECIPES[family] + COMMON + extra
    cmd += shlex.split(os.environ.get("CROWN_EXTRA_ARGS", ""))   # local experiments only; unset on the validator
    if args.trigger_word:
        cmd += ["--trigger-word", args.trigger_word]
    if logs_enabled():
        say("running:", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-zip", required=True)
    p.add_argument("--model-type", required=True)
    p.add_argument("--expected-repo-name", required=True)
    p.add_argument("--hours-to-complete", type=float, required=True)
    p.add_argument("--trigger-word", default=None)
    a = p.parse_args()

    family = a.model_type.lower().replace("_", "-")
    if family not in RECIPES:
        say(f"unsupported model type {a.model_type}")
        sys.exit(2)
    dataset = str(C.dataset_zip(a.task_id))
    if not os.path.exists(dataset) and os.path.exists(a.dataset_zip):
        dataset = a.dataset_zip
    model_dir = resolve_model_dir(a.model)
    out_final = str(C.output_dir(a.task_id, a.expected_repo_name))
    work = str(C.WORK_ROOT / "out")
    os.makedirs(work, exist_ok=True)
    deadline = START + a.hours_to_complete * 3600.0

    rc = run_trainer(a, family, model_dir, dataset, work, deadline, [])
    src = os.path.join(work, C.OUTPUT_LORA_NAME)

    # Fallback: nothing produced (crash before the identity save) and time remains -> identity-only run.
    if not os.path.exists(src) and (deadline - time.time()) > 8 * 60:
        say(f"trainer rc={rc} with no artifact; running identity-only fallback")
        rc2 = run_trainer(a, family, model_dir, dataset, work, deadline, ["--identity-only"])
        say(f"identity-only rc={rc2}")

    os.makedirs(out_final, exist_ok=True)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(out_final, C.OUTPUT_LORA_NAME))
        for f in ("summary.json", "config.json"):
            if os.path.exists(os.path.join(work, f)):
                shutil.copy2(os.path.join(work, f), os.path.join(out_final, f))
        say(f"published {os.path.join(out_final, C.OUTPUT_LORA_NAME)} (trainer rc={rc})")
        sys.exit(0 if rc == 0 else rc)
    say("no checkpoint produced")
    sys.exit(1)


if __name__ == "__main__":
    main()
