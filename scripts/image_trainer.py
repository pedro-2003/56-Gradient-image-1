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
    "ideogram4":  ["--rank", "16", "--lr", "2e-4", "--include", "layers[.][0-9]+[.](attention|adaln_modulation|feed_forward)", "--no-ckpt", "--train-base", "requant"],   # requant: measured -1.03% vs the dequantised base (2026-09-25, af4138aa), closes the gap to the champion   # block layers only: the evaluator re-quantises every patched tensor; the 7 embedder/final-layer tensors alone cost 0.7% (probe_requant 2026-09-24)   # twin released between scores -> fits: 0.68 s/step, peak 64.7 GB (smoke 2026-09-24); the OOM guard re-enables checkpointing if a task needs it
    "z-image":    ["--rank", "16", "--lr", "3e-4"],
    "flux":       ["--rank", "16", "--lr", "3e-4"],
    "qwen-image": ["--rank", "8", "--lr", "1e-4", "--include", "transformer_blocks[.][0-9]+[.]attn[.](to_q|to_k|to_v|to_out)"],   # the public champion recipe's target set (240 vs 846 Linears): 1.47 -> ~1.05 s/step; Round 2 can be a qwen knockout
}
COMMON = ["--holdout-frac", "0.10", "--holdout-min", "3", "--eval-share", "0.15",
          "--confirm-top", "3", "--confirm-noises", "4", "--ema", "0.99", "--phase2", "--max-members", "2"]


def logs_enabled():
    # on by default: the validator keeps container logs, and one line per 25 steps is the only
    # diagnostic a failed task leaves behind (set GOD_TRAIN_LOGS=0 to silence)
    return os.environ.get("GOD_TRAIN_LOGS", "1").strip().lower() not in {"0", "false", "no", "off"}


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
    # the child must be gone before the validator's own clock (deadline) kills the container:
    # whatever it saved by then is already in /app/checkpoints (see main), so a kill loses nothing
    limit = max(30.0, deadline - time.time() - 60.0)
    proc = subprocess.Popen(cmd)
    try:
        return proc.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        say(f"trainer exceeded its wall-clock ({limit:.0f}s); terminating it, the published artifact stands")
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return 124


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task-id", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset-zip", required=True)
    p.add_argument("--model-type", required=True)
    p.add_argument("--expected-repo-name", required=True)
    p.add_argument("--hours-to-complete", type=float, required=True)
    p.add_argument("--trigger-word", default=None)
    a, unknown = p.parse_known_args()
    if unknown:
        say(f"ignoring unknown validator arguments: {unknown}")
    os.environ.setdefault("GOD_TRAIN_LOGS", "1")

    family = a.model_type.lower().replace("_", "-")
    if family not in RECIPES:
        say(f"unsupported model type {a.model_type}")
        sys.exit(2)
    dataset = str(C.dataset_zip(a.task_id))
    if not os.path.exists(dataset) and os.path.exists(a.dataset_zip):
        dataset = a.dataset_zip
    model_dir = resolve_model_dir(a.model)
    out_final = str(C.output_dir(a.task_id, a.expected_repo_name))
    # the trainer writes straight into the validator's checkpoints volume (atomic tmp + os.replace per
    # save), so an artifact is on the volume from the identity save onwards: a timeout kill, a crash
    # or a hang after that point still leaves the best-so-far where the uploader reads it
    os.makedirs(out_final, exist_ok=True)
    work = out_final
    deadline = START + a.hours_to_complete * 3600.0

    rc = run_trainer(a, family, model_dir, dataset, work, deadline, [])
    src = os.path.join(work, C.OUTPUT_LORA_NAME)

    # Fallback: nothing produced (crash before the identity save) and time remains -> lean identity-only
    # run (diffusion weights only: no dataset, VAE or text encoders)
    if not os.path.exists(src) and (deadline - time.time()) > 3 * 60:
        say(f"trainer rc={rc} with no artifact; running the identity-only fallback")
        rc2 = run_trainer(a, family, model_dir, dataset, work, deadline, ["--identity-only"])
        say(f"identity-only rc={rc2}")

    if os.path.exists(src):
        # the validator uploads only when the container exits 0: a published artifact is a success
        # whatever the trainer's own exit status was
        say(f"published {src} (trainer rc={rc})")
        sys.exit(0)
    say("no checkpoint produced")
    sys.exit(1)


if __name__ == "__main__":
    main()
