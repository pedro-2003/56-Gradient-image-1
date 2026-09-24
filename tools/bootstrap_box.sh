#!/usr/bin/env bash
# One-shot bootstrap for a freshly created GPU box (destroy/recreate workflow).
#
#   bash tools/bootstrap_box.sh krea2 z-image        # families to stage this session
#
# Env:
#   IMAGE          docker image tag; pulled if it exists in a registry, else built here  (default crown-image-trainer)
#   REGISTRY       optional registry prefix, e.g. ghcr.io/<user>  -> pulls $REGISTRY/$IMAGE
#   CACHE_DIR      where /cache is staged                                               (default $HOME/cache)
#   R2_ACCOUNT_ID / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_BUCKET   if set, models come from the R2 mirror (tools/r2.sh)
#   HF_TOKEN       needed for gated repos (krea/Krea-2-Raw, FLUX) when pulling from HuggingFace
#   DATASETS_DIR   local research/datasets copy (task dirs with NNN.png/NNN.txt/task.json)
#
# Idempotent: re-running skips what is already present.
set -euo pipefail
FAMILIES=("$@"); [ ${#FAMILIES[@]} -gt 0 ] || { echo "usage: $0 <family> [family...]"; exit 2; }
IMAGE=${IMAGE:-crown-image-trainer}; REGISTRY=${REGISTRY:-}; CACHE_DIR=${CACHE_DIR:-$HOME/cache}
DATASETS_DIR=${DATASETS_DIR:-$(cd "$(dirname "$0")/../.." && pwd)/research/datasets}
HERE=$(cd "$(dirname "$0")/.." && pwd)

declare -A BASE=( [krea2]="krea/Krea-2-Raw" [ideogram4]="gradients-io-tournaments/ideogram-4-fp8"
                  [qwen-image]="gradients-io-tournaments/Qwen-Image" [z-image]="gradients-io-tournaments/Z-Image-Turbo"
                  [flux]="rayonlabs/FLUX.1-dev" )

echo "== 0. sanity"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "no GPU visible"; exit 1; }
df -h "$HOME" | tail -1; (docker --version 2>/dev/null || echo "docker: absent (venv mode)")

echo "== 1. runtime"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  MODE=docker
  if [ -n "$REGISTRY" ] && docker pull "$REGISTRY/$IMAGE" 2>/dev/null; then docker tag "$REGISTRY/$IMAGE" "$IMAGE"; echo "pulled $REGISTRY/$IMAGE"
  elif docker image inspect "$IMAGE" >/dev/null 2>&1; then echo "image present"
  else (cd "$HERE" && docker build -t "$IMAGE" -f ops/docker/standalone-image-toolkit-trainer.dockerfile .); fi
else
  MODE=venv   # container-based GPU providers (Vast.ai, RunPod): no docker-in-docker -> reproduce the image in a venv
  echo "docker unavailable -> venv mode"; bash "$HERE/tools/setup_venv.sh"
fi

echo "== 2. cache ($CACHE_DIR)"; mkdir -p "$CACHE_DIR"/{models,datasets,hf_cache}
if [ -n "${R2_ACCESS_KEY_ID:-}" ] && command -v rclone >/dev/null; then
  bash "$HERE/tools/r2.sh" setup >/dev/null && bash "$HERE/tools/r2.sh" pull "$CACHE_DIR" "${FAMILIES[@]}"
else
  pip show huggingface_hub >/dev/null 2>&1 || pip install -q huggingface_hub
  for fam in "${FAMILIES[@]}"; do
    d=$(ls -d "$DATASETS_DIR"/*_"$fam" 2>/dev/null | head -1); [ -n "$d" ] || { echo "no local dataset for $fam"; continue; }
    python "$HERE/tools/prepare_cache.py" --cache "$CACHE_DIR" --task-dir "$d" --model "${BASE[$fam]}" --model-type "$fam"
  done
fi

echo "== 3. dataset zips for every local task dir"
for d in "$DATASETS_DIR"/*/; do [ -f "$d/task.json" ] || continue
  python "$HERE/tools/prepare_cache.py" --cache "$CACHE_DIR" --task-dir "$d" --model x --model-type x --skip-model >/dev/null; done
ls "$CACHE_DIR/datasets" | head; echo "..."

echo "== 4. ready ($MODE)"; du -sh "$CACHE_DIR"/models/* 2>/dev/null
if [ "$MODE" = docker ]; then echo "next: CACHE_DIR=$CACHE_DIR tools/local_run.sh <task_id> <model_id> <family> <hours>"; else echo "next: CACHE_DIR=$CACHE_DIR tools/local_run_venv.sh <task_id> <model_id> <family> <hours>"; fi
