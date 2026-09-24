#!/usr/bin/env bash
# Run a CHAMPION checkout's validator entrypoint in our venv, with the production paths it expects
# (/app = its tree, /cache, /app/checkpoints, /opt/godtrainer/assets), so it can be paired against
# our trainer on the same box, data and horizon. The 0921 champion's image is the same stack as
# ours (ComfyUI @694815f4, torch 2.9.1), so no second environment is needed.
#
#   CHAMP_ROOT=/workspace/champ0921 CACHE_DIR=... OUT_DIR=... tools/champ_run_venv.sh <task_id> <model_id> <model_type> <hours> [trigger_word]
set -euo pipefail
TASK_ID=${1:?task_id}; MODEL=${2:?model_id}; MODEL_TYPE=${3:?model_type}; HOURS=${4:?hours}; TRIGGER=${5:-}
CHAMP_ROOT=${CHAMP_ROOT:-/workspace/champ0921}
CACHE_DIR=${CACHE_DIR:-$HOME/cache}; OUT_DIR=${OUT_DIR:-$HOME/outputs_champ}
ROOT=${CROWN_ROOT:-$HOME/crown-env}
# shellcheck disable=SC1091
source "$ROOT/env.sh"
mkdir -p "$OUT_DIR"
[ -e /app ] && [ ! -L /app ] && { echo "/app exists and is not a symlink; refusing"; exit 1; }
ln -sfn "$CHAMP_ROOT" /app
ln -sfn "$CACHE_DIR" /cache
if [ -d "$CHAMP_ROOT/checkpoints" ] && [ ! -L "$CHAMP_ROOT/checkpoints" ]; then
  rmdir "$CHAMP_ROOT/checkpoints" 2>/dev/null || { echo "$CHAMP_ROOT/checkpoints is a real, non-empty directory; move it aside"; exit 1; }
fi
ln -sfn "$OUT_DIR" "$CHAMP_ROOT/checkpoints"
[ "$(readlink -f /app/checkpoints)" = "$(readlink -f "$OUT_DIR")" ] || { echo "/app/checkpoints does not resolve to $OUT_DIR"; exit 1; }
# the champion bakes qwen_image_vae + flux2-vae under /opt/godtrainer/assets; ours has the same files
mkdir -p /opt/godtrainer
[ -e /opt/godtrainer/assets ] && [ ! -L /opt/godtrainer/assets ] && { echo "/opt/godtrainer/assets exists and is not a symlink; refusing"; exit 1; }
ln -sfn "$CROWN_ASSETS" /opt/godtrainer/assets
rm -rf /tmp/godtrainer-out
ARGS=(--task-id "$TASK_ID" --model "$MODEL" --dataset-zip "/cache/datasets/${TASK_ID}_tourn.zip" \
      --model-type "$MODEL_TYPE" --expected-repo-name "champ-${TASK_ID:0:8}" --hours-to-complete "$HOURS")
[ -n "$TRIGGER" ] && ARGS+=(--trigger-word "$TRIGGER")
export GOD_TRAIN_LOGS=1 PYTHONPATH=/app/trainer TRANSFORMERS_CACHE=/cache/hf_cache
python /app/scripts/image_trainer.py "${ARGS[@]}"
echo "artifact: $OUT_DIR/$TASK_ID/champ-${TASK_ID:0:8}/"
