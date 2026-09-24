#!/usr/bin/env bash
# Run the validator entrypoint WITHOUT docker (after tools/setup_venv.sh), keeping the code
# byte-identical to production: the fixed paths /app, /cache, /app/checkpoints are provided
# as symlinks instead of bind mounts. Needs root (or writable /); Vast/RunPod containers are root.
#
#   CACHE_DIR=~/cache OUT_DIR=~/outputs tools/local_run_venv.sh <task_id> <model_id> <model_type> <hours> [trigger_word]
#
# Not reproduced here (docker-only): --network none, --memory 110g, --cpus 24. Do not rely on
# this runner for memory-limit or offline behaviour; use tools/local_run.sh on a real VM for that.
set -euo pipefail
TASK_ID=${1:?task_id}; MODEL=${2:?model_id}; MODEL_TYPE=${3:?model_type}; HOURS=${4:?hours}; TRIGGER=${5:-}
CACHE_DIR=${CACHE_DIR:-$HOME/cache}; OUT_DIR=${OUT_DIR:-$HOME/outputs}
TRAINER=$(cd "$(dirname "$0")/.." && pwd); ROOT=${CROWN_ROOT:-$HOME/crown-env}
# shellcheck disable=SC1091
source "$ROOT/env.sh"
mkdir -p "$OUT_DIR"
[ -e /app ] && [ ! -L /app ] && { echo "/app exists and is not a symlink; refusing"; exit 1; }
ln -sfn "$TRAINER" /app
ln -sfn "$CACHE_DIR" /cache
# /app is $TRAINER, so /app/checkpoints must be a symlink to OUT_DIR. (ln -sfn into an existing
# directory would create checkpoints/<basename> instead, silently sending artifacts elsewhere.)
if [ -d "$TRAINER/checkpoints" ] && [ ! -L "$TRAINER/checkpoints" ]; then
  rmdir "$TRAINER/checkpoints" 2>/dev/null || { echo "$TRAINER/checkpoints is a real, non-empty directory; move it aside"; exit 1; }
fi
ln -sfn "$OUT_DIR" "$TRAINER/checkpoints"
[ "$(readlink -f /app/checkpoints)" = "$(readlink -f "$OUT_DIR")" ] || { echo "/app/checkpoints does not resolve to $OUT_DIR"; exit 1; }
ARGS=(--task-id "$TASK_ID" --model "$MODEL" --dataset-zip "/cache/datasets/${TASK_ID}_tourn.zip" \
      --model-type "$MODEL_TYPE" --expected-repo-name "local-${TASK_ID:0:8}" --hours-to-complete "$HOURS")
[ -n "$TRIGGER" ] && ARGS+=(--trigger-word "$TRIGGER")
export GOD_TRAIN_LOGS=${GOD_TRAIN_LOGS:-1} TRANSFORMERS_CACHE=/cache/hf_cache
python /app/scripts/image_trainer.py "${ARGS[@]}"
echo "artifact: $OUT_DIR/$TASK_ID/local-${TASK_ID:0:8}/"
