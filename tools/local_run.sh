#!/usr/bin/env bash
# Local harness that mirrors the validator's production constraints EXACTLY
# (trainer/constants.py + docs/miner.md). The stock ops/examples runner mounts
# /cache read-write and caps memory at 32g; both hide production failure modes.
#
#   usage: tools/local_run.sh <task_id> <model_id> <model_type> <hours> [trigger_word]
#   expects: ./cache populated by the trainer-downloader (models/, datasets/{task}_tourn.zip, hf_cache/)
set -euo pipefail
TASK_ID=${1:?task_id}; MODEL=${2:?model_id}; MODEL_TYPE=${3:?model_type}; HOURS=${4:?hours}; TRIGGER=${5:-}
CACHE_DIR=${CACHE_DIR:-$(pwd)/cache}; OUT_DIR=${OUT_DIR:-$(pwd)/outputs}; IMAGE=${IMAGE:-crown-image-trainer}
mkdir -p "$OUT_DIR"
docker build -t "$IMAGE" -f ops/docker/standalone-image-toolkit-trainer.dockerfile .
ARGS=(--task-id "$TASK_ID" --model "$MODEL" --dataset-zip "/cache/datasets/${TASK_ID}_tourn.zip" \
      --model-type "$MODEL_TYPE" --expected-repo-name "local-${TASK_ID:0:8}" --hours-to-complete "$HOURS")
[ -n "$TRIGGER" ] && ARGS+=(--trigger-word "$TRIGGER")
docker run --rm --gpus '"device=0"' \
  --security-opt=no-new-privileges --cap-drop=ALL \
  --memory=110g --cpus=24 --network none \
  --env TRANSFORMERS_CACHE=/cache/hf_cache --env GOD_TRAIN_LOGS=1 \
  --volume "$CACHE_DIR:/cache:ro" \
  --volume "$OUT_DIR:/app/checkpoints:rw" \
  "$IMAGE" "${ARGS[@]}"
echo "artifact: $OUT_DIR/$TASK_ID/local-${TASK_ID:0:8}/"
