#!/usr/bin/env bash
# Build /cache entries for one or more downloaded task directories, reading each task's
# base_model_repository and model_type from its task.json (as written by tools/fetch_datasets.py).
#
#   tools/cache_tasks.sh <cache_dir> <task_dir> [<task_dir> ...]
#
# Needs HF_TOKEN in the environment for gated repos; prints <family>_EXIT=<rc> per task.
set -uo pipefail
CACHE=${1:?cache_dir}; shift
PY=${PYTHON:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
for d in "$@"; do
  [ -f "$d/task.json" ] || { echo "no task.json in $d"; continue; }
  repo=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_model_repository"])' "$d/task.json")
  fam=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_type"])' "$d/task.json")
  echo "== $fam <- $repo ($d) $(date -u +%H:%M:%S)"
  "$PY" "$HERE/prepare_cache.py" --cache "$CACHE" --task-dir "$d" --model "$repo" --model-type "$fam"
  echo "${fam}_EXIT=$?"
done
echo CACHE_TASKS_DONE
