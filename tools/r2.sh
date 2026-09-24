#!/usr/bin/env bash
# Cloudflare R2 cache mirror helper (rclone, S3-compatible).
#
#   tools/r2.sh setup            # writes the rclone remote "r2" from env vars, then lists buckets
#   tools/r2.sh push <cache_dir> # upload a staged /cache tree  -> r2:$R2_BUCKET/cache/
#   tools/r2.sh pull <cache_dir> [family...]   # download all, or only the given families' models
#   tools/r2.sh ls               # what is in the mirror
#   tools/r2.sh push-tree <dir> <prefix>   # any directory (e.g. research/datasets) -> r2:$R2_BUCKET/<prefix>/
#   tools/r2.sh pull-tree <prefix> <dir>
#
# Env (from the R2 API-token screen):
#   R2_ACCOUNT_ID   R2_ACCESS_KEY_ID   R2_SECRET_ACCESS_KEY   R2_BUCKET (default sn56-cache)
set -euo pipefail
CMD=${1:-}; shift || true
R2_BUCKET=${R2_BUCKET:-sn56-cache}
REMOTE="r2:$R2_BUCKET/cache"

need() { command -v "$1" >/dev/null || { echo "missing: $1"; [ "$1" = rclone ] && echo "install: curl https://rclone.org/install.sh | sudo bash"; exit 1; }; }

case "$CMD" in
  setup)
    need rclone
    : "${R2_ACCOUNT_ID:?}" "${R2_ACCESS_KEY_ID:?}" "${R2_SECRET_ACCESS_KEY:?}"
    mkdir -p "$HOME/.config/rclone"
    # remove an old [r2] block if present, then append a fresh one
    if [ -f "$HOME/.config/rclone/rclone.conf" ]; then
      awk 'BEGIN{skip=0} /^\[r2\]$/{skip=1;next} /^\[/{skip=0} !skip' "$HOME/.config/rclone/rclone.conf" > "$HOME/.config/rclone/rclone.conf.tmp"
      mv "$HOME/.config/rclone/rclone.conf.tmp" "$HOME/.config/rclone/rclone.conf"
    fi
    cat >> "$HOME/.config/rclone/rclone.conf" <<EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = $R2_ACCESS_KEY_ID
secret_access_key = $R2_SECRET_ACCESS_KEY
endpoint = https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com
acl = private
no_check_bucket = true
EOF
    chmod 600 "$HOME/.config/rclone/rclone.conf"
    # the API token is scoped to one bucket (no ListBuckets right), so verify by listing the bucket itself
    echo "remote r2 configured; bucket $R2_BUCKET:"; rclone lsd "r2:$R2_BUCKET" || true; rclone size "r2:$R2_BUCKET" ;;

  push)
    need rclone; SRC=${1:?cache_dir}
    # everything: models/, hf_cache/ (the expensive parts) and datasets/, so a fresh box needs
    # nothing from any other machine
    rclone copy --progress --transfers 16 --checkers 32 --s3-chunk-size 64M --s3-upload-concurrency 8 "$SRC" "$REMOTE"
    echo "pushed $SRC -> $REMOTE"; rclone size "$REMOTE" ;;

  pull)
    need rclone; DST=${1:?cache_dir}; shift || true
    mkdir -p "$DST"/{models,hf_cache,datasets}
    if [ $# -eq 0 ]; then
      rclone copy --progress --transfers 16 --checkers 32 "$REMOTE" "$DST"
    else
      declare -A BASE=( [krea2]="krea--Krea-2-Raw" [ideogram4]="gradients-io-tournaments--ideogram-4-fp8"
                        [qwen-image]="gradients-io-tournaments--Qwen-Image" [z-image]="gradients-io-tournaments--Z-Image-Turbo"
                        [flux]="rayonlabs--FLUX.1-dev" )
      for fam in "$@"; do
        rclone copy --progress --transfers 16 "$REMOTE/models/${BASE[$fam]}" "$DST/models/${BASE[$fam]}"
        case $fam in
          krea2)     rclone copy --transfers 16 "$REMOTE/hf_cache/Qwen--Qwen3-VL-4B-Instruct" "$DST/hf_cache/Qwen--Qwen3-VL-4B-Instruct";;
          ideogram4) rclone copy --transfers 16 "$REMOTE/hf_cache/Qwen--Qwen3-VL-8B-Instruct" "$DST/hf_cache/Qwen--Qwen3-VL-8B-Instruct";;
        esac
      done
    fi
    du -sh "$DST"/models/* 2>/dev/null ;;

  push-tree)
    need rclone; SRC=${1:?dir}; PFX=${2:?prefix}
    rclone copy --progress --transfers 16 --checkers 32 "$SRC" "r2:$R2_BUCKET/$PFX"; rclone size "r2:$R2_BUCKET/$PFX" ;;

  pull-tree)
    need rclone; PFX=${1:?prefix}; DST=${2:?dir}; mkdir -p "$DST"
    rclone copy --progress --transfers 16 --checkers 32 "r2:$R2_BUCKET/$PFX" "$DST"; du -sh "$DST" ;;

  ls)
    need rclone; rclone lsd "r2:$R2_BUCKET" 2>/dev/null || echo "(empty)"; rclone lsd "$REMOTE/models" 2>/dev/null; rclone size "r2:$R2_BUCKET" ;;

  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
