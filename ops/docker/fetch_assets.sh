#!/usr/bin/env bash
# Build-time asset pulls under a hard time budget. The validator cold-builds this image with
# no cache under a 30-minute cap for EVERY task, so a slow mirror must never turn into a failed
# build: files are fetched smallest first, each under its own timeout, the whole step under
# ASSET_BUDGET_S, and a file that does not land is simply absent — the trainer degrades at run
# time (no flux text encoders -> the task repo's own; no flux VAE -> the repo's diffusers VAE).
# Exit status is always 0. The evaluator's ideogram4 base (8.7 GB) is no longer pulled: the twin
# rebuilds it from the task cache (crown.twin.rebuild_ideogram4_evaluator_base; evaluator scores
# within 0.058 % of the file on four LoRAs, same rank order, 2026-09-26).
set -u
DEST=${ASSET_DIR:-/opt/crown/assets}
BUDGET=${ASSET_BUDGET_S:-540}
REV=${FLUX_TE_REV:-6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5}
mkdir -p "$DEST"
start=$(date +%s)
fetch() {  # name url
  local name=$1 url=$2 left
  left=$(( BUDGET - ($(date +%s) - start) ))
  if [ "$left" -le 30 ]; then echo "asset budget exhausted; skipping $name"; return 0; fi
  # -sS: no progress meter. The meter is '\r'-separated with no '\n' for minutes, and the validator's build-log
  # reader (G.O.D core/logging.py stream_image_build_logs) buffers the stream until '\n' and rescans that
  # buffer on every chunk; a 2026-09-24 cold build on the VPS wrote 4.6 GB of meter output. Errors still print.
  if timeout "$left" curl -fsSL --retry 3 --retry-delay 2 -o "$DEST/$name.part" "$url"; then
    mv -f "$DEST/$name.part" "$DEST/$name"
    echo "fetched $name ($(du -h "$DEST/$name" | cut -f1)) at t=$(( $(date +%s) - start ))s"
  else
    rc=$?
    rm -f "$DEST/$name.part"
    echo "MISSING $name (exit $rc) at t=$(( $(date +%s) - start ))s; trainer will degrade for this asset"
  fi
  return 0
}
fetch clip_l.safetensors               "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/${REV}/clip_l.safetensors"
fetch qwen_image_vae.safetensors       "https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors"
fetch flux2-vae.safetensors            "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/vae/flux2-vae.safetensors"
fetch ae.safetensors                   "https://huggingface.co/rayonlabs/FLUX.1-dev/resolve/main/ae.safetensors"
fetch t5xxl_fp16.safetensors           "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/${REV}/t5xxl_fp16.safetensors"
echo "assets present:"; ls -la --block-size=M "$DEST" | tail -n +2 | awk '{print $5, $9}'
exit 0
