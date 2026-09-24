#!/usr/bin/env bash
# Reproduce ops/docker/standalone-image-toolkit-trainer.dockerfile WITHOUT docker, for GPU
# providers whose instances are themselves containers (Vast.ai, RunPod pods). Same ComfyUI
# commit, same torch build, same pins, same baked VAEs — the numerics are the evaluator's.
#
#   bash tools/setup_venv.sh            # installs into $CROWN_ROOT (default ~/crown-env)
#   source ~/crown-env/env.sh           # activates: COMFY_ROOT, CROWN_ASSETS, PYTHONPATH, venv
set -euo pipefail
COMFY_COMMIT=694815f498295080a0e15a1502edc9dba841b110
ROOT=${CROWN_ROOT:-$HOME/crown-env}; COMFY_ROOT=$ROOT/ComfyUI; VENV=$ROOT/venv; ASSETS=$ROOT/assets
TRAINER=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$ROOT" "$ASSETS"

PY=$(command -v python3.11 || command -v python3.12 || command -v python3.10 || command -v python3)
echo "== python: $($PY --version)  (evaluator image is 3.11)"

if [ ! -f "$COMFY_ROOT/main.py" ]; then
  echo "== ComfyUI @ $COMFY_COMMIT"
  rm -rf "$COMFY_ROOT"; git init -q "$COMFY_ROOT"
  ( cd "$COMFY_ROOT" && git remote add origin https://github.com/comfyanonymous/ComfyUI.git \
    && git fetch -q --depth 1 origin "$COMFY_COMMIT" && git checkout -q FETCH_HEAD && rm -rf .git )
fi

if [ ! -x "$VENV/bin/python" ]; then $PY -m venv "$VENV"; fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip
echo "== torch 2.9.1 cu128"
pip install --timeout 120 --retries 10 torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
echo "== ComfyUI requirements + evaluator pins"
pip install --timeout 120 --retries 10 -r "$COMFY_ROOT/requirements.txt" \
    diffusers==0.39.0 transformers==5.10.2 huggingface-hub==1.18.0 safetensors==0.8.0 \
    pydantic==2.13.4 accelerate==1.6.0 pillow numpy boto3

echo "== evaluator VAEs -> $ASSETS"
[ -s "$ASSETS/qwen_image_vae.safetensors" ] || curl -fL --retry 5 -o "$ASSETS/qwen_image_vae.safetensors" https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors
[ -s "$ASSETS/flux2-vae.safetensors" ]     || curl -fL --retry 5 -o "$ASSETS/flux2-vae.safetensors"     https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/vae/flux2-vae.safetensors
# FLUX text encoders (the evaluator's files; the validator cache does not stage them)
FLUX_TE_REV=${FLUX_TE_REV:-6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5}
[ -s "$ASSETS/clip_l.safetensors" ]        || curl -fL --retry 5 -o "$ASSETS/clip_l.safetensors"        "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/$FLUX_TE_REV/clip_l.safetensors"
[ -s "$ASSETS/t5xxl_fp16.safetensors" ]    || curl -fL --retry 5 -o "$ASSETS/t5xxl_fp16.safetensors"    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/$FLUX_TE_REV/t5xxl_fp16.safetensors"
# Ideogram-4 base exactly as the evaluator loads it (per-tensor comfy_quant); used by the twin scorer only
[ -s "$ASSETS/ideogram4_fp8_scaled.safetensors" ] || curl -fL --retry 5 -o "$ASSETS/ideogram4_fp8_scaled.safetensors" https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_fp8_scaled.safetensors

python - <<'PY'
import torch, diffusers, transformers, comfy_kitchen
assert torch.__version__.startswith("2.9.1"), torch.__version__
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "| diffusers", diffusers.__version__, "| transformers", transformers.__version__)
PY

cat > "$ROOT/env.sh" <<EOF
export COMFY_ROOT="$COMFY_ROOT" CROWN_ASSETS="$ASSETS" PYTHONPATH="$TRAINER" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
source "$VENV/bin/activate"
EOF
echo "== done. activate with:  source $ROOT/env.sh"
