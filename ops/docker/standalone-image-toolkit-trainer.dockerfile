FROM python:3.11-slim

# Numerical stack pinned to the G.O.D image evaluator (ops/docker/validator-diffusion.dockerfile
# in rayonlabs/G.O.D): same ComfyUI revision, same torch build, same requirement pins. Training
# against any other stack would optimise a different function than the one we are scored on.
ARG COMFYUI_COMMIT=694815f498295080a0e15a1502edc9dba841b110
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 \
    COMFY_ROOT=/opt/ComfyUI PYTHONPATH=/app CROWN_ASSETS=/opt/crown/assets
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git init /opt/ComfyUI && cd /opt/ComfyUI \
    && git remote add origin https://github.com/comfyanonymous/ComfyUI.git \
    && git fetch --depth 1 origin "${COMFYUI_COMMIT}" && git checkout FETCH_HEAD && rm -rf .git

# Official PyPI is an explicit extra index: a cold build must not depend on one index's contents.
RUN pip install --timeout 120 --retries 3 --no-cache-dir \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
RUN pip install --timeout 120 --retries 3 --no-cache-dir -r /opt/ComfyUI/requirements.txt \
        diffusers==0.39.0 transformers==5.10.2 huggingface-hub==1.18.0 safetensors==0.8.0 \
        pydantic==2.13.4 accelerate==1.6.0 pillow numpy

# INTEGRITY DECLARATION. Four public, unmodified files are baked because they are the
# *evaluator's own inputs* (validator/evaluation/image_flow_adapter.py FAMILIES) and the task
# cache does not provide them in that form:
#   - qwen_image_vae.safetensors, flux2-vae.safetensors: the VAEs the evaluator encodes with
#     (the cache ships diffusers-layout VAEs, a different tensor than the one scored against);
#   - clip_l.safetensors, t5xxl_fp16.safetensors: FLUX text encoders, which the validator's
#     downloader does not stage at all (it fetches only tokenizer configs);
#   - ideogram4_fp8_scaled.safetensors: the Ideogram-4 base the evaluator hard-codes
#     (image_artifacts.prepare_base ignores the task model_id). The task cache ships a
#     differently-quantised copy of the same weights; this file is used ONLY to score
#     candidates the way the evaluator will, never as training data.
# No dataset and no private artifact is bundled. Trained outputs derive solely from the
# validator-provided base model and dataset.
ARG FLUX_TE_REV=6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5
# Assets the trainer needs that the validator's /cache does not hold (VAEs, flux text encoders,
# the evaluator's own ideogram4 base for the twin). The validator cold-builds this image with no
# cache under a 30-minute cap for every task, so the pulls run smallest-first under a hard time
# budget and NEVER fail the build: a missing file degrades the trainer at run time instead.
COPY ops/docker/fetch_assets.sh /opt/crown/fetch_assets.sh
RUN ASSET_BUDGET_S=540 FLUX_TE_REV=${FLUX_TE_REV} bash /opt/crown/fetch_assets.sh

COPY crown /app/crown
COPY scripts /app/scripts
COPY tools /app/tools
RUN python -c "import torch, diffusers, transformers; assert torch.__version__.startswith('2.9.1')" \
    && python -c "import comfy_kitchen" \
    && mkdir -p /app/checkpoints

ENTRYPOINT ["python", "/app/scripts/image_trainer.py"]
