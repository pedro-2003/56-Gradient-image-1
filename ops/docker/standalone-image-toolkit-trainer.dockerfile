FROM python:3.11-slim

# Numerical stack pinned to the G.O.D image evaluator (ops/docker/validator-diffusion.dockerfile
# in rayonlabs/G.O.D): same ComfyUI revision, same torch build, same requirement pins. Training
# against any other stack would optimise a different function than the one we are scored on.
#
# STEP ORDER. The validator cold-builds this image with the legacy builder (docker-py /build, no cache,
# no build args) under a 30-minute cap for every task. On docker 29's containerd image store every step
# after a large layer pays a fixed commit cost (measured 2026-09-26 on the build VPS: ~35 s per step,
# even for ARG/COPY, once the torch layer exists), so every cheap step (ARG, ENV, WORKDIR, ENTRYPOINT,
# COPY) comes BEFORE the large layers, the import check shares the requirements RUN, and the 19 GB asset
# pull is the last step. The image content is unchanged.
ARG COMFYUI_COMMIT=694815f498295080a0e15a1502edc9dba841b110
ARG FLUX_TE_REV=6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5
ARG ASSET_BUDGET_S=540
ENV PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_OFFLINE=1 \
    COMFY_ROOT=/opt/ComfyUI PYTHONPATH=/app CROWN_ASSETS=/opt/crown/assets
WORKDIR /app
ENTRYPOINT ["python", "/app/scripts/image_trainer.py"]

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /app/checkpoints

RUN git init /opt/ComfyUI && cd /opt/ComfyUI \
    && git remote add origin https://github.com/comfyanonymous/ComfyUI.git \
    && git fetch --depth 1 origin "${COMFYUI_COMMIT}" && git checkout FETCH_HEAD && rm -rf .git

COPY ops/docker/fetch_assets.sh /opt/crown/fetch_assets.sh
COPY crown /app/crown
COPY scripts /app/scripts
COPY tools /app/tools

# Official PyPI is an explicit extra index: a cold build must not depend on one index's contents.
RUN pip install --timeout 120 --retries 3 --no-cache-dir \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
RUN pip install --timeout 120 --retries 3 --no-cache-dir -r /opt/ComfyUI/requirements.txt \
        diffusers==0.39.0 transformers==5.10.2 huggingface-hub==1.18.0 safetensors==0.8.0 \
        pydantic==2.13.4 accelerate==1.6.0 pillow numpy \
    && python -c "import torch, diffusers, transformers; assert torch.__version__.startswith('2.9.1')" \
    && python -c "import comfy_kitchen"

# INTEGRITY DECLARATION. Five public, unmodified files are baked because they are the
# *evaluator's own inputs* (validator/evaluation/image_flow_adapter.py FAMILIES) and the task
# cache does not provide them in that form:
#   - qwen_image_vae.safetensors, flux2-vae.safetensors: the VAEs the evaluator encodes with
#     (the cache ships diffusers-layout VAEs, a different tensor than the one scored against);
#   - ae.safetensors: the FLUX VAE the evaluator encodes with (rayonlabs/FLUX.1-dev);
#   - clip_l.safetensors, t5xxl_fp16.safetensors: FLUX text encoders, which the validator's
#     downloader does not stage at all (it fetches only tokenizer configs).
# The Ideogram-4 base the evaluator hard-codes is NOT baked: the evaluator twin rebuilds it from the
# task cache's copy of the same weights (crown/twin.py) and uses it only to score candidates.
# No dataset and no private artifact is bundled. Trained outputs derive solely from the
# validator-provided base model and dataset.
# The pulls run smallest-first under a hard time budget and NEVER fail the build: a missing file
# degrades the trainer at run time instead.
RUN ASSET_BUDGET_S=${ASSET_BUDGET_S} FLUX_TE_REV=${FLUX_TE_REV} bash /opt/crown/fetch_assets.sh
