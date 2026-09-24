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
RUN pip install --timeout 120 --retries 10 --no-cache-dir \
        torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
RUN pip install --timeout 120 --retries 10 --no-cache-dir -r /opt/ComfyUI/requirements.txt \
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
# one layer per asset: a legacy (non-BuildKit) builder commits each RUN by copying its diff, so the
# transient disk peak is the largest single file, not the whole 19 GB set
RUN mkdir -p /opt/crown/assets
RUN curl -fL --retry 5 -o /opt/crown/assets/qwen_image_vae.safetensors \
         https://huggingface.co/Comfy-Org/Krea-2/resolve/main/vae/qwen_image_vae.safetensors
RUN curl -fL --retry 5 -o /opt/crown/assets/flux2-vae.safetensors \
         https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/vae/flux2-vae.safetensors
RUN curl -fL --retry 5 -o /opt/crown/assets/clip_l.safetensors \
         https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/${FLUX_TE_REV}/clip_l.safetensors
RUN curl -fL --retry 5 -o /opt/crown/assets/t5xxl_fp16.safetensors \
         https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/${FLUX_TE_REV}/t5xxl_fp16.safetensors
RUN curl -fL --retry 5 -o /opt/crown/assets/ideogram4_fp8_scaled.safetensors \
         https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_fp8_scaled.safetensors
RUN curl -fL --retry 5 -o /opt/crown/assets/ae.safetensors \
         https://huggingface.co/rayonlabs/FLUX.1-dev/resolve/main/ae.safetensors

COPY crown /app/crown
COPY scripts /app/scripts
COPY tools /app/tools
RUN python -c "import torch, diffusers, transformers; assert torch.__version__.startswith('2.9.1')" \
    && python -c "import comfy_kitchen" \
    && mkdir -p /app/checkpoints

ENTRYPOINT ["python", "/app/scripts/image_trainer.py"]
