"""The validator contract, as read from G.O.D (2026-09-23).

Every constant here is dictated by the validator or the evaluator, not chosen
by us. Sources are cited inline. Change nothing without re-reading the source.
"""

from pathlib import Path

# --- container paths (trainer/constants.py, docs/miner.md) ---------------------
CACHE_ROOT = Path("/cache")                      # mounted read-only
CACHE_MODELS = CACHE_ROOT / "models"             # {model_id with '/' -> '--'}
CACHE_DATASETS = CACHE_ROOT / "datasets"         # {task_id}_tourn.zip
CACHE_HF = CACHE_ROOT / "hf_cache"               # text encoders for krea2/ideogram4
CHECKPOINTS_ROOT = Path("/app/checkpoints")      # the only writable output path
WORK_ROOT = Path("/tmp/crown")                   # scratch inside the container

# --- families (validator/evaluation/image_flow_adapter.py FAMILIES) -------------
FAMILIES = ("flux", "z-image", "qwen-image", "ideogram4", "krea2")

# Evaluator text-encoder / VAE identities (image_flow_adapter.py). We bake the
# two VAEs the cache does not provide in evaluator format; text encoders come
# from the cache (see assets.py for the documented parity gap).
BAKED_DIR = Path("/opt/crown/assets")
BAKED_VAE = {
    "krea2": "qwen_image_vae.safetensors",       # Comfy-Org/Krea-2/vae/
    "qwen-image": "qwen_image_vae.safetensors",  # Comfy-Org/Qwen-Image_ComfyUI/split_files/vae/ (same bytes)
    "ideogram4": "flux2-vae.safetensors",        # Comfy-Org/Ideogram-4/vae/
}
# FLUX: the validator's downloader stages only the repo snapshot (flux1-dev.safetensors +
# ae.safetensors) and tokenizer configs — no text-encoder weights. The evaluator encodes
# prompts with comfyanonymous/flux_text_encoders {t5xxl_fp16, clip_l}; we bake those two.
BAKED_FLUX_TE = ("clip_l.safetensors", "t5xxl_fp16.safetensors")
FLUX_VAE_FILE = "ae.safetensors"                 # root of the model snapshot == evaluator's rayonlabs/FLUX.1-dev/ae.safetensors
# ideogram4: the evaluator ignores the task model_id and loads this file (image_artifacts.prepare_base).
# The cache ships a differently-quantised copy (per-row scales); this one is per-tensor comfy_quant.
# Baked so the twin scorer can load the model exactly as the evaluator does.
BAKED_IDEOGRAM4_BASE = "ideogram4_fp8_scaled.safetensors"   # Comfy-Org/Ideogram-4/diffusion_models/
CLIP_TYPES = {"krea2": "KREA2", "ideogram4": "IDEOGRAM4", "qwen-image": "QWEN_IMAGE", "z-image": "LUMINA2", "flux": "FLUX"}
FAMILY_GUIDANCE = {"flux": 3.5}                  # FlowFamily.guidance
FAMILY_SHIFT = {"qwen-image": 3.0}               # FlowFamily.shift (ModelSamplingAuraFlow)

# --- scoring (validator/evaluation/denoising_mse.py, evaluators/diffusion.py) ----
EVAL_STRATA = 16
EVAL_NOISES = 16                                 # DEFAULT_NOISES; no production override exists
EVAL_MASTER_SEED = 42
EVAL_TEXT_WEIGHT = 0.5                           # DIFFUSION_TEXT_GUIDED_EVAL_WEIGHT
EVAL_BATCH = 2                                   # IMAGE_EVAL_BATCH_SIZE default
EVAL_CFG = 1.0
EVAL_SAMPLER_SEED = 42

# --- dataset (validator/tasks/datasets/preparation.py, core/constants/datasets.py)
TEST_SPLIT_FRACTION = 0.10                       # ceil(N * 0.10) held out, disjoint, unseeded shuffle
MIN_IMAGE_TEXT_PAIRS = 10
MAX_IMAGE_TEXT_PAIRS = 50
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# --- output (docs/miner.md, image_artifacts.select_lora) ----------------------
OUTPUT_LORA_NAME = "last.safetensors"            # exactly one -> selected unambiguously
OUTPUT_SUBDIR = "checkpoints"                    # HF_REPO_SUBFOLDER the uploader uses

# --- timing --------------------------------------------------------------------
# The validator waits `hours_to_complete` then kills the container. Everything
# after the kill is lost, so the run must be *finished and written* before it.
KILL_MARGIN_S = 180.0                            # never train into the last 3 minutes
PUBLISH_RESERVE_S = 90.0                         # time to save + copy the final artifact


def cached_model_dir(model_id: str) -> Path:
    return CACHE_MODELS / model_id.replace("/", "--")


def dataset_zip(task_id: str) -> Path:
    return CACHE_DATASETS / f"{task_id}_tourn.zip"


def output_dir(task_id: str, expected_repo_name: str) -> Path:
    return CHECKPOINTS_ROOT / task_id / expected_repo_name
