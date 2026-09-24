"""G0 check: load our exported LoRA the way the evaluator does and fail loudly on
any unmatched key or zero applied patches (image_flow_adapter.apply_lora).

    python tools/check_lora_keys.py --family krea2 --model-dir /cache/models/... --lora out/last.safetensors
Run inside the trainer image (needs ComfyUI on sys.path).
"""
import argparse, logging, sys
from crown import comfy_boot
from crown.assets import Assets

def main():
    p = argparse.ArgumentParser(); p.add_argument("--family", required=True); p.add_argument("--model-dir", required=True); p.add_argument("--lora", required=True)
    a = p.parse_args()
    comfy_boot.boot()
    import comfy.sd, comfy.utils, comfy.lora
    from crown.engine import load_diffusion_model
    assets = Assets(a.family, a.model_dir)
    model = load_diffusion_model(assets.diffusion_sd(), a.family)
    lora = comfy.utils.load_torch_file(a.lora, safe_load=True)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    missing = []
    class Cap(logging.Handler):
        def emit(self, r):
            m = r.getMessage()
            if m.startswith(("lora key not loaded:", "NOT LOADED ")): missing.append(m)
    h = Cap(); logging.getLogger().addHandler(h)
    patches = comfy.lora.load_lora(lora, key_map, log_missing=True)
    logging.getLogger().removeHandler(h)
    n = len(patches)
    print(f"lora tensors={len(lora)} mapped_patches={n} unmatched={len(missing)}")
    for m in missing[:10]: print("  ", m)
    sys.exit(0 if (n > 0 and not missing) else 1)

if __name__ == "__main__":
    main()
