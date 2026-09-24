"""Run the evaluator's own load_base + apply_lora on one of our LoRAs, exactly as
tools/local_evaluate.py does, and report which keys the evaluator rejects — with the CLIP
attached (the real path) and again with clip=None, to isolate the text-encoder key map.

    python tools/probe_evalpath.py --family ideogram4 --base <file> --lora <last.safetensors> [--comfy-root ...]
"""
import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--comfy-root", default=os.environ.get("COMFY_ROOT", "/app/validator/evaluation/ComfyUI"))
    a = p.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "0"

    import evaluator_vendor.image_artifacts as artifacts
    import evaluator_vendor.evaluators.diffusion as diffusion
    from huggingface_hub import HfApi

    root = Path(a.comfy_root)
    sys.path.insert(0, str(root))          # the evaluator bootstraps ComfyUI exactly like this
    import comfy.model_management  # noqa: F401
    import comfy.samplers  # noqa: F401
    folder = "unet" if a.family == "flux" else "diffusion_models"
    base_name, lora_name = "probe_base.safetensors", "probe_lora.safetensors"
    for src, dst in ((a.base, root / "models" / folder / base_name), (a.lora, root / "models" / "loras" / lora_name)):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(Path(src).resolve())
        except OSError:
            shutil.copy2(src, dst)

    def local_prepare_base(api, task, root_):
        return base_name, {"repo": "local", "revision": "local", "filename": base_name}

    diffusion.prepare_base = local_prepare_base
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    api = HfApi()
    adapter, model, clip, encoder = diffusion.load_base(api, "local/base", a.family, root)
    print(f"model {type(model.model).__name__}; clip {type(clip.cond_stage_model).__name__}")

    import comfy.lora
    import comfy.utils
    km_model = comfy.lora.model_lora_keys_unet(model.model, {})
    km_both = comfy.lora.model_lora_keys_clip(clip.cond_stage_model, dict(km_model))
    lora = comfy.utils.load_torch_file(str(root / "models" / "loras" / lora_name), safe_load=True)
    mods = sorted({k.rsplit(".", 2)[0] if k.endswith((".lora_up.weight", ".lora_down.weight")) else k[:-len(".alpha")] for k in lora})
    print(f"key_map model-only {len(km_model)}, model+clip {len(km_both)}; lora modules {len(mods)}")
    print("modules missing from model-only map:", [m for m in mods if m not in km_model][:5])
    print("modules missing from model+clip map:", [m for m in mods if m not in km_both][:5])
    changed = [k for k in km_model if km_both.get(k) != km_model[k]]
    print(f"entries the clip map OVERWROTE: {len(changed)}", changed[:5])

    # the evaluator frees the VAE and unloads every model before scoring candidates
    import comfy.model_management
    comfy.model_management.unload_all_models()
    km_after = comfy.lora.model_lora_keys_unet(model.model, {})
    sdk_after = list(model.model.state_dict().keys())
    print(f"AFTER unload_all_models: state_dict keys {len(sdk_after)}, key_map {len(km_after)}; sample sdk {sdk_after[:3]}")
    print("modules missing from post-unload map:", len([m for m in mods if m not in km_after]), "matched:", [m for m in mods if m in km_after][:5])
    print("post-unload key_map sample:", [k for k in km_after][:8])
    for label, c in (("model+clip", clip), ("model only", None)):
        try:
            patched, _ = adapter.apply_lora(model, c, lora_name)
            print(f"[{label}] apply_lora OK: {len(patched.patches)} patches")
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] apply_lora FAILED: {e}")


if __name__ == "__main__":
    main()
