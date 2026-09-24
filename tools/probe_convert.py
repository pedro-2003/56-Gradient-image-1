"""Reproduce the evaluator's exact LoRA path (nodes.LoraLoader -> comfy.sd.load_lora_for_models ->
comfy.lora_convert.convert_lora -> comfy.lora.load_lora) on one of our exported LoRAs and report
which keys survive each step.

    python tools/probe_convert.py --base /opt/crown/assets/ideogram4_fp8_scaled.safetensors --lora <last.safetensors>
"""
import argparse
import inspect
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crown import comfy_boot  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--lora", required=True)
    a = p.parse_args()
    comfy_boot.boot()
    import comfy.lora
    import comfy.lora_convert
    import comfy.sd
    import comfy.utils

    print("--- comfy.sd.load_lora_for_models source ---")
    print(inspect.getsource(comfy.sd.load_lora_for_models))

    lora = comfy.utils.load_torch_file(a.lora, safe_load=True)
    keys0 = list(lora.keys())
    conv = comfy.lora_convert.convert_lora(dict(lora))
    keys1 = list(conv.keys())
    lost = [k for k in keys0 if k not in conv]
    added = [k for k in keys1 if k not in lora]
    print(f"convert_lora: {len(keys0)} -> {len(keys1)} keys; lost {len(lost)}, added {len(added)}")
    for k in lost[:8]:
        print("  lost:", k)
    for k in added[:8]:
        print("  added:", k)

    model = comfy.sd.load_diffusion_model(a.base, model_options={})
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    missing = []

    class Cap(logging.Handler):
        def emit(self, r):
            m = r.getMessage()
            if m.startswith(("lora key not loaded:", "NOT LOADED ")):
                missing.append(m)

    h = Cap()
    logging.getLogger().addHandler(h)
    loaded = comfy.lora.load_lora(conv, key_map, log_missing=True)
    logging.getLogger().removeHandler(h)
    print(f"load_lora(converted): {len(loaded)} patches, {len(missing)} not-loaded warnings")
    for m in missing[:6]:
        print("  ", m)

    # the full evaluator call, model only (clip=None) — same as LoraLoader with no clip
    missing.clear()
    logging.getLogger().addHandler(h)
    new_model, _ = comfy.sd.load_lora_for_models(model, None, dict(lora), 1.0, 0.0)
    logging.getLogger().removeHandler(h)
    print(f"load_lora_for_models(model, None): patches {len(new_model.patches)}, {len(missing)} not-loaded warnings")
    for m in missing[:6]:
        print("  ", m)


if __name__ == "__main__":
    main()
