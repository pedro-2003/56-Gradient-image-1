"""Why does the evaluator reject the last ~30 keys of our LoRA on krea2/ideogram4/qwen while
tools/check_lora_keys.py accepts all of them?  Load the base exactly as the evaluator does
(Comfy's diffusion-model loader on the file the evaluator would select), build the same key_map
(comfy.lora.model_lora_keys_unet) and test every key of the LoRA file against it, in file order.

    python tools/probe_keymap.py --family ideogram4 --base /opt/crown/assets/ideogram4_fp8_scaled.safetensors --lora <last.safetensors>
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crown import comfy_boot  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--lora", required=True)
    a = p.parse_args()
    comfy_boot.boot()
    import comfy.lora
    import comfy.sd
    import comfy.utils
    from safetensors import safe_open

    model = comfy.sd.load_diffusion_model(a.base, model_options={})
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    sdk = list(model.model.state_dict().keys())
    print(f"model {type(model.model).__name__}: state_dict keys {len(sdk)}, key_map entries {len(key_map)}")
    with safe_open(a.lora, framework="pt") as f:
        keys = list(f.keys())
    print(f"lora file keys {len(keys)} (file order = safetensors header order)")
    bases = []
    for k in keys:
        for suf in (".lora_up.weight", ".lora_down.weight", ".alpha"):
            if k.endswith(suf):
                bases.append(k[: -len(suf)])
                break
    seen, order = set(), []
    for b in bases:
        if b not in seen:
            seen.add(b); order.append(b)
    missing = [b for b in order if b not in key_map]
    print(f"distinct lora modules {len(order)}; NOT in key_map: {len(missing)}")
    for b in missing[:40]:
        w = b + ".weight"
        print(f"  {b}   model has '{w}': {w in sdk}   position in file order: {order.index(b)}/{len(order)}")
    # is it positional? compare with the tail of the file order
    tail = set(order[-len(missing):]) if missing else set()
    print(f"missing == last {len(missing)} modules in file order: {set(missing) == tail}")
    # and the evaluator's own loader
    lora = comfy.utils.load_torch_file(a.lora, safe_load=True)
    loaded = comfy.lora.load_lora(lora, key_map, log_missing=False)
    print(f"comfy.lora.load_lora: {len(loaded)} patches from {len(lora)} tensors")
    if missing:
        for b in missing[:3]:
            cands = [k for k in sdk if k.startswith("diffusion_model." + b.split(".")[0])][:3]
            print(f"  state_dict keys starting like {b.split('.')[0]}: {cands}")


if __name__ == "__main__":
    main()
