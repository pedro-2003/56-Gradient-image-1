"""CPU check of the asset resolution against every flux repo layout the validator can present
(and two hypothetical ones), using tiny fake safetensors and a patched file-size rule:

  snapshot_bare     root bare transformer + ae.safetensors + diffusers dirs   (rayonlabs)
  snapshot_multi    three root files (bf16 / fp8 / nf4) + dirs               (PixelWave; evaluator rule ambiguous -> logged)
  allinone_fp8      single root file, model.diffusion_model.* in fp8 + vae + text encoders (MonochromeManga)
  bare_single       single root bare bf16 transformer, nothing else          (fluxunchained after normalisation)
  diffusers_only    no root file, transformer/ folder only                   (hypothetical: no loadable base -> clean error)
  scaled_fp8        single root file with a scaled_fp8 marker                 (hypothetical: must not be silently cast)

Asserts: the diffusion dict is the bare transformer keys; fp8 dtype is preserved and flagged (base_fp8);
the VAE always comes from the baked rayonlabs ae; text encoders from the baked pair; ambiguous or missing
roots produce the documented log line / a FileNotFoundError (never a wrong file).

    python tools/check_layouts.py     (exit 1 on any failure)
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from crown import assets as A  # noqa: E402

RESULTS = []
BIG = set()   # paths that pretend to be > 5 GiB

_real_getsize = os.path.getsize


def fake_getsize(p):
    return 6 * 1024 ** 3 if os.path.abspath(p) in BIG else _real_getsize(p)


A.os.path.getsize = fake_getsize   # assets.py calls os.path.getsize through its own os import


def bare(dtype=torch.bfloat16, prefix=""):
    return {f"{prefix}double_blocks.0.img_attn.qkv.weight": torch.zeros(4, 4, dtype=dtype),
            f"{prefix}single_blocks.0.linear1.weight": torch.zeros(4, 4, dtype=dtype),
            f"{prefix}img_in.weight": torch.zeros(4, 4, dtype=dtype)}


def write(path, sd, big=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(sd, path)
    if big:
        BIG.add(os.path.abspath(path))


def baked_dir():
    d = tempfile.mkdtemp(prefix="baked-")
    write(os.path.join(d, "ae.safetensors"), {"decoder.conv_in.weight": torch.zeros(2, 2)}, big=False)
    write(os.path.join(d, "clip_l.safetensors"), {"text_model.x.weight": torch.zeros(2, 2)}, big=False)
    write(os.path.join(d, "t5xxl_fp16.safetensors"), {"encoder.x.weight": torch.zeros(2, 2)}, big=False)
    return d


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True)); print(f"PASS {name}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False)); print(f"FAIL {name}: {type(e).__name__}: {e}")


def resolve(model_dir):
    a = A.Assets("flux", model_dir, baked_dir())
    sd = a.diffusion_sd()
    kind, vae = a.vae()
    tes = a.text_encoder_sds()
    return a, sd, kind, vae, tes


def expect_bare(sd, dtype):
    assert set(sd) == set(bare()), f"unexpected keys {sorted(sd)[:3]}"
    assert all(v.dtype == dtype for v in sd.values()), {k: v.dtype for k, v in sd.items()}


def snapshot_bare():
    d = tempfile.mkdtemp(prefix="rayon-")
    write(os.path.join(d, "flux1-dev.safetensors"), bare())
    write(os.path.join(d, "ae.safetensors"), {"decoder.conv_in.weight": torch.ones(2, 2)}, big=False)
    os.makedirs(os.path.join(d, "transformer")); os.makedirs(os.path.join(d, "vae"))
    a, sd, kind, vae, tes = resolve(d)
    expect_bare(sd, torch.bfloat16)
    assert not a.base_fp8
    assert kind == "comfy" and float(vae["decoder.conv_in.weight"].sum()) == 0.0, "VAE must be the baked rayonlabs ae, not the repo copy"
    assert len(tes) == 2


def snapshot_multi():
    d = tempfile.mkdtemp(prefix="pixel-")
    write(os.path.join(d, "pixelwave_bf16.safetensors"), bare())
    write(os.path.join(d, "pixelwave_fp8.safetensors"), bare(torch.float8_e4m3fn))
    write(os.path.join(d, "pixelwave_nf4.safetensors"), {"x": torch.zeros(2, 2)})
    os.makedirs(os.path.join(d, "transformer"))
    # the bf16 file must win by size: make it the largest
    _sizes = {os.path.abspath(os.path.join(d, "pixelwave_bf16.safetensors")): 24 * 1024 ** 3,
              os.path.abspath(os.path.join(d, "pixelwave_fp8.safetensors")): 12 * 1024 ** 3,
              os.path.abspath(os.path.join(d, "pixelwave_nf4.safetensors")): 6 * 1024 ** 3}
    A.os.path.getsize = lambda p: _sizes.get(os.path.abspath(p), fake_getsize(p))
    try:
        a, sd, kind, vae, tes = resolve(d)
    finally:
        A.os.path.getsize = fake_getsize
    expect_bare(sd, torch.bfloat16)
    assert not a.base_fp8


def allinone_fp8():
    d = tempfile.mkdtemp(prefix="mono-")
    sd_all = bare(torch.float8_e4m3fn, prefix="model.diffusion_model.")
    sd_all.update({"vae.decoder.conv_in.weight": torch.ones(2, 2), "text_encoders.clip_l.transformer.x.weight": torch.ones(2, 2),
                   "text_encoders.t5xxl.transformer.x.weight": torch.ones(2, 2)})
    write(os.path.join(d, "FLUX-DEV_MonochromeManga.safetensors"), sd_all)
    a, sd, kind, vae, tes = resolve(d)
    expect_bare(sd, torch.float8_e4m3fn)
    assert a.base_fp8, "fp8 base not flagged (the twin would not be built)"
    assert float(vae["decoder.conv_in.weight"].sum()) == 0.0, "VAE must be the baked rayonlabs ae, not the embedded one"
    assert len(tes) == 2 and float(tes[0]["text_model.x.weight"].sum()) == 0.0, "text encoders must be the baked pair"


def bare_single():
    d = tempfile.mkdtemp(prefix="unch-")
    write(os.path.join(d, "fluxunchained-dev-fp16.safetensors"), bare())
    a, sd, kind, vae, tes = resolve(d)
    expect_bare(sd, torch.bfloat16)
    assert kind == "comfy" and len(tes) == 2


def diffusers_only():
    d = tempfile.mkdtemp(prefix="diff-")
    os.makedirs(os.path.join(d, "transformer"))
    write(os.path.join(d, "transformer", "diffusion_pytorch_model.safetensors"), bare())
    try:
        resolve(d)
        raise AssertionError("a repo without a root checkpoint must fail loudly (identity fallback), not load a wrong file")
    except FileNotFoundError:
        pass


def scaled_fp8():
    d = tempfile.mkdtemp(prefix="scaled-")
    sd_s = bare(torch.float8_e4m3fn)
    sd_s["scaled_fp8"] = torch.zeros(0, dtype=torch.float8_e4m3fn)
    sd_s["double_blocks.0.img_attn.qkv.scale_weight"] = torch.ones(1)
    write(os.path.join(d, "flux-scaled.safetensors"), sd_s)
    a, sd, kind, vae, tes = resolve(d)
    assert a.base_fp8 and "scaled_fp8" in sd, "scaled-fp8 marker must survive to Comfy's loader (never cast away)"


if __name__ == "__main__":
    for name, fn in [("snapshot_bare", snapshot_bare), ("snapshot_multi", snapshot_multi), ("allinone_fp8", allinone_fp8),
                     ("bare_single", bare_single), ("diffusers_only", diffusers_only), ("scaled_fp8", scaled_fp8)]:
        check(name, fn)
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} layout checks passed")
    sys.exit(1 if bad else 0)
