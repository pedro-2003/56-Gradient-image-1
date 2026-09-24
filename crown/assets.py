"""Resolve base weights, text encoders and VAE for a family from the validator
cache, in the form the evaluator loads them.

Parity status per family (research/docs/research_evaluator_and_field.md §2):
  krea2, z-image  base = same file as the evaluator (bf16)                    -> full parity
  qwen-image      base = same plain-fp8 file; Comfy manual_cast on both sides  -> parity (merge RTN vs SR)
  flux            base = same root checkpoint                                 -> parity
  ideogram4       cache is per-row fp8, evaluator is per-tensor comfy_quant   -> NOT parity (see L2)
Text encoders come from the cache in every family and differ from the
evaluator's fp8_scaled files; that gap is measured, not assumed (L7).
"""

import glob
import json
import os

import torch
from safetensors.torch import load_file

from . import contract as C


# --- file helpers --------------------------------------------------------------

def _shards(dir_path):
    for idx in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        p = os.path.join(dir_path, idx)
        if os.path.exists(p):
            wm = json.load(open(p))["weight_map"]
            return [os.path.join(dir_path, f) for f in sorted(set(wm.values()))]
    return sorted(glob.glob(os.path.join(dir_path, "*.safetensors")))


def load_sharded(files):
    sd = {}
    for f in files:
        sd.update(load_file(f, device="cpu"))
    return sd


def _root_safetensors(model_dir, min_bytes=1 << 30):
    files = [f for f in glob.glob(os.path.join(model_dir, "*.safetensors")) if os.path.getsize(f) > min_bytes]
    return sorted(files, key=os.path.getsize, reverse=True)


def resolve_model_dir(model_id: str) -> str:
    cand = str(C.cached_model_dir(model_id))
    if os.path.isdir(cand):
        return cand
    dirs = [d for d in glob.glob(str(C.CACHE_MODELS / "*")) if os.path.isdir(d) and not os.path.basename(d).startswith(".")]
    if len(dirs) == 1:
        return dirs[0]
    raise FileNotFoundError(f"model dir for {model_id!r} not found under {C.CACHE_MODELS}")


def hf_snapshot_dir(repo: str):
    flat = str(C.CACHE_HF / repo.replace("/", "--"))
    if os.path.isdir(flat) and _shards(flat):
        return flat
    snaps = glob.glob(str(C.CACHE_HF / ("models--" + repo.replace("/", "--")) / "snapshots" / "*"))
    if snaps:
        return sorted(snaps)[-1]
    return None


def sub_dict(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


def to_bf16(sd):
    return {k: (v.to(torch.bfloat16) if v.is_floating_point() and v.dtype != torch.bfloat16 else v) for k, v in sd.items()}


def dequantize_scaled_fp8(sd):
    """Dequantise a scaled-fp8 state dict (per-row `weight_scale [rows]` or
    per-tensor `weight_scale []`) to bf16. Scale keys are dropped."""
    out = {}
    for k, v in sd.items():
        if k.endswith("weight_scale") or k.endswith("scale_weight") or k.endswith("comfy_quant"):
            continue
        if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            s = sd.get(k + "_scale")
            if s is None and k.endswith(".weight"):
                s = sd.get(k[:-len(".weight")] + ".scale_weight")
            w = v.float()
            if s is not None:
                w = w * s.float().reshape(-1, *([1] * (v.dim() - 1)))
            out[k] = w.to(torch.bfloat16)
        else:
            out[k] = v
    return out


# --- per-family assets ---------------------------------------------------------

class Assets:
    def __init__(self, family: str, model_dir: str, baked_dir=None):
        if family not in C.FAMILIES:
            raise ValueError(f"unknown family {family!r}")
        self.family = family
        self.model_dir = model_dir
        self.baked_dir = baked_dir or os.environ.get("CROWN_ASSETS", str(C.BAKED_DIR))
        self._ckpt = None

    def _full_checkpoint(self):
        if self._ckpt is None:
            roots = _root_safetensors(self.model_dir)
            if not roots:
                raise FileNotFoundError(f"no root checkpoint under {self.model_dir}")
            self._ckpt = load_sharded(roots[:1])
        return self._ckpt

    def diffusion_sd(self):
        f = self.family
        if f == "ideogram4":
            tdir = os.path.join(self.model_dir, "transformer")
            shards = _shards(tdir) if os.path.isdir(tdir) else _root_safetensors(self.model_dir)[:1]
            return dequantize_scaled_fp8(load_sharded(shards))
        if f == "flux":
            # rayonlabs/FLUX.1-dev: flux1-dev.safetensors is a bare diffusion model (double_blocks.*,
            # single_blocks.*, ...), exactly what the evaluator's UNETLoader reads. If a full
            # checkpoint with a model.diffusion_model. prefix ever appears instead, strip it.
            sd = self._full_checkpoint()
            inner = sub_dict(sd, "model.diffusion_model.")
            return to_bf16(inner if inner else sd)
        roots = _root_safetensors(self.model_dir)
        if f == "krea2":
            # the evaluator takes the single root file > 5 GB; the repo's is raw.safetensors
            pref = [r for r in roots if "raw" in os.path.basename(r).lower()] or roots
            return to_bf16(load_sharded(pref[:1]))
        sd = load_sharded(roots[:1])
        if any(v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for v in sd.values()):
            return sd  # qwen-image: plain fp8, keep it (Comfy manual_cast == evaluator path)
        return to_bf16(sd)

    def text_encoder_sds(self):
        f = self.family
        if f in ("krea2", "ideogram4"):
            repo = "Qwen/Qwen3-VL-4B-Instruct" if f == "krea2" else "Qwen/Qwen3-VL-8B-Instruct"
            d = hf_snapshot_dir(repo)
            if d is None:
                d = os.path.join(self.model_dir, "text_encoder")
            sd = load_sharded(_shards(d))
            if any(v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for v in sd.values()):
                sd = dequantize_scaled_fp8(sd)
            return [sd]
        if f in ("qwen-image", "z-image"):
            return [load_sharded(_shards(os.path.join(self.model_dir, "text_encoder")))]
        # flux: baked clip_l + t5xxl_fp16 (the evaluator's files); fall back to a full checkpoint's
        # embedded encoders only if someone ships one
        baked = [os.path.join(self.baked_dir, n) for n in C.BAKED_FLUX_TE]
        if all(os.path.exists(b) for b in baked):
            return [load_file(b, device="cpu") for b in baked]
        ck = self._full_checkpoint()
        sds = [sub_dict(ck, "text_encoders.clip_l.transformer."), sub_dict(ck, "text_encoders.t5xxl.transformer.")]
        if not all(sds):
            raise FileNotFoundError(f"flux text encoders not baked under {self.baked_dir} and not embedded in the checkpoint")
        return sds

    def vae(self):
        """('comfy', state_dict) or ('diffusers', path) — whichever the evaluator uses."""
        f = self.family
        if f == "z-image":
            return "diffusers", os.path.join(self.model_dir, "vae")
        if f == "flux":
            ae = os.path.join(self.model_dir, C.FLUX_VAE_FILE)   # the evaluator's rayonlabs/FLUX.1-dev/ae.safetensors
            if os.path.exists(ae):
                return "comfy", load_file(ae, device="cpu")
            embedded = sub_dict(self._full_checkpoint(), "vae.")
            if embedded:
                return "comfy", embedded
            baked = os.path.join(self.baked_dir, C.FLUX_VAE_FILE)   # task repos (e.g. PixelWave) ship no ae.safetensors
            if os.path.exists(baked):
                return "comfy", load_file(baked, device="cpu")
            raise FileNotFoundError(f"flux VAE {C.FLUX_VAE_FILE} not found under {self.model_dir} or {self.baked_dir}")
        baked = os.path.join(self.baked_dir, C.BAKED_VAE[f])
        if os.path.exists(baked):
            return "comfy", load_file(baked, device="cpu")
        # fallback keeps the run alive; parity with the evaluator's VAE is then not guaranteed
        return "diffusers", os.path.join(self.model_dir, "vae")

    def release(self):
        self._ckpt = None
