"""Anatomy of LoRA artefacts (CPU only): where the change sits and how many directions it uses.

For every module, dW = (alpha / r) * up @ down is analysed through its r x r core (QR of up and down^T),
so no out x in matrix is ever formed:
  * |dW|_F and |dW|_F / |W|_F against the base weights (optional --base),
  * singular-value energy captured by the top k directions (k = 8, 16, 32, 64) and the entropy
    effective rank exp(H(s / sum s)).
Results are grouped by module kind (double img/txt attention, mlp, modulation; single linear1/linear2/
modulation; text encoders) and by depth, and the totals say where the energy of the change sits.

    python tools/lora_anatomy.py --base /workspace/cache/models/rayonlabs--FLUX.1-dev/flux1-dev.safetensors \
        --out /root/anatomy/9159e3dc.json 5DXVNvDm=/workspace/replay/9159e3dc/loras/5DXVNvDm/checkpoints/last.safetensors \
        ours=/workspace/outputs_replay/flux_9159_default/<task>/local-9159e3dc/last.safetensors
"""

import argparse
import collections
import json
import math
import re

import torch
from safetensors import safe_open

KS = (8, 16, 32, 64)


def module_name(key):
    """(stream, module path in BFL/Comfy naming) for a LoRA key prefix; stream is unet / te1 / te3."""
    if key.startswith("diffusion_model."):
        return "unet", key[len("diffusion_model."):]
    if key.startswith("lora_unet_"):
        rest = key[len("lora_unet_"):]
        m = re.match(r"^(double_blocks|single_blocks)_(\d+)_(.+)$", rest)
        if not m:
            return "unet", rest
        tail = (m.group(3).replace("modulation_lin", "modulation.lin").replace("mod_lin", "mod.lin")
                .replace("attn_", "attn.").replace("mlp_", "mlp."))
        return "unet", f"{m.group(1)}.{m.group(2)}.{tail}"
    for te in ("lora_te1_", "lora_te2_", "lora_te3_"):
        if key.startswith(te):
            return te[5:8], key[len(te):]
    return "other", key


def kind_of(stream, mod):
    if stream != "unet":
        return stream
    m = re.match(r"^(double_blocks|single_blocks)\.(\d+)\.(.+)$", mod)
    if not m:
        return "unet_other:" + mod
    return ("double." if m.group(1) == "double_blocks" else "single.") + m.group(3)


def depth_of(mod):
    m = re.match(r"^(double_blocks|single_blocks)\.(\d+)\.", mod)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def analyse(path, base_norms):
    f = safe_open(path, "pt", device="cpu")
    keys = list(f.keys())
    downs = [k for k in keys if k.endswith(".lora_down.weight")]
    rows = []
    for dk in downs:
        pre = dk[: -len(".lora_down.weight")]
        down = f.get_tensor(dk).float()
        up = f.get_tensor(pre + ".lora_up.weight").float()
        r = down.shape[0]
        alpha = float(f.get_tensor(pre + ".alpha").float().item()) if (pre + ".alpha") in keys else float(r)
        scale = alpha / r
        if up.dim() > 2:
            up = up.flatten(1)
        if down.dim() > 2:
            down = down.flatten(1)
        qu, ru = torch.linalg.qr(up)                # up = Qu Ru
        qd, rd = torch.linalg.qr(down.t())          # down^T = Qd Rd
        s = torch.linalg.svdvals(ru @ rd.t()) * abs(scale)
        energy = (s ** 2).sum().item()
        fro = math.sqrt(energy)
        cum = torch.cumsum(s ** 2, 0) / max(energy, 1e-30)
        p = s / max(s.sum().item(), 1e-30)
        erank = math.exp(-(p[p > 0] * p[p > 0].log()).sum().item())
        stream, mod = module_name(pre)
        w = base_norms.get(mod) if stream == "unet" else None
        rows.append({"key": pre, "stream": stream, "module": mod, "kind": kind_of(stream, mod), "depth": depth_of(mod),
                     "rank": r, "alpha": alpha, "fro": fro, "rel": (fro / w) if w else None, "erank": erank,
                     "top": {k: float(cum[min(k, len(cum)) - 1].item()) for k in KS}, "s1": float(s[0].item())})
    return rows


def base_norm_table(base, modules):
    out = {}
    if not base:
        return out
    f = safe_open(base, "pt", device="cpu")
    keys = set(f.keys())
    for mod in modules:
        for cand in (f"{mod}.weight", f"model.diffusion_model.{mod}.weight", f"diffusion_model.{mod}.weight"):
            if cand in keys:
                out[mod] = f.get_tensor(cand).float().norm().item()
                break
    return out


def summarise(name, rows):
    tot = sum(r["fro"] ** 2 for r in rows) or 1.0
    by = collections.defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r)
    print(f"=== {name}: {len(rows)} modules, ranks {dict(collections.Counter(r['rank'] for r in rows))}, "
          f"alpha {dict(collections.Counter(r['alpha'] for r in rows))}, total |dW|_F {math.sqrt(tot):.3f}")
    print(f"  {'kind':28} {'n':>3} {'energy%':>8} {'rel|dW|/|W|':>12} {'erank':>6} " + " ".join(f"top{k:<3}" for k in KS))
    for kind in sorted(by, key=lambda k: -sum(r["fro"] ** 2 for r in by[k])):
        rs = by[kind]
        e = sum(r["fro"] ** 2 for r in rs) / tot * 100
        rel = [r["rel"] for r in rs if r["rel"] is not None]
        relm = f"{sum(rel) / len(rel):.5f}" if rel else "   n/a"
        tops = " ".join(f"{sum(r['top'][k] for r in rs) / len(rs):.3f} " for k in KS)
        print(f"  {kind:28} {len(rs):>3} {e:>7.1f}% {relm:>12} {sum(r['erank'] for r in rs) / len(rs):>6.1f} {tops}")
    for blk in ("double_blocks", "single_blocks"):
        d = collections.defaultdict(float)
        for r in rows:
            if r["depth"][0] == blk:
                d[r["depth"][1]] += r["fro"] ** 2
        if d:
            print(f"  energy by depth ({blk}): " + " ".join(f"{i}:{d[i] / tot * 100:.1f}" for i in sorted(d)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=None, help="base diffusion weights, for |dW|/|W|")
    p.add_argument("--out", default=None)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("loras", nargs="+", help="name=path")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    results = {}
    parsed = {}
    mods = set()
    for s in a.loras:
        name, path = s.split("=", 1)
        parsed[name] = analyse(path, {})
        mods.update(r["module"] for r in parsed[name] if r["stream"] == "unet")
    norms = base_norm_table(a.base, sorted(mods))
    for name, rows in parsed.items():
        for r in rows:
            w = norms.get(r["module"]) if r["stream"] == "unet" else None
            r["rel"] = (r["fro"] / w) if w else None
        summarise(name, rows)
        results[name] = rows
    if a.out:
        json.dump(results, open(a.out, "w"))


if __name__ == "__main__":
    main()
