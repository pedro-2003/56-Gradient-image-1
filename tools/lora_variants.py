"""Controlled variants of a LoRA artefact for diagnosis (CPU only). Keys keep their original names,
so every variant loads through the evaluator exactly like its source.

    python tools/lora_variants.py --src winner.safetensors --out w_r16.safetensors --truncate 16
    python tools/lora_variants.py --src winner.safetensors --out w_note.safetensors --drop "^lora_te"
    python tools/lora_variants.py --src winner.safetensors --out w_mod.safetensors --keep "mod_lin|modulation_lin"

--truncate k   per module, the best rank-k approximation of dW = (alpha/r) up @ down (SVD through the
               r x r core: up = Qu Ru, down^T = Qd Rd, Ru Rd^T = U S V^T), written as rank k with alpha = k;
               the retained energy share is reported per module kind.
--keep / --drop  regexes over the module prefix (the key without .lora_up/.lora_down/.alpha).
"""

import argparse
import collections
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SUFFIXES = (".lora_up.weight", ".lora_down.weight", ".alpha")


def prefix_of(key):
    for s in SUFFIXES:
        if key.endswith(s):
            return key[: -len(s)]
    return None


def truncate(up, down, alpha, k):
    r = down.shape[0]
    scale = alpha / r
    qu, ru = torch.linalg.qr(up.double())
    qd, rd = torch.linalg.qr(down.double().t())
    u, s, vh = torch.linalg.svd(ru @ rd.t())
    k = min(k, s.numel())
    root = torch.sqrt(s[:k] * scale)
    new_up = (qu @ u[:, :k]) * root
    new_down = root[:, None] * (vh[:k] @ qd.t())
    kept = float((s[:k] ** 2).sum() / (s ** 2).sum()) if s.numel() else 1.0
    return new_up, new_down, float(k), kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--truncate", type=int, default=0)
    p.add_argument("--keep", default=None, help="regex: keep only modules whose prefix matches")
    p.add_argument("--drop", default=None, help="regex: drop modules whose prefix matches")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    keep = re.compile(a.keep) if a.keep else None
    drop = re.compile(a.drop) if a.drop else None
    f = safe_open(a.src, "pt", device="cpu")
    meta = f.metadata() or {}
    keys = list(f.keys())
    prefixes = sorted({prefix_of(k) for k in keys if prefix_of(k)})
    out, kept_by_kind, n_kept, n_dropped = {}, collections.defaultdict(list), 0, 0
    for pre in prefixes:
        if (keep and not keep.search(pre)) or (drop and drop.search(pre)):
            n_dropped += 1
            continue
        n_kept += 1
        up = f.get_tensor(pre + ".lora_up.weight")
        down = f.get_tensor(pre + ".lora_down.weight")
        has_alpha = (pre + ".alpha") in keys
        alpha = float(f.get_tensor(pre + ".alpha").float().item()) if has_alpha else float(down.shape[0])
        if a.truncate and down.shape[0] > a.truncate and up.dim() == 2 and down.dim() == 2:
            nu, nd, na, share = truncate(up, down, alpha, a.truncate)
            kind = re.sub(r"\d+", "N", pre)
            kept_by_kind[kind].append(share)
            out[pre + ".lora_up.weight"] = nu.to(up.dtype).contiguous()
            out[pre + ".lora_down.weight"] = nd.to(down.dtype).contiguous()
            out[pre + ".alpha"] = torch.tensor(na, dtype=torch.float32 if not has_alpha else f.get_tensor(pre + ".alpha").dtype)
        else:
            out[pre + ".lora_up.weight"] = up
            out[pre + ".lora_down.weight"] = down
            if has_alpha:
                out[pre + ".alpha"] = f.get_tensor(pre + ".alpha")
    if not out:
        raise SystemExit("no module left")
    meta = {k: str(v) for k, v in meta.items()}
    meta["variant_of"] = a.src
    meta["variant"] = f"truncate={a.truncate} keep={a.keep} drop={a.drop}"
    save_file(out, a.out, metadata=meta)
    print(f"VARIANT {a.out}: kept {n_kept} modules, dropped {n_dropped}; truncate={a.truncate}")
    for kind, v in sorted(kept_by_kind.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"   energy kept {sum(v) / len(v):.4f} (min {min(v):.4f})  {len(v):>3}  {kind}")


if __name__ == "__main__":
    main()
