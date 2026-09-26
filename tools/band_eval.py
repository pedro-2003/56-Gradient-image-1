"""F2 instrument: the validator's own evaluator (vendored verbatim, tools/evaluator_vendor) run on
several LoRAs in ONE process, keeping every (image, mode, band, noise) row that the evaluator computes
and then averages away. Each image's reconstructed means are asserted EQUAL to the evaluator's own
vectors, so the rows are exactly the numbers the validator's score is made of.

    python tools/band_eval.py --family flux --base /cache/models/rayonlabs--FLUX.1-dev/flux1-dev.safetensors \
        --dataset /workspace/replay/9159e3dc/cache/holdout --out /root/band/9159e3dc.json --with-base \
        ours=/workspace/outputs_replay/.../last.safetensors 5DXVNvDm=/workspace/replay/9159e3dc/loras/5DXVNvDm/checkpoints/last.safetensors

The score is a plain mean over (image, mode, band, noise) rows, all equally weighted, so a difference
between two LoRAs splits exactly into per-band parts: delta = sum_b delta_b / strata. Rows are paired
across LoRAs (same image, mode, band and seed), which gives each band difference a paired SE.
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np

MODES = ("text", "no_text")
KEYS = {"text": "text_guided_losses", "no_text": "no_text_losses"}


def _link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(Path(src).resolve())
    except OSError:
        shutil.copy2(src, dst)


def run(a, entries):
    sys.path.insert(0, "/app")
    try:
        import validator.evaluation.evaluators.diffusion as diffusion
        import validator.evaluation.image_artifacts as artifacts
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import evaluator_vendor.evaluators.diffusion as diffusion
        import evaluator_vendor.image_artifacts as artifacts

    root = Path(a.comfy_root)
    folder = "unet" if a.family == "flux" else "diffusion_models"
    # same sweep as tools/local_evaluate.py: Comfy resolves diffusion_models names through models/unet
    # first, so stale local_* links from another family must never survive
    for d in ("unet", "diffusion_models", "loras"):
        (root / "models" / d).mkdir(parents=True, exist_ok=True)
        for stale in (root / "models" / d).glob("local_*.safetensors"):
            stale.unlink()
    base_name = f"local_base_{a.family}.safetensors"
    _link(a.base, root / "models" / folder / base_name)
    lora_names = {}
    for i, (name, path) in enumerate(entries):
        lora_names[f"local/{name}"] = f"local_lora_{a.family}_{i}.safetensors"
        _link(path, root / "models" / "loras" / lora_names[f"local/{name}"])

    def local_prepare_base(api, task, root_):
        return base_name, {"repo": "local", "revision": "local", "filename": base_name}

    def local_materialize(api, repo, filename, directory):
        if filename is None:
            return lora_names[repo], {"repo": repo, "revision": "local", "filename": lora_names[repo]}
        return artifacts.materialize_model(api, repo, filename, directory)

    recorded, names = [], []
    parent = diffusion.FlowPredictionSampler

    class RecordingSampler(parent):
        def sample(self, *args, **kwargs):
            out = super().sample(*args, **kwargs)
            recorded.append([dict(r) for r in self.rows])
            return out

    decode = diffusion.decoded_images

    def recording_decode(rows):
        out = decode(rows)
        names.extend(item["name"] for item in out)
        return out

    diffusion.prepare_base = local_prepare_base
    diffusion.materialize_model = local_materialize
    diffusion.FlowPredictionSampler = RecordingSampler
    diffusion.decoded_images = recording_decode

    config = {"dataset": a.dataset, "repo": a.repo_dir or "local/base", "family": a.family,
              "models": [f"local/{n}" for n, _ in entries], "comfy_root": root, "strata": a.strata,
              "noises": a.noises, "batch_size": a.batch}
    result = diffusion.evaluate(config, Path(a.out).with_suffix(".raw.json"))
    return result, recorded, names


def split(result, recorded, names, entries, strata, noises):
    """Map the recorded samplers back to (model, image, mode): the evaluator runs, per model in order,
    every image x (text, no_text) and then one repeatability sampler. Fail closed on any mismatch."""
    k, models = 0, {}
    for name, _ in entries:
        r = result.get(f"local/{name}")
        if not isinstance(r, dict):
            raise SystemExit(f"{name}: evaluation failed: {r}")
        per = {}
        for ii, img in enumerate(names):
            per[img] = {}
            for mode in MODES:
                rows = recorded[k]
                k += 1
                if len(rows) != strata * noises:
                    raise SystemExit(f"{name}/{img}/{mode}: {len(rows)} rows, expected {strata * noises}")
                mean = float(np.mean([x["mse"] for x in rows]))
                want = r["eval_loss"][KEYS[mode]][ii]
                if mean != want:
                    raise SystemExit(f"{name}/{img}/{mode}: reconstructed mean {mean!r} != evaluator {want!r} (mapping broken)")
                per[img][mode] = [[x["stratum"], x["seed"], x["mse"]] for x in rows]
        k += 1  # the repeatability sampler of this model
        t, n = r["eval_loss"]["text_guided_losses"], r["eval_loss"]["no_text_losses"]
        img_scores = [0.5 * x + 0.5 * y for x, y in zip(t, n)]
        models[name] = {"score": sum(img_scores) / len(img_scores), "per_image": img_scores, "text": t, "no_text": n, "rows": per}
    if k != len(recorded):
        raise SystemExit(f"{len(recorded)} samplers recorded, {k} mapped (mapping broken)")
    return models


def table(models, names, strata, ref):
    """Rows keyed (image, mode, band, seed) -> mse, per model."""
    flat = {}
    for m, r in models.items():
        d = {}
        for img in names:
            for mode in MODES:
                for band, seed, mse in r["rows"][img][mode]:
                    d[(img, mode, band, seed)] = mse
        flat[m] = d
    keys = sorted(next(iter(flat.values())))
    bands = {}
    for m, d in flat.items():
        bands[m] = {mode: [float(np.mean([d[q] for q in keys if q[1] == mode and q[2] == b])) for b in range(strata)] for mode in MODES}
        bands[m]["both"] = [0.5 * (x + y) for x, y in zip(bands[m]["text"], bands[m]["no_text"])]
    paired = {}
    for m in flat:
        if m == ref:
            continue
        out = []
        for b in range(strata):
            diff = np.array([flat[m][q] - flat[ref][q] for q in keys if q[2] == b])
            se = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
            out.append({"band": b, "delta": float(diff.mean()), "se": se, "n": int(len(diff))})
        paired[m] = out
    return bands, paired


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--dataset", required=True, help="directory or zip of images + captions")
    p.add_argument("--out", required=True)
    p.add_argument("--comfy-root", default=os.environ.get("COMFY_ROOT", "/app/validator/evaluation/ComfyUI"))
    p.add_argument("--strata", type=int, default=16)
    p.add_argument("--noises", type=int, default=16)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--repo-dir", default=None)
    p.add_argument("--with-base", action="store_true", help="also score a zeroed copy of the first LoRA (= the base model)")
    p.add_argument("--ref", default=None, help="reference for the paired band deltas (default: base if scored, else the first LoRA)")
    p.add_argument("loras", nargs="+", help="name=path")
    a = p.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    entries = [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in a.loras]
    for n, path in entries:
        if not os.path.exists(path):
            raise SystemExit(f"{n}: missing {path}")
    if a.with_base:
        from safetensors.torch import load_file, save_file

        sd = load_file(entries[0][1])
        zpath = os.path.join(os.path.dirname(os.path.abspath(a.out)), "identity_" + os.path.basename(a.out).replace(".json", ".safetensors"))
        save_file({k: (v * 0 if k.endswith("lora_up.weight") else v) for k, v in sd.items()}, zpath)
        entries.insert(0, ("base", zpath))
    result, recorded, names = run(a, entries)
    models = split(result, recorded, names, entries, a.strata, a.noises)
    ref = a.ref or ("base" if a.with_base else entries[0][0])
    bands, paired = table(models, names, a.strata, ref)
    json.dump({"family": a.family, "dataset": a.dataset, "strata": a.strata, "noises": a.noises, "images": names, "ref": ref,
               "scores": {m: r["score"] for m, r in models.items()}, "per_image": {m: r["per_image"] for m, r in models.items()},
               "bands": bands, "paired_vs_ref": paired, "rows": {m: r["rows"] for m, r in models.items()}},
              open(a.out, "w"))
    print(f"--- band_eval {a.family}: {len(names)} images {names}, {a.strata} bands x {a.noises} noises x 2 modes; ref = {ref}")
    for m, r in models.items():
        rel = f" ({(r['score'] / models[ref]['score'] - 1) * 100:+.3f}% vs {ref})" if m != ref else ""
        print(f"{m:>10}: {r['score']:.6f}{rel}  per-image {[round(x, 6) for x in r['per_image']]}")
    hdr = " band  sigma " + "".join(f"{m[:10]:>12}" for m in models)
    for mode in ("both", "text", "no_text"):
        print(f"--- per-band mean loss ({mode})")
        print(hdr)
        for b in range(a.strata):
            print(f"  {b:>3}  {(b + 0.5) / a.strata:.3f} " + "".join(f"{bands[m][mode][b]:>12.5f}" for m in models))
    for m, rows in paired.items():
        total = sum(x["delta"] for x in rows) / a.strata
        print(f"--- {m} minus {ref}: total {total:+.6f} ({total / models[ref]['score'] * 100:+.3f}%); per band delta (paired SE) and share of the total")
        for x in rows:
            share = (x["delta"] / a.strata) / total * 100 if total else float("nan")
            print(f"  {x['band']:>3}  {x['delta']:+.5f} ({x['se']:.5f})  {share:+6.1f}%")


if __name__ == "__main__":
    main()
