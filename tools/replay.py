"""Tournament replay: measure our trainer against the REAL entrants of a past tournament task.

Every image task's score is deterministic (seeded noise per image hash) and public, and every
entrant's LoRA is public on Hugging Face (gradients-io-tournaments/tournament-<tourn>-<task>-<hk8>).
The validator's train/test split is an unseeded shuffle, so it cannot be recomputed — but it can be
IDENTIFIED: an entrant's per-image evaluator losses on all N images, averaged over the hidden subset,
must equal that entrant's public test_loss. With the hidden subset known, the exact training set is
all other images, our trainer runs on it at the task's hours, and our evaluator score on the hidden
images ranks directly among the public test_losses of every entrant.

    python tools/replay.py fetch    --tournament tourn_... --task <uuid> --dataset-dir <all images> --out /workspace/replay
    python tools/replay.py identify --task-dir /workspace/replay/<task8> --family F --base B [--repo-dir R] [--max-entrants 2]
    python tools/replay.py build    --task-dir /workspace/replay/<task8> --cache /workspace/cache
    python tools/replay.py rank     --task-dir /workspace/replay/<task8> --pair-json /root/pair/<tag>/pair.json --name <tag>
    python tools/replay.py rehearse --dataset-dir <images> --name r1flux_x --family flux --base-repo rayonlabs/FLUX.1-dev                                     --hours 0.75 --n 30 --seed 20260926 --out /workspace/replay

`rehearse` builds a replay-shaped task (no entrants) from any image set, for regimes no public task covers
(flux at the round-1 size): a seeded subset of --n images and the validator's hidden count drawn by the
same seeded rng; build / score / rank then run unchanged (rank reports ours against the base).

Runs on the GPU box (downloads go box <- HF directly; nothing passes through the PC).
"""
import argparse
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sys
import urllib.request
import uuid
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
API = "https://api.gradients.io"
HF_ORG = "gradients-io-tournaments"
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def get(u, timeout=60):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "curl/8"}), timeout=timeout))


def image_names(dataset_dir):
    """The evaluator's order: sorted image file names (image_test_data.py sorts paths)."""
    return sorted(f for f in os.listdir(dataset_dir) if f.lower().endswith(IMG_EXT))


def hidden_count(n):
    return math.ceil(n * 0.10)        # validator: split_idx = ceil(len(keys) * TRAIN_TEST_SPLIT_PERCENTAGE)


# ------------------------------------------------------------------------------------------ fetch
def cmd_fetch(a):
    d = get(f"{API}/tournament/{a.tournament}/details")
    entrants = None
    for r in d.get("rounds") or []:
        for t in r.get("tasks") or []:
            if t["task_id"] == a.task:
                entrants = [{"hotkey": p["hotkey"], "test_loss": p.get("test_loss"), "quality_score": p.get("quality_score"),
                             "round": r.get("round_number")} for p in t.get("participant_scores") or []]
    if entrants is None:
        raise SystemExit(f"task {a.task} not in {a.tournament}")
    task = get(f"{API}/v1/tasks/{a.task}")
    out = os.path.join(a.out, a.task[:8])
    os.makedirs(os.path.join(out, "loras"), exist_ok=True)
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    for e in entrants:
        repo = f"{HF_ORG}/tournament-{a.tournament}-{a.task}-{e['hotkey'][:8]}"
        e["repo"] = repo
        try:
            files = [s.rfilename for s in api.model_info(repo).siblings]
            cks = sorted(f for f in files if f.startswith(("checkpoints/", "checkpoint/")) and f.endswith(".safetensors"))
            pick = [f for f in cks if os.path.basename(f) == "last.safetensors"] or cks[-1:]   # select_lora: last.safetensors first
            if not pick:
                e["lora"] = None
                e["note"] = "no checkpoint in repo"
                continue
            p = hf_hub_download(repo, pick[0], local_dir=os.path.join(out, "loras", e["hotkey"][:8]))
            e["lora"] = p
        except Exception as ex:  # noqa: BLE001
            e["lora"] = None
            e["note"] = f"{type(ex).__name__}: {str(ex)[:120]}"
    names = image_names(a.dataset_dir)
    meta = {"tournament": a.tournament, "task": a.task, "family": task.get("model_type"), "base_repo": task.get("base_model_repository"),
            "hours": task.get("hours_to_complete"), "dataset_dir": os.path.abspath(a.dataset_dir), "n_images": len(names),
            "n_hidden": hidden_count(len(names)), "entrants": entrants}
    json.dump(meta, open(os.path.join(out, "replay.json"), "w"), indent=1)
    ok = sum(1 for e in entrants if e.get("lora"))
    print(f"{a.task[:8]} {meta['family']} N={len(names)} hidden={meta['n_hidden']} hours={meta['hours']} entrants={len(entrants)} loras={ok}")
    for e in sorted(entrants, key=lambda e: (e["test_loss"] is None, e["test_loss"] or 0)):
        print(f"   {e['hotkey'][:8]} test_loss {e['test_loss']} lora {'yes' if e.get('lora') else 'NO ' + e.get('note', '')}")


# --------------------------------------------------------------------------------------- identify
def best_subsets(losses, h, target, keep=5):
    """The `keep` h-subsets whose mean is closest to target, as (abs residual, indices)."""
    out = []
    worst = float("inf")
    for idx in itertools.combinations(range(len(losses)), h):
        r = abs(sum(losses[i] for i in idx) / h - target)
        if len(out) < keep or r < worst:
            out.append((r, idx))
            out.sort()
            out = out[:keep]
            worst = out[-1][0]
    return out


def cmd_identify(a):
    meta = json.load(open(os.path.join(a.task_dir, "replay.json")))
    from parity_check import run_eval

    names = image_names(meta["dataset_dir"])
    h = meta["n_hidden"]
    cands = [e for e in meta["entrants"] if e.get("lora") and e.get("test_loss") is not None]
    cands.sort(key=lambda e: e["test_loss"])
    # entrants with well-separated test losses constrain the subset independently
    chosen, seen = [], []
    for e in cands:
        if all(abs(e["test_loss"] - s) > 1e-4 for s in seen):
            chosen.append(e)
            seen.append(e["test_loss"])
        if len(chosen) >= a.max_entrants:
            break
    per = {}
    for e in chosen:
        outp = os.path.join(a.task_dir, f"perimage_{e['hotkey'][:8]}_n16.json")
        if os.path.exists(outp):
            r = json.load(open(outp))
        else:
            r = run_eval(meta["family"], a.base, e["lora"], meta["dataset_dir"], outp + ".raw", 16, a.repo_dir)
            json.dump(r, open(outp, "w"))
        if len(r["per_image"]) != len(names):
            raise SystemExit(f"per-image count {len(r['per_image'])} != {len(names)} images")
        per[e["hotkey"][:8]] = (e["test_loss"], r["per_image"])
        top = best_subsets(r["per_image"], h, e["test_loss"])
        print(f"{e['hotkey'][:8]} test_loss {e['test_loss']:.10f}:")
        for res, idx in top[:3]:
            print(f"   residual {res:.3e} ({res / e['test_loss']:.2e} rel)  {[names[i] for i in idx]}")
    # joint: the subset minimising the worst relative residual over all scored entrants
    joint = []
    for idx in itertools.combinations(range(len(names)), h):
        worst = max(abs(sum(L[i] for i in idx) / h - t) / t for t, L in per.values())
        joint.append((worst, idx))
    joint.sort()
    best, second = joint[0], joint[1] if len(joint) > 1 else (float("inf"), ())
    ident = {"hidden": [names[i] for i in best[1]], "worst_rel_residual": best[0], "runner_up_rel_residual": second[0],
             "margin": (second[0] / best[0]) if best[0] > 0 else float("inf"), "entrants_used": list(per)}
    # Measured replica precision vs the validator is ~1e-4 relative (9159e3dc, 87f69ea0), so an exact match
    # cannot be demanded. Identified = the most precise entrant's own best subset is (a) within --tol,
    # (b) at least --margin times better than its runner-up, and (c) the joint best over all entrants.
    per_ent = {}
    for hk, (t, L) in per.items():
        ranked = sorted((abs(sum(L[i] for i in idx) / h - t) / t, idx) for idx in itertools.combinations(range(len(names)), h))
        per_ent[hk] = {"best": [names[i] for i in ranked[0][1]], "rel": ranked[0][0],
                       "margin": ranked[1][0] / ranked[0][0] if len(ranked) > 1 and ranked[0][0] > 0 else float("inf")}
    anchor = min(per_ent.values(), key=lambda v: v["rel"])
    ident["per_entrant"] = per_ent
    ident["identified"] = anchor["rel"] < a.tol and anchor["margin"] > a.margin and anchor["best"] == ident["hidden"]
    json.dump(ident, open(os.path.join(a.task_dir, "hidden.json"), "w"), indent=1)
    print(f"IDENTIFY {meta['task'][:8]}: hidden {ident['hidden']} joint worst rel residual {best[0]:.2e} (runner-up x{ident['margin']:.0f}); "
          f"anchor entrant rel {anchor['rel']:.2e} margin x{anchor['margin']:.0f} -> {'IDENTIFIED' if ident['identified'] else 'AMBIGUOUS'}")
    for hk, v in per_ent.items():
        print(f"   {hk}: own best {v['best']} rel {v['rel']:.2e} (runner-up x{v['margin']:.0f})")


# ------------------------------------------------------------------------------------------ build
def cmd_build(a):
    meta = json.load(open(os.path.join(a.task_dir, "replay.json")))
    ident = json.load(open(os.path.join(a.task_dir, "hidden.json")))
    if not ident.get("identified") and not a.force:
        raise SystemExit("hidden subset not identified; refusing to build (use --force to override)")
    src = meta["dataset_dir"]
    hidden = set(ident["hidden"])
    cache = os.path.join(a.task_dir, "cache")
    shutil.rmtree(cache, ignore_errors=True)
    os.makedirs(os.path.join(cache, "datasets"))
    for sub in ("models", "hf_cache"):
        os.symlink(os.path.join(os.path.abspath(a.cache), sub), os.path.join(cache, sub))
    hdir = os.path.join(cache, "holdout")
    os.makedirs(hdir)
    zpath = os.path.join(cache, "datasets", f"{meta['task']}_tourn.zip")
    n_train = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for n in image_names(src):
            stem = os.path.splitext(n)[0]
            files = [n] + ([stem + ".txt"] if os.path.exists(os.path.join(src, stem + ".txt")) else [])
            if n in hidden:
                for f in files:
                    shutil.copy2(os.path.join(src, f), os.path.join(hdir, f))
            else:
                n_train += 1
                for f in files:
                    z.write(os.path.join(src, f), arcname=f)
    print(f"BUILD {meta['task'][:8]}: train {n_train} images -> {zpath}; hidden {sorted(hidden)} -> {hdir}")


# ------------------------------------------------------------------------------------------- rank
def cmd_rehearse(a):
    names = image_names(a.dataset_dir)
    rng = random.Random(a.seed)
    pick = sorted(rng.sample(names, a.n)) if a.n < len(names) else names
    h = hidden_count(len(pick))
    hidden = sorted(rng.sample(pick, h))
    out = os.path.join(a.out, a.name)
    sub = os.path.join(out, "images")
    shutil.rmtree(sub, ignore_errors=True)
    os.makedirs(sub)
    for n in pick:
        stem = os.path.splitext(n)[0]
        for f in [n] + ([stem + ".txt"] if os.path.exists(os.path.join(a.dataset_dir, stem + ".txt")) else []):
            shutil.copy2(os.path.join(a.dataset_dir, f), os.path.join(sub, f))
    task = str(uuid.UUID(hashlib.sha256(f"rehearse:{a.name}".encode()).hexdigest()[:32]))
    meta = {"tournament": "rehearsal", "task": task, "family": a.family, "base_repo": a.base_repo, "hours": a.hours,
            "dataset_dir": os.path.abspath(sub), "n_images": len(pick), "n_hidden": h, "entrants": [],
            "source": os.path.abspath(a.dataset_dir), "seed": a.seed}
    json.dump(meta, open(os.path.join(out, "replay.json"), "w"), indent=1)
    json.dump({"hidden": hidden, "identified": True, "rehearsal": True}, open(os.path.join(out, "hidden.json"), "w"), indent=1)
    print(f"REHEARSE {a.name}: {len(pick)} of {len(names)} images from {a.dataset_dir}; hidden {hidden}; given {len(pick) - h}; task {task}")


def cmd_rank(a):
    meta = json.load(open(os.path.join(a.task_dir, "replay.json")))
    pair = json.load(open(a.pair_json))
    ours = pair[a.name]["score"]
    if not meta["entrants"]:
        base = (pair.get("base") or {}).get("score")
        rel = f", {(ours / base - 1) * 100:+.3f}% vs base {base:.6f}" if base else ""
        print(f"RANK {meta['task'][:8]} {meta['family']}: ours {ours:.6f}{rel} (rehearsal: no entrants)")
        return
    board = sorted((e["test_loss"], e["hotkey"][:8]) for e in meta["entrants"] if e.get("test_loss") is not None)
    failed = [e["hotkey"][:8] for e in meta["entrants"] if e.get("test_loss") is None]
    rank = 1 + sum(1 for t, _ in board if t < ours)
    best = board[0][0] if board else None
    print(f"RANK {meta['task'][:8]} {meta['family']}: ours {ours:.6f} -> rank {rank} of {len(board) + 1} "
          f"(best entrant {best:.6f}, ours {(ours / best - 1) * 100:+.3f}% vs best; {len(failed)} entrants failed)")
    for t, hk in board[:6]:
        print(f"   {hk} {t:.6f} ({(ours / t - 1) * 100:+.3f}% ours vs it)")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--tournament", required=True)
    f.add_argument("--task", required=True)
    f.add_argument("--dataset-dir", required=True)
    f.add_argument("--out", required=True)
    i = sub.add_parser("identify")
    i.add_argument("--task-dir", required=True)
    i.add_argument("--base", required=True)
    i.add_argument("--repo-dir", default=None)
    i.add_argument("--max-entrants", type=int, default=2)
    i.add_argument("--tol", type=float, default=3e-4, help="max relative residual of the anchoring entrant (replica precision ~1e-4)")
    i.add_argument("--margin", type=float, default=20.0, help="the anchoring entrant's runner-up must be this many times worse")
    b = sub.add_parser("build")
    b.add_argument("--task-dir", required=True)
    b.add_argument("--cache", required=True)
    b.add_argument("--force", action="store_true")
    r = sub.add_parser("rank")
    r.add_argument("--task-dir", required=True)
    r.add_argument("--pair-json", required=True)
    r.add_argument("--name", required=True)
    h = sub.add_parser("rehearse")
    h.add_argument("--dataset-dir", required=True)
    h.add_argument("--name", required=True)
    h.add_argument("--family", required=True)
    h.add_argument("--base-repo", required=True)
    h.add_argument("--hours", type=float, required=True)
    h.add_argument("--n", type=int, required=True)
    h.add_argument("--seed", type=int, required=True)
    h.add_argument("--out", required=True)
    a = p.parse_args()
    {"fetch": cmd_fetch, "identify": cmd_identify, "build": cmd_build, "rank": cmd_rank, "rehearse": cmd_rehearse}[a.cmd](a)


if __name__ == "__main__":
    main()
