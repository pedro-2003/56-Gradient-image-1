"""G-PARITY gate: score a finished run's LoRA with the validator's own evaluator code
(tools/local_evaluate.py, vendored verbatim) on the run's holdout images, and compare with
the numbers the trainer computed for the same candidate.

    python tools/parity_check.py --family z-image --dataset-dir research/datasets/59b62bbc_z-image \
        --run-dir /workspace/outputs/<task>/<repo> --base /cache/models/.../z_image_turbo_bf16.safetensors \
        --out /root/parity/zimage [--noises 16] [--base-too]

Three comparisons, all on the same holdout and the same case seeds:
  * evaluator @1 noise  vs the trainer's hook screen  (bf16 families must agree to < 0.05%);
  * evaluator @k noises vs the trainer's confirm at k noises (the evaluator twin on ideogram4/qwen,
    the hook otherwise) — this is the number the selection is made on;
  * evaluator @--noises (default 16, the validator's DEFAULT_NOISES) = what the validator would report,
    plus the same for a zeroed copy of the LoRA (= the base model) with --base-too.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from crown import data  # noqa: E402


def summarize(raw):
    """local_evaluate's --out file is the evaluator's raw result; reduce it like the validator does."""
    if "score" in raw and "per_image" in raw:
        return raw
    if "eval_loss" not in raw:   # {"model_params_count": ..., "local/lora": {"eval_loss": {...}, "is_finetune": ...}}
        raw = next(v for v in raw.values() if isinstance(v, dict) and "eval_loss" in v)
    t, n = raw["eval_loss"]["text_guided_losses"], raw["eval_loss"]["no_text_losses"]
    per = [0.5 * x + 0.5 * y for x, y in zip(t, n)]
    return {"score": sum(per) / len(per), "per_image": per, "text": t, "no_text": n}


def run_eval(family, base, lora, dataset, out, noises, repo_dir=None):
    cmd = [sys.executable, os.path.join(HERE, "local_evaluate.py"), "--family", family, "--base", base, "--lora", lora,
           "--dataset", dataset, "--out", out, "--noises", str(noises)]
    if repo_dir:
        cmd += ["--repo-dir", repo_dir]
    # the evaluator fetches its own text encoders / VAE from HF (Comfy-Org/* repos); the venv sets
    # HF_HUB_OFFLINE=1 to mirror the validator's trainer container, so lift it for the evaluator only
    env = {**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stdout[:1500], "...", r.stdout[-1500:], r.stderr[:1500], "...", r.stderr[-1500:])   # head AND tail: a tail alone hides the first failure
        raise SystemExit(f"local_evaluate failed rc={r.returncode}")
    if os.path.exists(out):
        return summarize(json.load(open(out)))
    return summarize(json.loads(r.stdout.strip().splitlines()[-1]))


def pct(a, b):
    return (a / b - 1) * 100 if b else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--run-dir", required=True, help="dir with last.safetensors + summary.json")
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--noises", type=int, default=16)
    p.add_argument("--base-too", action="store_true")
    p.add_argument("--holdout-frac", type=float, default=0.10)
    p.add_argument("--holdout-min", type=int, default=3)
    p.add_argument("--repo-dir", default=None, help="base repo snapshot dir (z-image)")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    summary = json.load(open(os.path.join(a.run_dir, "summary.json")))
    lora = os.path.join(a.run_dir, "last.safetensors")

    # rebuild the holdout split exactly as the run did (seeded from the dataset digests)
    items = data.load_items(a.dataset_dir)
    hold = data.choose_holdout(items, a.holdout_frac, a.holdout_min)
    names = [items[i]["name"] for i in hold]
    if summary.get("holdout") and summary["holdout"] != names:
        raise SystemExit(f"holdout mismatch: run {summary['holdout']} vs rebuilt {names}")
    hdir = os.path.join(a.out, "holdout")
    shutil.rmtree(hdir, ignore_errors=True)
    os.makedirs(hdir)
    for n in names:
        stem = os.path.splitext(n)[0]
        shutil.copy2(os.path.join(a.dataset_dir, n), os.path.join(hdir, n))
        shutil.copy2(os.path.join(a.dataset_dir, stem + ".txt"), os.path.join(hdir, stem + ".txt"))
    print(f"holdout {len(names)}: {names}")

    final = summary.get("final") or {}
    picked, step = final.get("picked"), final.get("step")
    screen = next((e for e in summary.get("evals", []) if e["tag"] == picked and e["step"] == step), None)
    confirm = next((c for c in summary.get("confirms", []) if c["tag"] == picked and c["step"] == step), None)
    print(f"trainer: picked {picked}@{step} score {final.get('score')} base {final.get('base')} rel {final.get('rel_pct')}; "
          f"screen(1 noise) {screen and screen['score']}; confirm {confirm}")
    result = {"holdout": names, "trainer": final, "screen": screen, "confirm": confirm, "parity_meter": summary.get("parity")}

    r1 = run_eval(a.family, a.base, lora, hdir, os.path.join(a.out, "eval_n1.json"), 1, a.repo_dir)
    gap1 = pct(r1["score"], screen["score"]) if screen else float("nan")
    print(f"evaluator(1 noise): {r1['score']:.6f}  vs hook screen {screen['score'] if screen else float('nan'):.6f}  gap {gap1:+.4f}%")
    print("  per-image evaluator:", [round(x, 6) for x in r1["per_image"]])
    if final.get("per_image"):
        print("  per-image trainer:  ", [round(x, 6) for x in final["per_image"]])
    result.update({"eval_n1": r1, "gap_screen_pct": gap1})

    gap_c = float("nan")
    if confirm and confirm.get("noises", 1) > 1:
        k = confirm["noises"]
        rk = run_eval(a.family, a.base, lora, hdir, os.path.join(a.out, f"eval_n{k}.json"), k, a.repo_dir)
        gap_c = pct(rk["score"], confirm["score"])
        print(f"evaluator({k} noises): {rk['score']:.6f}  vs trainer confirm via {confirm.get('via', 'hook')} {confirm['score']:.6f}  gap {gap_c:+.4f}%")
        result.update({f"eval_n{k}": rk, "gap_confirm_pct": gap_c, "confirm_via": confirm.get("via", "hook")})

    if a.noises > 1:
        rN = run_eval(a.family, a.base, lora, hdir, os.path.join(a.out, f"eval_n{a.noises}.json"), a.noises, a.repo_dir)
        print(f"evaluator({a.noises} noises): {rN['score']:.6f}")
        result[f"eval_n{a.noises}"] = rN
        if a.base_too:
            from safetensors.torch import load_file, save_file
            sd = load_file(lora)
            zero = {k: (v * 0 if k.endswith("lora_up.weight") else v) for k, v in sd.items()}
            zpath = os.path.join(a.out, "identity.safetensors")
            save_file(zero, zpath)
            rB = run_eval(a.family, a.base, zpath, hdir, os.path.join(a.out, f"base_n{a.noises}.json"), a.noises, a.repo_dir)
            print(f"evaluator base({a.noises} noises): {rB['score']:.6f}  -> gain {pct(rN['score'], rB['score']):+.3f}%")
            result[f"base_n{a.noises}"] = rB
            result["gain_pct"] = pct(rN["score"], rB["score"])
    json.dump(result, open(os.path.join(a.out, "parity.json"), "w"), indent=1)
    # the selection number is what must match; on ideogram4/qwen that is the twin confirm
    decisive = gap_c if confirm and confirm.get("noises", 1) > 1 else gap1
    print("PARITY_OK" if abs(decisive) < 0.05 else "PARITY_GAP", f"decisive gap {decisive:+.4f}%")


if __name__ == "__main__":
    main()
