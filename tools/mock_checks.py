"""Defect-specific CPU checks for the v6.1 fixes, on top of tools/mock_run.py's fakes. Each check
builds a fresh Trainer on the tiny mock model and asserts the behaviour the audit found broken:

  identity_first      an artifact exists before the first screen (a failing first forward cannot leave nothing)
  restore_on_error    a scorer exception during a screen leaves the LoRA wrappers on the training params
  per_member_soup     with two members the second member's best is its OWN candidate and the soup exists
  nonfinite_skip      non-finite losses are skipped and counted, the run completes
  unloadable_veto     a candidate the twin refuses is never picked
  selector_rules      1-SE tiebreak prefers the better of two same-step candidates; top() is one per point
  cadence_cap         with --adaptive-cadence at least three eval points still fit a short horizon
  trailing_ignores_0  the identity point is not counted as an improvement

    python tools/mock_checks.py            (exit 1 on any failure)
"""
import importlib.util
import json
import os
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
spec = importlib.util.spec_from_file_location("mock_run", os.path.join(HERE, "mock_run.py"))
mock_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mock_run)          # installs the fake comfy modules

import torch  # noqa: E402

from crown import engine  # noqa: E402
from crown.budget import Budget  # noqa: E402
from crown.selector import Candidate, Selector  # noqa: E402
from crown.scoring import ScoreReport  # noqa: E402

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"PASS {name}")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"FAIL {name}: {type(e).__name__}: {e}")


def build(seconds=40, n_images=12, max_members=1, twin=True, **over):
    torch.manual_seed(0)
    Cc, H, W = 4, 8, 8
    prompts = ["", "a red cube", "a blue sphere", "a green cone"]
    cond_vecs = {p: (torch.zeros(1, 8) if p == "" else torch.randn(1, 8)) for p in prompts}
    items = []
    for i in range(n_images):
        cap = prompts[1 + i % 3]
        x = torch.randn(1, Cc, H, W) * 0.5 + 0.3 * cond_vecs[cap].mean()
        items.append({"name": f"{i:03d}.png", "caption": cap, "sha256": f"{i:064x}", "scaled": x, "holdout": i % 4 == 0})
    dm = mock_run.TinyFlow(Cc * H * W)
    model = mock_run.FakeModel(dm)
    guider = mock_run.FakeGuider(model, cond_vecs)
    cfg = types.SimpleNamespace(
        family="mock", seed=0, include=None, exclude=None, rank=4, alpha=4.0, ckpt=False, ckpt_stride=1,
        lr=3e-3, lr_final_frac=0.1, warmup_steps=5, weight_decay=0.0, grad_clip=1.0, lora_plus_ratio=1.0, ema=0.99,
        eval_share=0.25, confirm_top=3, confirm_noises=4, phase2=True, max_members=max_members,
        polish=False, polish_lr_frac=0.3, band_power=0.0, plateau_window=2, screen_ema_only=False,
        empty_prompt_frac=0.5, select_metric="mean", eval_every=0, holdout_names="", replan=False, replan_max_members=3,
        init_lora="", flip=False, adaptive_cadence=False)
    for k, v in over.items():
        setattr(cfg, k, v)
    out = tempfile.mkdtemp(prefix="crown-check-")
    budget = Budget(time.time() + seconds, kill_margin_s=0.0, publish_reserve_s=2.0)
    os.environ["GOD_TRAIN_LOGS"] = "0"
    extra = {"model_options": {}, "seed": 42}

    class FakeTwin:
        source = "mock"
        base = model
        veto_steps = set()

        def __init__(self, tr_ref):
            self.tr_ref = tr_ref

        def _load(self):
            pass

        def release(self):
            pass

        def score(self, state, scale, noises=1):
            tr = self.tr_ref[0]
            from crown.twin import apply_lora_like_evaluator
            _, missing = apply_lora_like_evaluator(model, state, scale)
            if missing:
                return None, missing
            if getattr(self, "veto_current", None) is not None and self.veto_current(state):
                return None, ["vetoed-by-check"]
            tr.lora.set_tensors(state, scale)
            rep = tr.scorer.score(guider, extra, lambda p, it: tr._set_cond(guider, p, it), noises=noises)
            tr.lora.restore()
            return rep, []

    ref = []
    tw = FakeTwin(ref) if twin else None
    tr = engine.Trainer(cfg, model, items, {p: p for p in prompts}, out, budget, twin=tw)
    ref.append(tr)
    return tr, guider, extra, out, tw


def run(tr, guider, extra):
    tr.run(guider, extra)
    return json.load(open(os.path.join(tr.out_dir, "summary.json")))


# --- checks ------------------------------------------------------------------------

def identity_first():
    import crown.scoring as scoring
    tr, guider, extra, out, _ = build(seconds=20)
    orig = scoring.HoldoutScorer.score

    def boom(*a, **k):
        raise RuntimeError("first forward exploded")
    scoring.HoldoutScorer.score = boom
    try:
        tr.run(guider, extra)
        raise AssertionError("run should have raised")
    except RuntimeError:
        pass
    finally:
        scoring.HoldoutScorer.score = orig
    assert os.path.exists(os.path.join(out, "last.safetensors")), "no artifact before the first screen"


def restore_on_error():
    from crown.scoring import HoldoutScorer
    tr, guider, extra, out, _ = build(seconds=20)
    tr.scorer = HoldoutScorer(tr.holdout_items, None, tr.device)     # run() would build it; we screen directly
    orig = tr.scorer.score

    def boom(*a, **k):
        raise ValueError("Non-finite denoising MSE")
    tr.scorer.score = boom
    cand = Candidate("x", 1, {n: (torch.randn_like(u), torch.randn_like(d)) for n, (u, d) in tr.lora.state().items()}, tr.lora.scale)
    try:
        tr._screen(guider, extra, cand)
    except ValueError:
        pass
    for i, n in enumerate(tr.lora.names):
        w = tr.lora.wrappers[n]
        assert w.up is tr.lora.params[2 * i] and w.down is tr.lora.params[2 * i + 1], f"wrapper {n} left on the candidate"
    tr.scorer.score = orig


def per_member_soup():
    tr, guider, extra, out, _ = build(seconds=60, max_members=2)
    s = run(tr, guider, extra)
    members = s["members"]
    assert len(members) == 2, f"phase 2 did not run (members={len(members)}); lengthen the horizon"
    assert members[1]["best_member"] == 1, f"member 1's best is member {members[1]['best_member']}'s candidate (the old defect)"
    soups = [c for c in tr.selector.cands if c.tag == "soup"]
    assert soups, "no soup candidate"
    r = tr.lora.rank
    assert next(iter(soups[0].state.values()))[0].shape[1] == 2 * r, "soup is not a rank-concat of two states"


def nonfinite_skip():
    tr, guider, extra, out, _ = build(seconds=25)
    calls = {"n": 0}
    orig = engine.flow_prediction_mse

    def flaky(noisy, den, target, sigma):
        calls["n"] += 1
        if calls["n"] % 7 == 0:
            raise ValueError("Non-finite denoising MSE")
        return orig(noisy, den, target, sigma)
    engine.flow_prediction_mse = flaky
    try:
        s = run(tr, guider, extra)
    finally:
        engine.flow_prediction_mse = orig
    assert s.get("nonfinite", 0) > 0, "non-finite steps were not counted"
    assert s["steps"] > 20, "training did not continue past the non-finite steps"
    assert os.path.exists(os.path.join(out, "last.safetensors"))


def unloadable_veto():
    tr, guider, extra, out, tw = build(seconds=40)
    # veto the first candidate the CONFIRM stage sends to the twin (the parity call is not a confirm)
    state = {"armed": False, "done": False}
    orig_confirm = tr._confirm

    def confirm(g, e, c):
        state["armed"] = not state["done"]
        try:
            return orig_confirm(g, e, c)
        finally:
            state["armed"] = False
    tr._confirm = confirm

    def veto(st):
        if state["armed"] and not state["done"]:
            state["done"] = True
            return True
        return False
    tw.veto_current = veto
    s = run(tr, guider, extra)
    vetoed = s.get("twin_unloadable") or []
    assert vetoed, "the twin veto never fired"
    fin = s["final"]
    assert not any(v["tag"] == fin["picked"] and v["step"] == fin["step"] for v in vetoed), "a vetoed candidate was shipped"
    assert all(not c.unloadable or c is not tr.selector.pick() for c in tr.selector.cands)


def _rep(per_case_scores, per_image=(1.0,)):
    per_case = {("img", "text", b, 0): v for b, v in enumerate(per_case_scores)}
    return ScoreReport(float(sum(per_case_scores) / len(per_case_scores)), list(per_image), [1.0], [1.0], per_case)


def selector_rules():
    sel = Selector()
    a = Candidate("raw", 100, {}, 1.0, screen=_rep([1.0, 1.0, 1.0, 1.0]), confirm=_rep([0.90, 0.91, 0.92, 0.93]))
    b = Candidate("ema", 100, {}, 1.0, screen=_rep([1.0, 1.0, 1.0, 1.0]), confirm=_rep([0.89, 0.90, 0.91, 0.92]))
    c = Candidate("raw", 200, {}, 1.0, screen=_rep([0.8, 0.8, 0.8, 0.8]), confirm=_rep([0.95, 0.96, 0.97, 0.98]))
    for x in (a, b, c):
        sel.add(x)
    p = sel.pick(use_confirm=True)
    assert p is b, f"tiebreak picked {p.tag}@{p.step}, expected ema@100 (better of the same step)"
    top = sel.top(3)
    assert len(top) == 2 and {(t.member, t.step) for t in top} == {(0, 100), (0, 200)}, "top() did not dedupe per point"
    c.unloadable = True
    assert c not in sel.top(3) and sel.best() is not c, "unloadable candidate still ranked"


def cadence_cap():
    tr, guider, extra, out, _ = build(seconds=25, adaptive_cadence=True)
    s = run(tr, guider, extra)
    steps = {e.get("step") for e in s["evals"] if e.get("step", 0) > 0}
    assert len(steps) >= 3, f"only {len(steps)} eval points with adaptive cadence on a short horizon"


def trailing_ignores_0():
    sel = Selector()
    sel.add(Candidate("identity", 0, {}, 1.0, screen=_rep([1.0, 1.0, 1.0, 1.0])))
    sel.add(Candidate("ema", 100, {}, 1.0, screen=_rep([0.5, 0.5, 0.5, 0.5])))
    assert sel.trailing_improvements(0) == 0, "identity->first point counted as an improvement"
    sel.add(Candidate("ema", 200, {}, 1.0, screen=_rep([0.4, 0.4, 0.4, 0.4])))
    assert sel.trailing_improvements(0) == 1


if __name__ == "__main__":
    for name, fn in [("identity_first", identity_first), ("restore_on_error", restore_on_error), ("per_member_soup", per_member_soup),
                     ("nonfinite_skip", nonfinite_skip), ("unloadable_veto", unloadable_veto), ("selector_rules", selector_rules),
                     ("cadence_cap", cadence_cap), ("trailing_ignores_0", trailing_ignores_0)]:
        check(name, fn)
    bad = [r for r in RESULTS if not r[1]]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    sys.exit(1 if bad else 0)
