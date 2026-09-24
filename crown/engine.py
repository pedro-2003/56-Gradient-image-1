"""Training engine.

The loop runs *inside* Comfy's sampling call, so that `model_wrap(noisy, t,
**extra_args)` is the same callable the evaluator uses to score a candidate.
Loss and holdout share one formulation (scoring.py). Selection is raced and
paired (selector.py). Budget is measured, never assumed (budget.py).

Phase 2: when the holdout plateaus with budget left, a second member is
trained from a fresh seed and the two are averaged (soup). Every member keeps
the same holdout, so the soup is scored, not shipped blind.
"""

import gc
import json
import math
import os
import random
import time

import numpy as np
import torch

from . import contract as C
from .budget import Budget
from .lora import Lora, enable_checkpointing, save_lora, select_targets, soup_state
from .scoring import HoldoutScorer, flow_prediction_mse, make_noisy, sigma_of
from .selector import Candidate, Selector
from .twin import verify_loadable

LOG = print


def logs_enabled():
    return os.environ.get("GOD_TRAIN_LOGS", "").strip().lower() in {"1", "true", "yes", "on"}


def log(*a):
    if logs_enabled():
        LOG(time.strftime("%H:%M:%S"), *a, flush=True)


# --- model / clip / vae, loaded the way the evaluator loads them ------------------

def load_diffusion_model(sd, family, metadata=None):
    import comfy.model_sampling
    import comfy.sd

    fp8 = any(v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) for v in sd.values())
    opts = {} if fp8 else {"dtype": torch.bfloat16}
    model = comfy.sd.load_diffusion_model_state_dict(sd, model_options=opts, metadata=metadata, disable_dynamic=True)
    if model is None:
        raise RuntimeError("Comfy could not detect the diffusion model")
    for p in model.model.parameters():
        p.requires_grad_(False)
    shift = C.FAMILY_SHIFT.get(family)
    if shift is not None:  # evaluator: ImageFlowAdapter.configure -> ModelSamplingAuraFlow
        from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow

        model = ModelSamplingAuraFlow().patch_aura(model, shift)[0]
    sampling = model.model.model_sampling
    if not isinstance(sampling, comfy.model_sampling.CONST) or getattr(sampling, "noise_scale", 1.0) != 1.0:
        raise ValueError("Model is not unit-noise rectified flow")
    return model


def load_clip(sds, family):
    import comfy.sd

    return comfy.sd.load_text_encoder_state_dicts(
        sds, embedding_directory=None, clip_type=getattr(comfy.sd.CLIPType, C.CLIP_TYPES[family]),
        model_options={}, disable_dynamic=True)


def encode_prompt(clip, text, family):
    cond = clip.encode_from_tokens_scheduled(clip.tokenize(text))  # == nodes.CLIPTextEncode
    g = C.FAMILY_GUIDANCE.get(family)
    if g is not None:
        import node_helpers

        cond = node_helpers.conditioning_set_values(cond, {"guidance": g})
    return cond


class LatentEncoder:
    """Evaluator VAE encoding: Comfy VAE in fp32 on pixels in [0,1], or the
    diffusers posterior mean on pixels in [-1,1] (z-image)."""

    def __init__(self, kind, payload):
        self.kind = kind
        if kind == "comfy":
            import comfy.sd

            self.vae = comfy.sd.VAE(sd=payload, dtype=torch.float32)
            self.vae.first_stage_model.eval().requires_grad_(False)
            reg = getattr(self.vae.first_stage_model, "regularization", None)
            if reg is not None and getattr(reg, "sample", False):
                raise ValueError("Evaluation VAE must encode posterior means")
        else:
            from diffusers import AutoencoderKL

            self.vae = AutoencoderKL.from_pretrained(payload, torch_dtype=torch.float32).eval().requires_grad_(False).to("cuda")

    @torch.inference_mode()
    def encode(self, image):
        if self.kind == "comfy":
            pixels = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32) / 255.0).unsqueeze(0)
            lat = self.vae.encode(pixels).float().cpu()
        else:
            arr = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 127.5 - 1.0
            if arr.shape[0] % 8 or arr.shape[1] % 8:
                raise ValueError("Image dimensions must be divisible by eight")
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to("cuda")
            lat = self.vae.encode(t).latent_dist.mode().float().cpu()
        if not torch.isfinite(lat).all():
            raise ValueError("non-finite latent")
        return lat

    def free(self):
        import comfy.model_management as mm

        self.vae = None
        mm.unload_all_models()
        gc.collect()
        torch.cuda.empty_cache()


def make_guider(model, cond):
    import comfy.samplers

    g = comfy.samplers.CFGGuider(model)
    g.set_conds(cond, [])
    g.set_cfg(C.EVAL_CFG)
    return g


def evaluator_schedule(strata=C.EVAL_STRATA):
    """The sigma schedule the evaluator passes to guider.sample (descending + 0),
    so `transformer_options["sample_sigmas"]` matches at training time too."""
    return torch.tensor([sigma_of(b, strata) for b in reversed(range(strata))] + [0.0])


class TrainSampler:
    """Comfy 'sampler' whose .sample runs the training loop with the real model_wrap."""

    def __init__(self, trainer):
        self.trainer = trainer

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        self.trainer.run(model_wrap, extra_args)
        return latent_image


# --- the trainer ------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg, model, items, raw_conds, out_dir, budget: Budget, twin=None):
        self.base_score = None  # set by run(); save() reads it on the identity-only path too
        self.cfg, self.model, self.items, self.raw_conds, self.out_dir, self.budget = cfg, model, items, raw_conds, out_dir, budget
        self.twin = twin   # EvaluatorTwin or None: evaluator-exact scoring for the confirm stage
        self.device = model.load_device
        self.train_items = [i for i in items if not i["holdout"]]
        self.holdout_items = [i for i in items if i["holdout"]]
        if not self.train_items or not self.holdout_items:
            raise ValueError("need at least one training and one holdout image")
        self.rng = random.Random(cfg.seed)
        self._processed = {}
        self.summary = {"family": cfg.family, "steps": 0, "evals": [], "confirms": [], "members": [], "n_train": len(self.train_items),
                        "n_holdout": len(self.holdout_items)}
        targets = select_targets(model.model.diffusion_model, cfg.include, cfg.exclude)
        if not targets:
            raise ValueError("LoRA include/exclude selected no Linear layers")
        self.lora = Lora(targets, cfg.rank, cfg.alpha, self.device, cfg.seed)
        self.lora.attach(model)
        if cfg.ckpt:
            log("gradient checkpointing on", enable_checkpointing(model.model.diffusion_model, stride=cfg.ckpt_stride), "blocks")
        log(f"lora targets={len(targets)} params={sum(p.numel() for p in self.lora.params) / 1e6:.1f}M rank={cfg.rank}")

    # --- conditioning, processed exactly as the evaluator's sample() does ---------
    # The evaluator calls guider.sample(zeros, latent, ...) per image, which runs
    # process_conds against that image's (scaled) latent. Cache per (prompt, shape).
    def _cond(self, guider, prompt, item):
        lat = item["scaled"]
        key = (prompt, tuple(lat.shape))
        c = self._processed.get(key)
        if c is None:
            import comfy.sampler_helpers
            import comfy.samplers

            lat = lat.to(self.device)
            conds = {"positive": comfy.sampler_helpers.convert_cond(self.raw_conds[prompt]),
                     "negative": comfy.sampler_helpers.convert_cond([])}
            c = comfy.samplers.process_conds(guider.inner_model, torch.zeros_like(lat), conds, self.device, lat, None, C.EVAL_SAMPLER_SEED)
            self._processed[key] = c
        return c

    def _set_cond(self, guider, prompt, item):
        guider.conds = self._cond(guider, prompt, item)

    def _forward(self, guider, extra, prompt, item, noisy, sigma):
        self._set_cond(guider, prompt, item)
        t = torch.full((noisy.shape[0],), sigma, device=self.device, dtype=torch.float32)
        return guider(noisy, t, **extra)

    # --- artifact ------------------------------------------------------------------
    def save(self, state, scale, score, tag):
        os.makedirs(self.out_dir, exist_ok=True)
        tmp = os.path.join(self.out_dir, C.OUTPUT_LORA_NAME + ".tmp")
        save_lora(tmp, state, scale, metadata={"crown": tag})
        os.replace(tmp, os.path.join(self.out_dir, C.OUTPUT_LORA_NAME))
        self.summary["saved"] = {"tag": tag, "score": score, "base": self.base_score,
                                 "rel_pct": (score / self.base_score - 1) * 100 if self.base_score else None}
        self._dump()
        log(f"saved {tag} score={score:.6f}")

    def _dump(self):
        self.summary["holdout"] = [it["name"] for it in self.holdout_items]  # lets tools/parity_check.py rebuild the split
        tmp = os.path.join(self.out_dir, "summary.json.tmp")
        json.dump(self.summary, open(tmp, "w"), indent=1)
        os.replace(tmp, os.path.join(self.out_dir, "summary.json"))

    # --- scoring -------------------------------------------------------------------
    def _screen(self, guider, extra, cand: Candidate, noises=1):
        self.lora.set_tensors(cand.state, cand.scale)
        t0 = time.time()
        rep = self.scorer.score(guider, extra, lambda p, it: self._set_cond(guider, p, it), noises=noises)
        dt = time.time() - t0
        if noises == 1:
            self.band_loss = self._per_band(rep)   # latest screened candidate's per-band loss (for --band-power)
        self.lora.restore()
        if noises == 1:
            self.budget.observe_eval(dt)
            cand.screen = rep
            self.summary["evals"].append({"step": cand.step, "tag": cand.tag, "member": cand.member, "score": rep.score,
                                          "text": float(np.mean(rep.text)), "no_text": float(np.mean(rep.no_text)), "t": time.time()})
        else:
            cand.confirm = rep
            self.summary["confirms"].append({"step": cand.step, "tag": cand.tag, "member": cand.member, "noises": noises, "score": rep.score})
        rel = (rep.score / self.base_score - 1) * 100 if self.base_score else 0.0
        log(f"[{'confirm' if noises > 1 else 'screen'} m{cand.member} step {cand.step}] {cand.tag}: {rep.score:.6f} ({rel:+.2f}% vs base) {dt:.0f}s")
        return rep

    def _reload_training_model(self):
        """After the twin's patched clone was loaded, make sure the training model (with
        its LoRA hooks) is resident on the GPU again — Comfy may have evicted it."""
        try:
            import comfy.model_management as mm

            mm.load_models_gpu([self.model])
        except Exception as e:
            log(f"reload training model: {type(e).__name__}: {e}")

    def _twin_score(self, state, scale, noises):
        try:
            return self.twin.score({n: (u.to(self.device), d.to(self.device)) for n, (u, d) in state.items()}, scale, noises=noises)
        finally:
            self._reload_training_model()

    def _confirm(self, guider, extra, cand: Candidate):
        """Confirm stage: the evaluator twin when we have one (exact numerics, real
        LoraLoader path), otherwise the hook path at more noise draws."""
        noises = self.cfg.confirm_noises
        if self.twin is None:
            return self._screen(guider, extra, cand, noises=noises)
        t0 = time.time()
        rep, missing = self._twin_score(cand.state, cand.scale, noises)
        dt = time.time() - t0
        if rep is None:
            log(f"[twin m{cand.member} step {cand.step}] {cand.tag}: NOT LOADABLE by the evaluator: {missing[:3]}")
            self.summary.setdefault("twin_unloadable", []).append({"tag": cand.tag, "step": cand.step, "missing": missing[:5]})
            return None
        cand.confirm = rep
        tb = (self.summary.get("parity") or {}).get("twin_base")   # twin numbers compare against the twin base
        rel = (rep.score / tb - 1) * 100 if tb else ((rep.score / self.base_score - 1) * 100 if self.base_score else 0.0)
        hook = cand.screen.score if cand.screen is not None else None
        self.summary["confirms"].append({"step": cand.step, "tag": cand.tag, "member": cand.member, "noises": noises, "score": rep.score,
                                         "via": "twin", "hook_screen": hook})
        log(f"[twin m{cand.member} step {cand.step}] {cand.tag}: {rep.score:.6f} ({rel:+.2f}% vs twin base; hook screen {hook if hook is None else round(hook, 6)}) {dt:.0f}s")
        return rep

    def _screens_per_point(self):
        if self.cfg.ema > 0 and getattr(self.cfg, "screen_ema_only", False):
            return 1                                  # ema only at intermediate points
        return 2 if self.cfg.ema > 0 else 1          # raw (+ ema); avg2 only at member end

    def _eval_every(self):
        """Steps between eval points so that screening takes ~cfg.eval_share of wall-clock."""
        share = self.cfg.eval_share
        cost = self.budget.est_eval() * self._screens_per_point()
        return max(50, int(math.ceil(cost * (1 - share) / (share * self.budget.est_step()))))

    # --- run -----------------------------------------------------------------------
    def run(self, guider, extra):
        cfg = self.cfg
        for it in self.items:
            it["scaled"] = it["scaled"].to(self.device)
        self.scorer = HoldoutScorer(self.holdout_items, None, self.device)
        self.selector = Selector()

        # base: identity LoRA (up == 0). Saved before any training so an artifact always exists.
        ident = Candidate("identity", 0, self.lora.state(), self.lora.scale)
        self.base_score = None
        base = self._screen(guider, extra, ident)
        self.base_score = base.score
        self.summary["base_score"] = self.base_score
        self.summary["base_per_band"] = self._per_band(base)
        self.selector.add(ident)
        self.best = ident
        self.save(ident.state, ident.scale, ident.score, "identity")
        log(f"base holdout {self.base_score:.6f} on {len(self.holdout_items)} imgs; train imgs {len(self.train_items)}; eval {self.budget.est_eval():.0f}s")
        # parity meter: the same identity state scored by the evaluator twin at the same draws.
        # Any gap here is base-model / merge / forward numerics, not the LoRA — recorded, not hidden.
        if self.twin is not None:
            t0 = time.time()
            trep, missing = self._twin_score(ident.state, ident.scale, 1)
            if trep is not None:
                gap = (trep.score / self.base_score - 1) * 100
                self.summary["parity"] = {"hook_base": self.base_score, "twin_base": trep.score, "gap_pct": gap, "twin_source": self.twin.source, "t": time.time() - t0}
                log(f"parity: hook base {self.base_score:.6f} vs twin base {trep.score:.6f} -> gap {gap:+.3f}% ({time.time() - t0:.0f}s)")
            else:
                self.summary["parity"] = {"error": missing[:5]}
                log(f"parity: identity NOT LOADABLE by evaluator path: {missing[:3]}")

        members = []
        member = 0
        while True:
            if member > 0:
                self.lora.reinit(cfg.seed + 1000 * member)
            m_best, plateau_exit = self._train_member(guider, extra, member)
            members.append(m_best)
            self.summary["members"].append({"member": member, "best_tag": m_best.tag, "best_step": m_best.step,
                                            "score": m_best.score, "plateau_exit": plateau_exit})
            # phase 2: the member returned early because its holdout plateaued and the
            # remaining budget can plausibly carry a fresh member to a comparable optimum
            if not plateau_exit or member + 1 >= cfg.max_members:
                break
            member += 1
            log(f"plateau reached with {self.budget.remaining():.0f}s left; starting member {member}")

        # soup of member bests (exact rank-concat average), scored like any candidate
        if len(members) > 1 and self.budget.fits(self.budget.est_eval()):
            sc = Candidate("soup", self.summary["steps"], soup_state([m.state for m in members]), self.lora.scale, member=-1)
            self._screen(guider, extra, sc)
            self.selector.add(sc)

        # confirm the top candidates with more noise draws (the evaluator's own draws 0..k-1) —
        # through the evaluator twin where numerics differ — then 1-SE pick
        top = self.selector.top(self._confirm_count())
        for c in top:
            if not self.budget.fits(self.budget.est_eval(cfg.confirm_noises) * (1.5 if self.twin else 1.0)):
                break
            self._confirm(guider, extra, c)
        picked = self.selector.pick(use_confirm=any(c.confirm is not None for c in top)) or self.best
        # never ship something the evaluator cannot load: check the final state on the real loader path
        ok, missing = verify_loadable(self.model, {n: (u.to(self.device), d.to(self.device)) for n, (u, d) in picked.state.items()}, picked.scale)
        self.summary["final_loadable"] = {"ok": ok, "missing": missing[:5]}
        if not ok:
            log(f"FINAL ARTIFACT NOT LOADABLE ({missing[:3]}); falling back to best loadable candidate")
            for alt in sorted([c for c in self.selector.cands if c is not picked], key=lambda c: c.score):
                ok2, _ = verify_loadable(self.model, {n: (u.to(self.device), d.to(self.device)) for n, (u, d) in alt.state.items()}, alt.scale)
                if ok2:
                    picked = alt
                    break
        self.save(picked.state, picked.scale, picked.score, f"m{picked.member}:{picked.tag}@{picked.step}")
        rep = picked.confirm or picked.screen
        self.summary["final"] = {"picked": picked.tag, "step": picked.step, "member": picked.member, "score": picked.score,
                                 "base": self.base_score, "rel_pct": (picked.score / self.base_score - 1) * 100,
                                 "per_band": self._per_band(rep), "per_image": rep.per_image, "wall_s": self.budget.elapsed()}
        self._dump()
        log("final:", json.dumps(self.summary["final"]))

    def _per_band(self, rep):
        acc = {}
        for (sha, mode, band, k), v in rep.per_case.items():
            acc.setdefault(band, []).append(v)
        return [float(np.mean(acc[b])) for b in sorted(acc)]

    def _train_member(self, guider, extra, member):
        cfg, lora, budget = self.cfg, self.lora, self.budget
        groups = [{"params": lora.up_params, "lr_mult": cfg.lora_plus_ratio}, {"params": lora.down_params, "lr_mult": 1.0}]
        opt = torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.99), weight_decay=cfg.weight_decay, eps=1e-8)
        ema = [p.detach().clone() for p in lora.params] if cfg.ema > 0 else None
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 7919 * member)
        t_start = time.time()
        # this member's wall-clock horizon: whatever is left, minus what the confirm stage will need
        horizon_end = time.time() + max(60.0, budget.remaining() - budget.est_eval(cfg.confirm_noises) * cfg.confirm_top)
        step, last_eval, cases, raw_hist = 0, 0, [], []
        m_best = self.best
        plateau_exit = False
        peak, warm_base, polishing, tag_prefix, pol_points = cfg.lr, 0, False, "", 0
        while True:
            now = time.time()
            due = (step - last_eval) >= self._eval_every()
            need = budget.est_step() + (budget.est_eval() * (self._screens_per_point() + 0.2) + 5 if due else 0)
            if now + need > horizon_end or not budget.fits(need):
                break
            # phase-2 exit: this member's screens have plateaued and another member could
            # reach a comparable optimum (time-to-best of this member, plus its evals) in the time left
            want_phase2 = cfg.phase2 and member + 1 < cfg.max_members
            want_polish = getattr(cfg, "polish", False) and not polishing
            plateau = (want_phase2 or want_polish or polishing) and self.selector.plateaued(window=getattr(cfg, "plateau_window", 3), member=member)
            if polishing and pol_points >= 3 and plateau:
                # the anneal has converged too: hand the rest to the confirm stage
                log(f"m{member} polish plateau at step {step} (best {m_best.tag}@{m_best.step}); {budget.remaining():.0f}s left for confirms")
                break
            if plateau and want_phase2:
                need_next = max(1, m_best.step) * budget.est_step() + 3 * budget.est_eval() + budget.est_eval(cfg.confirm_noises) * cfg.confirm_top
                if budget.fits(need_next):
                    plateau_exit = True
                    log(f"m{member} plateau at step {step} (best {m_best.tag}@{m_best.step}); handing {budget.remaining():.0f}s to the next member")
                    break
            if plateau and want_polish:
                if (m_best.step > 0 and m_best.member == member
                        and horizon_end - now > 3 * budget.est_eval() + 60):
                    # anneal from the best state with a low, decaying LR for the time left, instead of
                    # training on past the optimum (the champion's telemetry shows 50-90% of budget lost there)
                    lora.load_state({n: (u.to(self.device), d.to(self.device)) for n, (u, d) in m_best.state.items()})
                    peak = cfg.lr * cfg.polish_lr_frac
                    opt = torch.optim.AdamW(groups, lr=peak, betas=(0.9, 0.99), weight_decay=cfg.weight_decay, eps=1e-8)
                    if ema is not None:
                        ema = [q.detach().clone() for q in lora.params]
                    t_start, warm_base, polishing, tag_prefix = now, step, True, "pol-"
                    raw_hist.clear()
                    self.summary["polish"] = {"from": f"{m_best.tag}@{m_best.step}", "at_step": step, "peak_lr": peak, "seconds": horizon_end - now}
                    log(f"m{member} plateau at step {step}: polishing from {m_best.tag}@{m_best.step} at lr {peak:.1e} for {horizon_end - now:.0f}s")
            p = min(1.0, (now - t_start) / max(1.0, horizon_end - t_start))
            warm = min(1.0, (step - warm_base + 1) / max(1, cfg.warmup_steps))
            lr = peak * warm * (cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * p)))
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_mult"]

            # case cycle: every (image, band) pair once per pass, shuffled; uniform over bands like the score
            if not cases:
                reps = self._band_repeats()
                cases = [(i, b) for i in range(len(self.train_items)) for b in range(C.EVAL_STRATA) for _ in range(reps[b])]
                self.rng.shuffle(cases)
            i, band = cases.pop()
            item = self.train_items[i]
            sigma = sigma_of(band)
            noise = torch.randn(item["scaled"].shape, generator=gen, dtype=torch.float32).to(self.device)
            frac = float(getattr(cfg, "empty_prompt_frac", 0.5))
            if frac == 0.5:
                prompt = item["caption"] if (self.summary["steps"] % 2 == 0) else ""   # 50/50, like the score
            else:
                prompt = "" if self.rng.random() < frac else item["caption"]
            ts = time.time()
            with torch.enable_grad():
                noisy, target = make_noisy(item["scaled"], noise, sigma)
                den = self._forward(guider, extra, prompt, item, noisy, sigma)
                loss = flow_prediction_mse(noisy, den, target, sigma).mean()
                loss.backward()
            torch.nn.utils.clip_grad_norm_(lora.params, cfg.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                with torch.no_grad():
                    for e, q in zip(ema, lora.params):
                        e.mul_(cfg.ema).add_(q.detach(), alpha=1 - cfg.ema)
            budget.observe_step(time.time() - ts)
            step += 1
            self.summary["steps"] += 1
            if step % 25 == 0:
                log(f"m{member} step {step} loss {loss.item():.4f} band {band} lr {lr:.2e} step {budget.est_step():.2f}s left {horizon_end - time.time():.0f}s")
            if due:
                last_eval = step
                m_best = self._screen_candidates(guider, extra, member, step, ema, raw_hist, m_best, tag_prefix=tag_prefix)
                if polishing:
                    pol_points += 1
        # member end: a final screen (if the last eval point is stale) plus the average of the
        # last two raw states — one extra candidate that costs one screen per member, not per point
        if step > last_eval and not plateau_exit and budget.fits(budget.est_eval() * (2.2 if ema is not None else 1.2)):
            m_best = self._screen_candidates(guider, extra, member, step, ema, raw_hist, m_best, tag_prefix=tag_prefix, final=True)
        if len(raw_hist) >= 2 and budget.fits(budget.est_eval() * 1.2):
            m_best = self._screen_candidates(guider, extra, member, step, None, raw_hist, m_best, avg_only=True, tag_prefix=tag_prefix)
        log(f"member {member} done: steps={step} best={m_best.tag}@{m_best.step} {m_best.score:.6f} plateau_exit={plateau_exit}")
        return m_best, plateau_exit

    def _confirm_count(self):
        """How many top candidates to confirm: at least --confirm-top, more when time is left over
        (a member that stopped at its plateau leaves it), never more than 8."""
        cfg = self.cfg
        per = max(1.0, self.budget.est_eval(cfg.confirm_noises))
        spare = self.budget.remaining() - C.PUBLISH_RESERVE_S - 30.0
        k = int(spare // per)
        k = min(8, max(cfg.confirm_top, k))
        if k != cfg.confirm_top:
            log(f"confirm stage: {k} candidates fit at {cfg.confirm_noises} noises ({spare:.0f}s spare, {per:.0f}s each)")
        return k

    def _band_repeats(self):
        """How many times each sigma band appears per case-cycle pass: 1 (uniform) unless --band-power
        is set, then (loss_b / median loss)^p rounded, from the latest holdout per-band loss."""
        n = C.EVAL_STRATA
        bl = getattr(self, "band_loss", None)
        power = float(getattr(self.cfg, "band_power", 0.0) or 0.0)
        if power <= 0 or not bl or len(bl) != n:
            return [1] * n
        med = float(np.median(bl)) or 1.0
        reps = [max(1, int(round((float(v) / med) ** power))) for v in bl]
        if self.summary.get("band_repeats") != reps:
            self.summary["band_repeats"] = reps
            log(f"band repeats (power {power}): {reps}")
        return reps

    def _screen_candidates(self, guider, extra, member, step, ema, raw_hist, m_best, avg_only=False, tag_prefix="", final=False):
        lora = self.lora
        cands = []
        if avg_only:
            cands.append(Candidate(tag_prefix + "avg2", step, soup_state(raw_hist[-2:]), lora.scale, member=member))
        else:
            raw_state = {n: (u.detach().cpu().clone(), d.detach().cpu().clone()) for n, (u, d) in lora.state().items()}
            ema_only = ema is not None and getattr(self.cfg, "screen_ema_only", False) and not final
            if not ema_only:
                cands.append(Candidate(tag_prefix + "raw", step, raw_state, lora.scale, member=member))
            if ema is not None:
                cands.append(Candidate(tag_prefix + "ema", step, {n: (t[0].cpu(), t[1].cpu()) for n, t in lora.state_from_flat([e.clone() for e in ema]).items()}, lora.scale, member=member))
            raw_hist.append(raw_state)
        for c in cands:
            c.state = {n: (u.to(self.device), d.to(self.device)) for n, (u, d) in c.state.items()}
            self._screen(guider, extra, c)
            c.state = {n: (u.cpu(), d.cpu()) for n, (u, d) in c.state.items()}
            self.selector.add(c)
            if c.score < m_best.score:
                m_best = c
            if c.score < self.best.score:
                self.best = c
                self.save({n: (u.to(self.device), d.to(self.device)) for n, (u, d) in c.state.items()}, c.scale, c.score, f"m{member}:{c.tag}@{step}")
        return m_best
