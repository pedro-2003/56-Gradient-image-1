"""Orchestration: arguments -> assets -> latents -> conditioning -> model -> train.

Invoked by scripts/image_trainer.py inside the validator's container; can also
be run directly for local experiments.
"""

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=["flux", "z-image", "qwen-image", "ideogram4", "krea2"])
    p.add_argument("--model-dir", required=True)
    p.add_argument("--dataset", required=True, help="zip or directory of images + .txt captions")
    p.add_argument("--out", required=True)
    p.add_argument("--deadline-ts", type=float, required=True, help="unix time at which the validator kills the container")
    p.add_argument("--trigger-word", default=None)
    p.add_argument("--comfy-root", default=os.environ.get("COMFY_ROOT", "/opt/ComfyUI"))
    p.add_argument("--baked-dir", default=os.environ.get("CROWN_ASSETS", "/opt/crown/assets"))
    # LoRA
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=None, help="default = rank (scale 1.0)")
    p.add_argument("--include", default=None, help="regex over Linear module names; default = every Linear")
    p.add_argument("--exclude", default=None)
    # optimisation
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-final-frac", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lora-plus-ratio", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--ckpt", dest="ckpt", action="store_true", default=True)
    p.add_argument("--no-ckpt", dest="ckpt", action="store_false")
    p.add_argument("--ckpt-stride", type=int, default=1)
    p.add_argument("--text-enc-dtype", default=None, choices=[None, "fp16", "bf16", "fp32"])
    # holdout / selection / phase 2
    p.add_argument("--holdout-frac", type=float, default=0.10)
    p.add_argument("--holdout-min", type=int, default=3)
    p.add_argument("--eval-share", type=float, default=0.15, help="target fraction of wall-clock spent on holdout screens")
    p.add_argument("--confirm-top", type=int, default=3)
    p.add_argument("--confirm-noises", type=int, default=4, help="evaluator noise draws per stratum for the confirm stage")
    p.add_argument("--phase2", dest="phase2", action="store_true", default=True)
    p.add_argument("--no-phase2", dest="phase2", action="store_false")
    p.add_argument("--max-members", type=int, default=2)
    p.add_argument("--twin", dest="twin", action="store_true", default=True,
                   help="score the confirm stage through the evaluator twin on ideogram4/qwen-image")
    p.add_argument("--no-twin", dest="twin", action="store_false")
    p.add_argument("--identity-only", action="store_true", help="load, export the identity LoRA, exit (fallback path)")
    p.add_argument("--polish", action="store_true",
                   help="when a member plateaus and a second member cannot fit, restart from its best state with a low decaying LR "
                        "for the time left instead of training on past the optimum")
    p.add_argument("--polish-lr-frac", type=float, default=0.3, help="polish peak LR as a fraction of --lr")
    p.add_argument("--plateau-window", type=int, default=3, help="eval points without a real improvement that count as a plateau")
    p.add_argument("--screen-ema-only", action="store_true",
                   help="screen only the EMA candidate at intermediate eval points (raw at the member end), doubling the eval cadence")
    p.add_argument("--empty-prompt-frac", type=float, default=0.5,
                   help="share of training steps with the empty prompt; 0.5 alternates exactly like the score's 50/50 mix")
    p.add_argument("--band-power", type=float, default=0.0,
                   help="0 = uniform over sigma bands (like the score); p>0 repeats band b (loss_b/median)^p times per pass, "
                        "loss_b being the latest holdout per-band loss - importance sampling toward where the score mass is")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    cfg = parse(argv)
    if cfg.alpha is None:
        cfg.alpha = float(cfg.rank)
    if cfg.text_enc_dtype is None:
        cfg.text_enc_dtype = "bf16" if cfg.family == "qwen-image" else "fp16"
    os.makedirs(cfg.out, exist_ok=True)
    json.dump(vars(cfg), open(os.path.join(cfg.out, "config.json"), "w"), indent=1)

    from crown import comfy_boot, data, engine
    from crown.assets import Assets
    from crown.budget import Budget

    budget = Budget(cfg.deadline_ts)
    mm = comfy_boot.boot(cfg.comfy_root, text_enc_dtype=cfg.text_enc_dtype)
    if cfg.family == "flux":
        from crown import flux_autograd

        flux_autograd.apply()
    import torch

    engine.log(f"family={cfg.family} budget {budget.remaining() / 60:.1f} min; device {mm.get_torch_device()}")

    items = data.load_items(cfg.dataset, cfg.trigger_word)
    hold = data.choose_holdout(items, cfg.holdout_frac, cfg.holdout_min)
    engine.log(f"dataset: {len(items)} images, holdout {len(hold)}: {[items[i]['name'] for i in hold]}")

    assets = Assets(cfg.family, cfg.model_dir, cfg.baked_dir)

    t = time.time()
    kind, payload = assets.vae()
    enc = engine.LatentEncoder(kind, payload)
    for it in items:
        it["latent"] = enc.encode(it["image"])
        it["image"] = None
    enc.free()
    engine.log(f"latents ({kind}) in {time.time() - t:.0f}s; shape {tuple(items[0]['latent'].shape)}")

    t = time.time()
    clip = engine.load_clip(assets.text_encoder_sds(), cfg.family)
    raw_conds = {}
    for prompt in sorted({it["caption"] for it in items} | {""}):
        raw_conds[prompt] = engine.encode_prompt(clip, prompt, cfg.family)
    del clip
    mm.unload_all_models()
    gc.collect()
    torch.cuda.empty_cache()
    engine.log(f"{len(raw_conds)} prompts encoded in {time.time() - t:.0f}s")

    t = time.time()
    model = engine.load_diffusion_model(assets.diffusion_sd(), cfg.family)
    assets.release()
    gc.collect()
    for it in items:
        it["scaled"] = model.model.process_latent_in(it["latent"].clone()).float().cpu()
        it["latent"] = None
    engine.log(f"model in {time.time() - t:.0f}s: {type(model.model).__name__} "
               f"{sum(p.numel() for p in model.model.diffusion_model.parameters()) / 1e9:.2f}B")

    twin = None
    if cfg.twin and cfg.family in ("ideogram4", "qwen-image") and not cfg.identity_only:
        from crown.twin import EvaluatorTwin

        try:
            t = time.time()
            twin = EvaluatorTwin(cfg.family, assets, [it for it in items if it["holdout"]], raw_conds, model.load_device, cfg.baked_dir)
            engine.log(f"evaluator twin ready ({twin.source}) in {time.time() - t:.0f}s")
        except Exception as e:  # the twin is an instrument, never a reason to fail the run
            engine.log(f"evaluator twin unavailable: {type(e).__name__}: {e}")
            twin = None
    trainer = engine.Trainer(cfg, model, items, raw_conds, cfg.out, budget, twin=twin)
    if cfg.identity_only:
        trainer.save(trainer.lora.state(), trainer.lora.scale, float("nan"), "identity-only")
        engine.log("identity-only artifact written")
        return
    guider = engine.make_guider(model, raw_conds[""])
    lat0 = items[0]["scaled"]
    guider.sample(torch.zeros_like(lat0), lat0, engine.TrainSampler(trainer), engine.evaluator_schedule(),
                  disable_pbar=True, seed=engine.C.EVAL_SAMPLER_SEED)
    engine.log("done")


if __name__ == "__main__":
    main()
