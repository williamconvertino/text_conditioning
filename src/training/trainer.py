"""Shared training loop for all experiments (baseline / meco / text-conditioned).
Which variant a run trains is entirely determined by its experiment config (see
configs/experiment/*.yaml and src/config.py), not by separate training scripts.

Resume is automatic: rerunning the same command picks up ckpt_last.pt. The data
cursor is derived from the step count, so resumption replays the exact same
stream (src/data/shards.py's TokenizedDocStream).
"""

import math
import os
import time

import torch
from transformers import AutoTokenizer

from src.config import load_config, save_config
from src.data.collator import VARIANTS, train_batches, load_val_batches
from src.data.shards import load_manifest, split_shards, rank_shards, TokenizedDocStream
from src.models.configuration_llama import LlamaConfig
from src.models.modeling_llama import LlamaForCausalLM
from src.models.text_encoder import TextEncoderConfig, TextEncoder
from src.models.conditioning import TextConditioner
from src.models.conditioned_lm import ConditionedLM
from src.training.checkpointing import save_checkpoint, load_checkpoint, save_eval_checkpoint
from src.training.schedules import build_schedule, sample_keep_mask
from src.eval.perplexity import evaluate_loss
from src.utils.distributed import ddp_setup, ddp_cleanup, all_reduce_mean, broadcast_scalar
from src.utils.logging import MetricLogger


def lr_at(step, cfg):
    warmup, max_steps = cfg["warmup_steps"], cfg["max_steps"]
    lr, min_lr = cfg["lr"], cfg["min_lr"]
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    prog = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * prog))


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8, fused=fused)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def build_model(cfg, tokenizer):
    """Construct the ConditionedLM for a resolved experiment config. Shared by
    training (this module) and evaluation (scripts/evaluate.py), so a run's
    architecture is always rebuilt identically from its resolved_config.yaml."""
    lm_cfg = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=cfg["hidden_size"], intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"], num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=cfg["max_position_embeddings"], rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"], initializer_range=cfg["initializer_range"],
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        pad_token_id=tokenizer.pad_token_id, bos_token_id=(tokenizer.bos_token_id or 0),
        eos_token_id=tokenizer.eos_token_id)
    lm = LlamaForCausalLM(lm_cfg)

    variant = cfg["variant"]
    conditioner = None
    if variant == "text":
        enc_cfg_dict = load_config(cfg["text_encoder_config"])
        enc_cfg = TextEncoderConfig(
            vocab_size=len(tokenizer), hidden_size=enc_cfg_dict["hidden_size"],
            intermediate_size=enc_cfg_dict["intermediate_size"],
            num_hidden_layers=enc_cfg_dict["num_hidden_layers"],
            num_attention_heads=enc_cfg_dict["num_attention_heads"],
            num_key_value_heads=enc_cfg_dict["num_key_value_heads"],
            max_position_embeddings=enc_cfg_dict["max_position_embeddings"],
            rms_norm_eps=enc_cfg_dict["rms_norm_eps"], rope_theta=enc_cfg_dict["rope_theta"],
            initializer_range=enc_cfg_dict["initializer_range"], output_dim=cfg["hidden_size"],
            pooling=enc_cfg_dict.get("pooling", "cls"), pad_token_id=tokenizer.pad_token_id)
        encoder = TextEncoder(enc_cfg)
        conditioner = TextConditioner(encoder, cfg["hidden_size"])

    return ConditionedLM(lm, variant, conditioner), lm_cfg


def check_manifest(cfg, tokenizer):
    manifest = load_manifest(cfg["tok_dir"])
    if manifest["tokenizer"] != cfg["tokenizer"]:
        raise SystemExit(
            f"train config tokenizer={cfg['tokenizer']!r} does not match the tokenizer "
            f"{cfg['tok_dir']} was pretokenized with ({manifest['tokenizer']!r}). Point "
            f"--tokenizer at the same model, or re-run src/data/tokenize_fineweb.py "
            f"into a fresh tok_dir.")
    if len(tokenizer) != manifest["vocab_size"]:
        raise SystemExit(f"tokenizer vocab_size {len(tokenizer)} != manifest vocab_size "
                         f"{manifest['vocab_size']} for {cfg['tok_dir']}")


def run(cfg, run_dir):
    rank, world, local_rank, device, is_ddp = ddp_setup()
    is_main = rank == 0
    torch.manual_seed(cfg["seed"] + rank)

    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        save_config(cfg, os.path.join(run_dir, "resolved_config.yaml"))
    logger = MetricLogger(run_dir)
    last_path = os.path.join(run_dir, "ckpt_last.pt")
    best_path = os.path.join(run_dir, "ckpt_best.pt")

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    check_manifest(cfg, tokenizer)

    variant = cfg["variant"]
    assert variant in VARIANTS

    train_stems, val_stems = split_shards(cfg["tok_dir"], cfg["n_val_shards"])
    my_stems = rank_shards(train_stems, rank, world)
    if is_main:
        print(f"[{cfg['name']}] train shards: {len(train_stems)}  val shards: {len(val_stems)}  "
              f"world_size: {world}", flush=True)

    model, _ = build_model(cfg, tokenizer)
    model = model.to(device)
    if is_main:
        n_lm = count_parameters(model.lm)
        n_enc = count_parameters(model.conditioner.encoder) if model.conditioner is not None else 0
        print(f"[{cfg['name']}] backbone params: {n_lm/1e6:.1f}M  encoder params: {n_enc/1e6:.1f}M  "
              f"total: {(n_lm+n_enc)/1e6:.1f}M", flush=True)

    raw_model = model
    if is_ddp:
        ddp_kwargs = {"device_ids": [local_rank]} if device.type == "cuda" else {}
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
        raw_model = model.module

    opt = make_optimizer(raw_model, cfg)

    step, best_val, tokens_seen = 0, float("inf"), 0
    if os.path.exists(last_path) and not cfg.get("no_resume", False):
        step, best_val, tokens_seen = load_checkpoint(last_path, raw_model, opt, map_location=device)
        if is_main:
            print(f"resumed from {last_path} at step {step}", flush=True)

    # Standing, never-overwritten snapshots for evaluation (distinct from
    # ckpt_last.pt above, which is resume-only and gets overwritten on every
    # validation step). One is written every ckpt_save_interval_tokens tokens
    # actually consumed, so the cadence is correct regardless of
    # world_size/batch/block_size -- unlike a fixed step count, which would
    # only hit ~250M tokens for the world_size it was computed against.
    ckpt_token_interval = cfg.get("ckpt_save_interval_tokens", 250_000_000)
    eval_ckpt_dir = os.path.join(run_dir, "checkpoints")
    if is_main:
        os.makedirs(eval_ckpt_dir, exist_ok=True)
    next_ckpt_tokens = ((tokens_seen // ckpt_token_interval) + 1) * ckpt_token_interval

    encoder_max_tokens = cfg.get("encoder_max_tokens", 512)
    skip_docs = step * cfg["local_batch"] * cfg["grad_accum"]
    stream = TokenizedDocStream(cfg["tok_dir"], my_stems, cfg["seed"], skip_docs=skip_docs)

    max_steps = cfg["max_steps"]
    # "text" applies its schedule post-hoc, in embedding space (see the training
    # loop below); "meco"'s domain prefix is ordinary tokens, so its schedule has
    # to be applied at collation time instead (keep_mask_fn below) -- see
    # src/data/collator.py's train_batches/collate_lm_batch docstrings.
    schedule = None
    if variant in ("text", "meco"):
        schedule = build_schedule(cfg.get("conditioning_schedule", {"kind": "constant", "p_condition": 1.0}))

    def meco_keep_mask_fn():
        p = schedule(step, max_steps)
        return sample_keep_mask(p, cfg["local_batch"], "cpu").tolist()

    batches = train_batches(stream, tokenizer, variant, cfg["block_size"], cfg["meco_max_tokens"],
                            encoder_max_tokens, cfg["local_batch"],
                            keep_mask_fn=meco_keep_mask_fn if variant == "meco" else None)
    val_batches = load_val_batches(cfg["tok_dir"], val_stems, tokenizer, variant, cfg["block_size"],
                                   cfg["meco_max_tokens"], encoder_max_tokens, cfg["val_docs"],
                                   cfg["val_batch_size"])
    # meco also needs an "unconditioned" (no domain prefix) val set to measure
    # what its backbone does without metadata, using the same trained weights.
    # Always the two extremes (fully conditioned / fully unconditioned),
    # regardless of the training schedule's current p_condition -- same
    # convention "text" uses (keep_mask_value True/False below).
    val_batches_uncond = None
    if variant == "meco":
        val_batches_uncond = load_val_batches(cfg["tok_dir"], val_stems, tokenizer, "none",
                                              cfg["block_size"], cfg["meco_max_tokens"],
                                              encoder_max_tokens, cfg["val_docs"], cfg["val_batch_size"])

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[cfg.get("dtype", "bfloat16")]
    if device.type == "cuda":
        autocast = torch.autocast("cuda", dtype=dtype)
    else:
        autocast = torch.autocast("cpu", dtype=torch.bfloat16, enabled=(dtype == torch.bfloat16))

    model.train()
    t0 = time.time()
    while step < max_steps:
        lr = lr_at(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_accum, tok_accum = 0.0, 0
        for micro in range(cfg["grad_accum"]):
            batch = next(batches)
            keep_mask = None
            if variant == "text":
                p = schedule(step, max_steps)
                keep_mask = sample_keep_mask(p, batch["input_ids"].shape[0], device)
            if is_ddp:
                model.require_backward_grad_sync = (micro == cfg["grad_accum"] - 1)
            with autocast:
                loss, n_tok, _ = model(batch, keep_mask=keep_mask)
            (loss / cfg["grad_accum"]).backward()
            loss_accum += loss.item()
            tok_accum += n_tok.item()

        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        step += 1
        tokens_seen += tok_accum * world

        if is_main and tokens_seen >= next_ckpt_tokens:
            snap_path = os.path.join(
                eval_ckpt_dir, f"ckpt_step{step:07d}_tok{tokens_seen/1e6:.0f}M.pt")
            save_eval_checkpoint(snap_path, raw_model, step, best_val, tokens_seen, cfg)
            print(f"[{cfg['name']}] wrote eval checkpoint {snap_path}", flush=True)
            while tokens_seen >= next_ckpt_tokens:
                next_ckpt_tokens += ckpt_token_interval

        if is_main and step % cfg["log_interval"] == 0:
            dt = time.time() - t0
            t0 = time.time()
            rec = {"step": step, "split": "train", "loss": loss_accum / cfg["grad_accum"],
                   "lr": lr, "tokens": tokens_seen, "grad_norm": float(gnorm),
                   "sec_per_step": dt / cfg["log_interval"]}
            if variant in ("text", "meco"):
                rec["p_condition"] = schedule(step, max_steps)
            logger.log(rec)
            print(f"step {step:>6}  loss {rec['loss']:.4f}  lr {lr:.2e}  "
                  f"tok {tokens_seen/1e9:.3f}B  |g| {float(gnorm):.2f}  "
                  f"{rec['sec_per_step']:.2f}s/it", flush=True)

        if step % cfg["val_interval"] == 0 or step == max_steps:
            if variant == "text":
                val_loss, val_ppl = evaluate_loss(raw_model, val_batches, keep_mask_value=True)
                uncond_loss, uncond_ppl = evaluate_loss(raw_model, val_batches, keep_mask_value=False)
            elif variant == "meco":
                val_loss, val_ppl = evaluate_loss(raw_model, val_batches, keep_mask_value=None)
                uncond_loss, uncond_ppl = evaluate_loss(raw_model, val_batches_uncond, keep_mask_value=None)
            else:
                val_loss, val_ppl = evaluate_loss(raw_model, val_batches, keep_mask_value=None)
                uncond_loss, uncond_ppl = None, None

            val_loss = all_reduce_mean(val_loss, device, is_ddp)
            rec = {"step": step, "split": "val", "loss": val_loss,
                   "ppl": math.exp(min(val_loss, 20.0)), "tokens": tokens_seen}
            if uncond_loss is not None:
                uncond_loss = all_reduce_mean(uncond_loss, device, is_ddp)
                uncond_ppl = math.exp(min(uncond_loss, 20.0))
                rec["unconditioned_loss"] = uncond_loss
                rec["unconditioned_ppl"] = uncond_ppl
                if variant == "text":
                    rec["self_conditioned_loss"] = val_loss
                    rec["self_conditioned_ppl"] = rec["ppl"]
                elif variant == "meco":
                    rec["meco_conditioned_loss"] = val_loss
                    rec["meco_conditioned_ppl"] = rec["ppl"]

            if is_main:
                logger.log(rec)
                logger.export_csv()
                print(f"  [val] step {step}  loss {val_loss:.4f}  ppl {rec['ppl']:.2f}", flush=True)
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(best_path, raw_model, opt, step, best_val, tokens_seen, cfg)
                # ckpt_last.pt (the resume checkpoint) is saved on every validation
                # step, not on its own interval -- that keeps the two in lockstep so
                # a resumed run never has to re-derive a data cursor past the last
                # validation it actually logged, and bounds lost work on interruption
                # to at most val_interval steps.
                save_checkpoint(last_path, raw_model, opt, step, best_val, tokens_seen, cfg)
            best_val = broadcast_scalar(best_val, device, is_ddp)

    if is_main:
        print(f"[{cfg['name']}] done. best val loss {best_val:.4f}", flush=True)
        logger.export_csv()
    ddp_cleanup(is_ddp)
