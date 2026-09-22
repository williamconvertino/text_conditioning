"""Checkpoint save/load for a ConditionedLM run (src/training/trainer.py)."""

import os

import torch
from transformers import AutoTokenizer


def save_checkpoint(path, model, optimizer, step, best_val, tokens_seen, run_cfg):
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val": best_val,
        "tokens_seen": tokens_seen,
        "run_cfg": run_cfg,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"], ckpt["best_val"], ckpt["tokens_seen"]


def save_eval_checkpoint(path, model, step, best_val, tokens_seen, run_cfg):
    """Save a standing, never-overwritten snapshot for evaluation (src/training/
    trainer.py's periodic ckpt_save_interval_tokens checkpoints). No optimizer
    state -- these are for scripts/evaluate.py, not for resuming training, so
    skipping it keeps snapshot storage to roughly the model size instead of the
    ~3x an AdamW optimizer state would add."""
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "step": step,
        "best_val": best_val,
        "tokens_seen": tokens_seen,
        "run_cfg": run_cfg,
    }, tmp)
    os.replace(tmp, path)


def load_run(run_dir, checkpoint="ckpt_best.pt", device=None):
    """Rebuild a trained ConditionedLM + tokenizer from a run directory written
    by src/training/trainer.py (resolved_config.yaml + a checkpoint file).
    Shared by scripts/evaluate.py and scripts/benchmarks/run_benchmarks.py so a
    run's architecture and weights are always reconstructed identically,
    regardless of which evaluation axis (perplexity vs. lm-eval-harness
    benchmarks) is being run against it."""
    # deferred import: src/training/trainer.py imports this module, so a
    # top-level import here would be circular.
    from src.config import load_config
    from src.training.trainer import build_model

    cfg = load_config(os.path.join(run_dir, "resolved_config.yaml"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, _ = build_model(cfg, tokenizer)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    ckpt = torch.load(os.path.join(run_dir, checkpoint), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer, cfg
