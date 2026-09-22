"""Standard validation cross-entropy / perplexity. Used both frequently during
training (src/training/trainer.py, every val_interval steps) and by the
lightweight "quick" mode of scripts/evaluate.py.
"""

import math

import torch


def evaluate_loss(model, batches, keep_mask_value=None):
    """model: a ConditionedLM (unwrapped, not DDP). keep_mask_value: None (the
    variant's natural conditioning), True (force self-conditioning), False
    (force the null/unconditioned vector) -- only meaningful for
    variant == "text"; ignored otherwise.

    Returns (loss, perplexity).
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            keep_mask = None
            if model.variant == "text" and keep_mask_value is not None:
                B = batch["input_ids"].shape[0]
                keep_mask = torch.full((B,), keep_mask_value, dtype=torch.bool, device=device)
            loss, n_tok, _ = model(batch, keep_mask=keep_mask)
            tot_loss += loss.item() * n_tok.item()
            tot_tok += n_tok.item()
    if was_training:
        model.train()
    loss = tot_loss / max(1, tot_tok)
    return loss, math.exp(min(loss, 20.0))
