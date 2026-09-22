"""Conditioning-probability schedules: p_condition(step, max_steps) -> the
probability that a training example keeps its conditioning signal instead of
training unconditioned. Used by both the "text" variant (real encoded vector
vs. the null vector, applied post-hoc in src/models/conditioned_lm.py) and the
"meco" variant (the domain-prefix tokens vs. none, applied at collation time
in src/data/collator.py's collate_lm_batch/train_batches) -- see README
"Unconditioned Mode" and "Experiment E: Conditioning Schedule".
"""

import math

import torch


def constant_schedule(p_condition):
    return lambda step, max_steps: p_condition


def linear_cooldown_schedule(p_start, p_end, cooldown_start_frac, cooldown_end_frac):
    def f(step, max_steps):
        start = cooldown_start_frac * max_steps
        end = cooldown_end_frac * max_steps
        if step <= start:
            return p_start
        if step >= end:
            return p_end
        frac = (step - start) / max(1.0, end - start)
        return p_start + (p_end - p_start) * frac
    return f


def cosine_cooldown_schedule(p_start, p_end, cooldown_start_frac, cooldown_end_frac):
    def f(step, max_steps):
        start = cooldown_start_frac * max_steps
        end = cooldown_end_frac * max_steps
        if step <= start:
            return p_start
        if step >= end:
            return p_end
        frac = (step - start) / max(1.0, end - start)
        return p_end + 0.5 * (p_start - p_end) * (1 + math.cos(math.pi * frac))
    return f


def step_cooldown_schedule(p_start, p_end, step_frac):
    def f(step, max_steps):
        return p_start if step < step_frac * max_steps else p_end
    return f


def build_schedule(cfg):
    """cfg: the `conditioning_schedule` dict from an experiment yaml, e.g.
    {"kind": "constant", "p_condition": 0.9} or
    {"kind": "linear_cooldown", "p_start": 1.0, "p_end": 0.0,
     "cooldown_start_frac": 0.7, "cooldown_end_frac": 1.0}."""
    kind = cfg.get("kind", "constant")
    if kind == "constant":
        return constant_schedule(cfg.get("p_condition", 1.0))
    if kind == "linear_cooldown":
        return linear_cooldown_schedule(cfg["p_start"], cfg["p_end"],
                                        cfg["cooldown_start_frac"], cfg["cooldown_end_frac"])
    if kind == "cosine_cooldown":
        return cosine_cooldown_schedule(cfg["p_start"], cfg["p_end"],
                                        cfg["cooldown_start_frac"], cfg["cooldown_end_frac"])
    if kind == "step_cooldown":
        return step_cooldown_schedule(cfg["p_start"], cfg["p_end"], cfg["step_frac"])
    raise ValueError(f"unknown conditioning_schedule.kind: {kind!r}")


def sample_keep_mask(p_condition, batch_size, device):
    return torch.rand(batch_size, device=device) < p_condition


def schedule_tag(cfg):
    """Short filesystem-safe tag describing a conditioning_schedule dict, e.g.
    {"kind": "constant", "p_condition": 0.7} -> "p0.7". Used by
    scripts/train.py to fold a --p-condition/--conditioning-schedule override
    into the run dir name, so a p_condition sweep doesn't have every run
    resume/overwrite the same checkpoint directory."""
    kind = cfg.get("kind", "constant")

    def fmt(x):
        return f"{x:g}"

    if kind == "constant":
        return f"p{fmt(cfg.get('p_condition', 1.0))}"
    if kind in ("linear_cooldown", "cosine_cooldown"):
        shape = "lin" if kind == "linear_cooldown" else "cos"
        return f"{shape}{fmt(cfg['p_start'])}to{fmt(cfg['p_end'])}"
    if kind == "step_cooldown":
        return f"step{fmt(cfg['p_start'])}to{fmt(cfg['p_end'])}"
    return kind
