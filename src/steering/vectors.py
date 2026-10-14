"""Shared plumbing for the steering experiments (scripts/steering/):
loading a compared run at the shared checkpoint, turning anchor specs into
conditioning vectors, and combining vectors.

Vector arithmetic happens in the URL encoder's output space (the vector the
LM sees at its first position). Those vectors are far from the origin (page
vectors have norm ~45, the null vector ~1) and share a large common
direction, so:
  * interpolate()'s "linear" mode (plain convex combination) slightly
    shrinks the norm of dissimilar pairs; "slerp" keeps it on the arc
    between them, norm interpolated linearly.
  * scale() extrapolates around a reference (default: the library mean, the
    "average domain") -- z_ref + s * (z - z_ref) -- so s = 0 is "generic web
    text", s = 1 the anchor itself, s > 1 exaggerates what distinguishes it.
    Scaling around the origin or the null vector would instead mostly
    change the norm, which the LM never saw vary.
"""

import os
import re

import torch
import torch.nn.functional as F

from src.benchmarks.conditioning import ConditioningSpec
from src.comparison.checkpoints import checkpoint_for, shared_checkpoint
from src.config import load_config
from src.data.url_conditioning import build_url_source
from src.training.checkpointing import load_run
from src.url_analysis.encode import UrlEncoder
from src.utils.paths import model_size


def slug(s, max_len=40):
    return re.sub(r"[^a-z0-9.]+", "-", str(s).lower()).strip("-")[:max_len]


def load_compared_run(root, comparison_config, run, checkpoint="shared"):
    """run: a key of configs/comparison.yaml (baseline|meco|url) or a run dir.
    Returns dict(model, tokenizer, cfg, run_dir, run_key, size, ckpt_label, ckpt_info, encoder)
    -- size is the run's model_size, which picks its outputs/experiments/<size>/ dir
    -- encoder is a UrlEncoder sharing the model's conditioner for a
    URL-encoded run, else None."""
    cmp_cfg = load_config(comparison_config)
    run_dir = cmp_cfg["runs"].get(run, run)
    run_key = next((k for k, v in cmp_cfg["runs"].items()
                    if os.path.basename(os.path.normpath(v)) == os.path.basename(os.path.normpath(run_dir))),
                   os.path.basename(os.path.normpath(run_dir)))
    if checkpoint == "shared":
        pin = shared_checkpoint(cmp_cfg, root)
        ckpt, info = checkpoint_for(pin, run_dir)
        label = pin["label"]
    else:
        ckpt, info, label = checkpoint, {"path": checkpoint}, os.path.splitext(os.path.basename(checkpoint))[0]
    abs_run_dir = os.path.join(root, run_dir)
    model, tokenizer, cfg = load_run(abs_run_dir, ckpt)
    url_source = build_url_source(cfg, tokenizer)
    device = next(model.parameters()).device
    encoder = UrlEncoder(model.conditioner, tokenizer, url_source, device) if url_source else None
    return {"model": model, "tokenizer": tokenizer, "cfg": cfg, "run_dir": abs_run_dir, "run_key": run_key,
            "size": model_size(abs_run_dir), "ckpt_label": label, "ckpt_info": info, "encoder": encoder, "device": device}


def encode_anchor(encoder, value, lib=None, kind="input"):
    """value: a URL (or, for a domain encoder, a domain string).
    kind "input": E(value) directly -- for a source's root URL this is exactly
      the vector run_benchmarks.py --mode text --source uses.
    kind "centroid": the library centroid of value's domain (mean over its
      sampled pages; needs the analysis library)."""
    if kind == "input":
        text = encoder.metadata(value) if "://" in value else value
        return encoder.encode([text])[0]
    if kind == "centroid":
        from src.data.metadata import url_to_domain
        domain = url_to_domain(value)
        if lib is None or domain not in lib["domains"]:
            raise KeyError(f"{domain!r} is not in the domain library; use anchor_vector: input")
        return lib["centroid"][lib["domains"].index(domain)]
    raise ValueError(f"anchor kind must be input|centroid, got {kind!r}")


def interpolate(za, zb, alpha, mode="linear"):
    """alpha = 0 -> za, 1 -> zb."""
    if mode == "linear":
        return (1 - alpha) * za + alpha * zb
    if mode == "slerp":
        na, nb = za.norm(), zb.norm()
        ua, ub = za / na, zb / nb
        omega = torch.arccos(torch.clamp((ua * ub).sum(), -1, 1))
        if omega < 1e-4:
            u = ua
        else:
            u = (torch.sin((1 - alpha) * omega) * ua + torch.sin(alpha * omega) * ub) / torch.sin(omega)
        return u * ((1 - alpha) * na + alpha * nb)
    raise ValueError(f"interpolation must be linear|slerp, got {mode!r}")


def scale(z, ref, s):
    return ref + s * (z - ref)


def text_spec(tag, z, meta):
    return ConditioningSpec("text", tag, z=z.float(), meta=meta)


def centered_cos(a, b, mean):
    return float(F.cosine_similarity((a - mean).float(), (b - mean).float(), dim=0))
