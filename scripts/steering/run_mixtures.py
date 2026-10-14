#!/usr/bin/env python
"""Steering experiment 1: evaluate combinations of URL vectors on the lm-eval
suite (configs/steering/mixtures.yaml; README "Encoder Analysis & Steering").

Vectors (tags):
  anchor_<a>                 a single anchor
  mix_<a>+<b>_a<alpha>       interpolation between two anchors (alpha = weight on b)
  scale_<a>_s<s>             steering strength around the reference (s = 1 -> anchor_<a>)
  mean_<name>                equal-weight average of several anchors

Every vector's results go to the standard benchmark location,
outputs/experiments/<size>/benchmarks/<url run>/<ckpt label>/<tag>/results.json, so they sit next
to the comparison results; the vectors themselves (and their metadata) go
to outputs/experiments/<size>/steering/<ckpt label>/mixtures/vectors.pt. Existing results are
skipped unless --overwrite, and --shard/--num-shards split the vector list
across array jobs (slurm/steering/mixtures.slurm).

  python scripts/steering/run_mixtures.py --list          # show the vector list only
  python scripts/steering/run_mixtures.py --shard 0 --num-shards 4
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.benchmarks.results import result_path
from src.benchmarks.runner import benchmark_settings, evaluate_and_save
from src.config import load_config
from src.steering.vectors import encode_anchor, interpolate, load_compared_run, scale, text_spec
from src.url_analysis.encode import load_library
from src.utils import paths


def build_vectors(cfg, encoder, lib):
    """-> ordered {tag: (z, meta)}."""
    kind = cfg.get("anchor_vector", "input")
    anchors = {name: encode_anchor(encoder, url, lib, kind) for name, url in cfg["anchors"].items()}
    out = {}
    for name, z in anchors.items():
        out[f"anchor_{name}"] = (z, {"type": "anchor", "anchor": name, "input": cfg["anchors"][name],
                                     "anchor_vector": kind})
    mode = cfg.get("interpolation", "linear")
    for a, b in cfg.get("pairs", []):
        for alpha in cfg.get("alphas", []):
            if alpha in (0, 0.0, 1, 1.0):
                continue            # the anchors themselves
            out[f"mix_{a}+{b}_a{alpha:.2f}"] = (interpolate(anchors[a], anchors[b], alpha, mode),
                                                {"type": "mix", "a": a, "b": b, "alpha": alpha,
                                                 "interpolation": mode})
    sc = cfg.get("scale")
    if sc:
        if sc.get("reference", "mean") == "mean":
            if lib is None:
                raise SystemExit("scale.reference: mean needs the URL-encoder analysis library "
                                 "(run scripts/url_analysis/run_encoder_analysis.py first) -- or use null")
            ref = lib["mean"]
        else:
            ref = encoder.null_vector
        for a in sc["anchors"]:
            for s in sc["strengths"]:
                if s == 1:
                    continue
                out[f"scale_{a}_s{s:.2f}"] = (scale(anchors[a], ref, s),
                                              {"type": "scale", "anchor": a, "strength": s,
                                               "reference": sc.get("reference", "mean")})
    for name, members in (cfg.get("means") or {}).items():
        out[f"mean_{name}"] = (torch.stack([anchors[m] for m in members]).mean(0),
                               {"type": "mean", "name": name, "members": members})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "configs/steering/mixtures.yaml"))
    p.add_argument("--comparison-config", default=os.path.join(ROOT, "configs/comparison.yaml"))
    p.add_argument("--tasks-config", default=os.path.join(ROOT, "configs/benchmarks/tasks.yaml"))
    p.add_argument("--checkpoint", default="shared")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit", type=float, default=None,
                   help="Docs per task (smoke tests; results then go under '<ckpt label>-limit<N>').")
    p.add_argument("--tasks", nargs="+", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    R = load_compared_run(ROOT, args.comparison_config, cfg["run"], args.checkpoint)
    if R["encoder"] is None:
        raise SystemExit(f"{R['run_dir']} is not a URL-encoded run")
    out_dir = paths.experiments_dir(R["size"], "steering", R["ckpt_label"], "mixtures")
    lib_path = paths.experiments_dir(R["size"], "url_analysis", R["ckpt_label"], "library.pt")
    lib = load_library(lib_path) if os.path.exists(lib_path) else None

    vectors = build_vectors(cfg, R["encoder"], lib)
    tags = list(vectors)
    if args.list:
        for t in tags:
            print(t)
        print(f"{len(tags)} vectors")
        return
    if args.shard == 0:
        os.makedirs(out_dir, exist_ok=True)
        torch.save({t: {"z": z, "meta": m} for t, (z, m) in vectors.items()}, os.path.join(out_dir, "vectors.pt"))

    settings = benchmark_settings(load_config(args.tasks_config), tasks=args.tasks, limit=args.limit)
    label = R["ckpt_label"] + (f"-limit{args.limit:g}" if args.limit else "")
    mine = tags[args.shard::args.num_shards]
    print(f"[mixtures] {len(mine)}/{len(tags)} vectors on shard {args.shard}/{args.num_shards} "
          f"@ {R['ckpt_label']}", flush=True)
    for tag in mine:
        if not args.overwrite and os.path.exists(result_path(R["run_dir"], tag, label)):
            print(f"  {tag}: exists, skipping", flush=True)
            continue
        z, meta = vectors[tag]
        meta = {**meta, "norm": float(z.norm())}
        _, path = evaluate_and_save(R["model"], R["tokenizer"], R["cfg"], text_spec(tag, z, meta), settings,
                                    R["run_dir"], ckpt_label=label, checkpoint=R["ckpt_info"],
                                    device=R["device"])
        print(f"  {tag}: wrote {path}", flush=True)


if __name__ == "__main__":
    main()
