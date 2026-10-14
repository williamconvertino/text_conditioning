#!/usr/bin/env python
"""Profile packed training micro-steps of an experiment at a scale (one GPU):
prints the top CUDA / CPU ops and the mean step time, to find overhead.

  python scripts/profile_packed_step.py --experiment configs/experiment/text_conditioned_url_full.yaml --scale 1.5b
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import yaml

from src.config import _deep_merge, load_config
from src.data.packing import PackedBatcher, Prefetcher, pin_batch
from src.data.shards import split_shards
from src.data.tokenizer_utils import load_tokenizer
from src.data.url_conditioning import build_url_source
from src.training.trainer import build_model, make_optimizer, packing_mode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--scale", default="1.5b")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args()
    cfg = _deep_merge(load_config(args.experiment), load_config(os.path.join(ROOT, "configs", "scale", f"{args.scale}.yaml")))
    for pair in args.set:
        k, _, v = pair.partition("=")
        cfg[k] = yaml.safe_load(v)
    tok = load_tokenizer(cfg["tokenizer"])
    dev = torch.device("cuda")
    model, _ = build_model(cfg, tok)
    model.to(dev)
    if cfg.get("compile"):
        model.compile_for_training()
    opt = make_optimizer(model, cfg)
    train, _ = split_shards(cfg["tok_dir"], cfg["n_val_shards"])
    mode = packing_mode(cfg)
    src = build_url_source(cfg, tok)
    lb = cfg["local_batch"]
    packer = PackedBatcher(cfg["tok_dir"], train[:4], tok, mode, cfg["block_size"], rows_per_step=lb * 2,
                           micro_rows=lb, seed=0, url_field=src.field if src else "url",
                           url_max_tokens=src.encoder_max_tokens if src else 0, p_condition_fn=lambda: 0.9)
    batches = Prefetcher(packer, transform=lambda x: ([pin_batch(b) for b in x[0]], x[1]))

    def step():
        opt.zero_grad(set_to_none=True)
        micro, _ = next(batches)
        for b in micro:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _, _ = model(b)
            loss.backward()
        opt.step()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.steps):
        step()
    torch.cuda.synchronize()
    print(f"mean step {(time.time() - t0) / args.steps:.3f}s  ({2 * lb} rows/step)", flush=True)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            step()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))


if __name__ == "__main__":
    main()
