#!/usr/bin/env python
"""Build the training-split URL-domain index used by the domain-conditioned
experiments (configs/experiment/text_conditioned_domain_*.yaml, README
"Domain-Conditioned Training", src/data/domain_index.py). Run once per
tok_dir / n_val_shards split, before training any domain-conditioned run:

  python scripts/build_domain_index.py
  sbatch slurm/domain_conditioned/build_domain_index.slurm

The shard split and output location come from an experiment config (the
defaults match both domain-conditioned experiments), so the index always
covers exactly the shards that run will train on.

Smoke test on a couple of shards (written somewhere else, so it can never be
mistaken for the real index -- training refuses an index whose shard list
doesn't match its split):
  python scripts/build_domain_index.py --limit-shards 2 --out-dir /tmp/domain_index_smoke
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_config
from src.data.domain_index import DomainIndex
from src.data.shards import split_shards


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment",
                   default=os.path.join(ROOT, "configs/experiment/text_conditioned_domain_mean.yaml"),
                   help="Experiment config supplying tok_dir, n_val_shards and "
                        "domain_conditioning.index_dir.")
    p.add_argument("--out-dir", default=None, help="Override domain_conditioning.index_dir.")
    p.add_argument("--num-workers", type=int, default=os.cpu_count())
    p.add_argument("--limit-shards", type=int, default=None, help="Index only the first N training shards.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.experiment)
    out_dir = args.out_dir or cfg["domain_conditioning"]["index_dir"]
    train_stems, _ = split_shards(cfg["tok_dir"], cfg["n_val_shards"])
    if args.limit_shards is not None:
        train_stems = train_stems[: args.limit_shards]

    print(f"indexing {len(train_stems)} training shards from {cfg['tok_dir']} "
          f"with {args.num_workers} workers", flush=True)
    t0 = time.time()
    index = DomainIndex.build(cfg["tok_dir"], train_stems, num_workers=args.num_workers, verbose=True)
    index.save(out_dir)

    counts = np.diff(index.domain_offsets)
    print(f"wrote {out_dir} in {time.time() - t0:.0f}s")
    print(f"  docs: {index.n_docs:,}  domains: {index.n_domains:,}  "
          f"no parseable domain: {index.meta['n_unknown']:,}")
    for min_docs in (2, 5, 10, 100):
        print(f"  docs in a domain with >= {min_docs:>3} docs: {100 * index.coverage(min_docs):5.1f}%  "
              f"({int((counts >= min_docs).sum()):,} domains)")


if __name__ == "__main__":
    main()
