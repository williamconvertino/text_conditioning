#!/usr/bin/env python
"""Convenience single-process wrapper around the two-step data pipeline
(src/data/download_fineweb.py, src/data/tokenize_fineweb.py). For the real
100BT run, use the SLURM scripts instead (slurm/download_fineweb.slurm,
slurm/tokenize_fineweb.slurm), which parallelize tokenization across an array;
this script is for small local/smoke-test runs (e.g. --limit-shards).

Usage:
  python scripts/prepare_data.py --sample 100BT --raw-dir data/fineweb_raw \\
      --tok-dir data/fineweb_tokenized/100BT --limit-shards 2
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data import download_fineweb, tokenize_fineweb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", default="100BT", choices=["10BT", "100BT", "350BT"])
    p.add_argument("--raw-dir", default="data/fineweb_raw")
    p.add_argument("--tok-dir", default="data/fineweb_tokenized/100BT")
    p.add_argument("--tokenizer", default="meta-llama/Llama-3.2-3B")
    p.add_argument("--max-doc-tokens", type=int, default=4096)
    p.add_argument("--limit-shards", type=int, default=0, help="Smoke-test: only tokenize N shards.")
    p.add_argument("--skip-download", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not args.skip_download:
        download_fineweb.download(repo="HuggingFaceFW/fineweb", sample=args.sample, out_dir=args.raw_dir)

    tokenize_fineweb.tokenize_all(
        input_dir=os.path.join(args.raw_dir, "sample", args.sample),
        output_dir=args.tok_dir, tokenizer_name=args.tokenizer,
        max_doc_tokens=args.max_doc_tokens, num_workers=1, worker_id=0,
        limit_shards=args.limit_shards)


if __name__ == "__main__":
    main()
