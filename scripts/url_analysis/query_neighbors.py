#!/usr/bin/env python
"""Nearest library domains for any encoder input, interactively: where does
the URL encoder put a string, even one never seen in training?

  python scripts/url_analysis/query_neighbors.py https://boards.4chan.org/ https://arxiv.org/abs/1706.03762
  python scripts/url_analysis/query_neighbors.py --mix https://en.wikipedia.org/ https://stackoverflow.com/

--mix also reports the neighbors of the inputs' mean vector. Inputs are
URLs for a full-URL encoder (anything with "://" is converted to the run's
metadata field, e.g. its domain for a domain encoder). Uses the domain
library of the analysis dir (default: the shared checkpoint's).
"""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.comparison.checkpoints import shared_checkpoint
from src.config import load_config
from src.url_analysis.analyses import neighbors
from src.url_analysis.encode import load_library, load_url_encoder
from src.utils import paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--k", type=int, default=15)
    p.add_argument("--kind", choices=["centroid", "root"], default="centroid")
    p.add_argument("--mix", action="store_true")
    p.add_argument("--analysis-dir", default=None)
    p.add_argument("--comparison-config", default=os.path.join(ROOT, "configs/comparison.yaml"))
    args = p.parse_args()

    if not args.analysis_dir:
        cmp_cfg = load_config(args.comparison_config)
        d = paths.experiments_dir(paths.comparison_size(cmp_cfg), "url_analysis",
                                  shared_checkpoint(cmp_cfg, ROOT)["label"])
    else:
        d = args.analysis_dir
    lib = load_library(os.path.join(d, "library.pt"))
    prov = lib["provenance"]
    encoder, _, _ = load_url_encoder(paths.resolve_run_dir(prov["run_dir"]), prov["path"])
    texts = [encoder.metadata(x) if "://" in x else x for x in args.inputs]
    z = encoder.encode(texts)
    names = list(args.inputs)
    if args.mix:
        z = torch.cat([z, z.mean(0, keepdim=True)])
        names.append("mean(" + ", ".join(args.inputs) + ")")
    for name, lst in neighbors(z, names, lib, args.k, args.kind).items():
        print(f"\n{name}")
        for i, n in enumerate(lst, 1):
            print(f"  {i:2d}. {n['domain']:40s} {n['cos']:.3f}")


if __name__ == "__main__":
    main()
