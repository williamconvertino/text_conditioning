#!/usr/bin/env python
"""Show which checkpoint the comparison scripts will use: the furthest one
shared by every run in configs/comparison.yaml (src/comparison/checkpoints.py).
Every script resolves this itself at startup; this is just for checking.

  python scripts/comparison/show_checkpoint.py
  python scripts/comparison/show_checkpoint.py --config configs/comparison_3b.yaml
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.comparison.checkpoints import list_checkpoints, shared_checkpoint
from src.config import load_config
from src.utils import paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "configs/comparison.yaml"))
    args = p.parse_args()
    cfg = load_config(args.config)

    for name, run_dir in cfg["runs"].items():
        cands = list_checkpoints(os.path.join(ROOT, run_dir))
        print(f"{name:10s} {run_dir}: {len(cands)} candidate(s), furthest "
              + (f"step {cands[-1]['step']} ({cands[-1]['path']})" if cands else "none"))
    shared = shared_checkpoint(cfg, ROOT)
    print(f"\nshared checkpoint: {shared['label']} ({shared['rule']})  "
          f"-> outputs/{{experiments,results}}/{paths.comparison_size(cfg)}/")
    for name, info in shared["runs"].items():
        print(f"  {name:10s} step {info['step']:>6}  {info['tokens'] / 1e9:.3f}B tok  {info['path']}")


if __name__ == "__main__":
    main()
