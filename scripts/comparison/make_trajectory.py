#!/usr/bin/env python
"""Figures for the compared runs' benchmark accuracy across pretraining
(src/comparison/trajectory.py; benchmarks from
slurm/comparison/submit_trajectory.sh plus the main comparison's own
shared-checkpoint results). --config configs/comparison_<scale>.yaml for
another scale.

Writes to outputs/results/<size>/comparison/trajectory/:
  tables/trajectory.csv
  figures/trajectory_unconditioned_mean.{png,pdf}
  figures/trajectory_unconditioned_by_task.{png,pdf}
  figures/trajectory_conditioned_mean.{png,pdf}
  figures/trajectory_conditioned_by_task.{png,pdf}
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.comparison.trajectory import make_all
from src.config import load_config
from src.utils import paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "configs/comparison.yaml"))
    p.add_argument("--tasks-config", default=os.path.join(ROOT, "configs/benchmarks/tasks.yaml"))
    p.add_argument("--sources-config", default=os.path.join(ROOT, "configs/benchmarks/sources.yaml"))
    p.add_argument("--sources", nargs="+", default=None, help="Default: every sources.yaml source.")
    p.add_argument("--out-dir", default=None, help="Default: outputs/results/<size>/comparison/trajectory")
    args = p.parse_args()

    cmp_cfg = load_config(args.config)
    tasks = load_config(args.tasks_config)["tasks"]
    sources = args.sources or list(load_config(args.sources_config)["sources"])
    out_dir = args.out_dir or paths.results_dir(paths.comparison_size(cmp_cfg), "comparison", "trajectory")
    series, skipped = make_all(cmp_cfg, ROOT, tasks, sources, out_dir)
    for (name, tag), pts in sorted(series.items()):
        print(f"  {name:9s} {tag:22s} {len(pts)} milestones: " + ", ".join(f"{t / 1e9:g}B" for t, _ in pts))
    if skipped:
        print(f"skipped {skipped} results that lack some of {tasks} (older task list)")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
