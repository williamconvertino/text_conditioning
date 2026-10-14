#!/usr/bin/env python
"""Print the benchmark jobs behind the pretraining-trajectory comparison
(scripts/comparison/make_trajectory.py), one per line as
"<run_dir> <tok milestone> <mode> [<source>]", for slurm/comparison/bench_trajectory.slurm.

Every token milestone each compared run has a snapshot for (src/comparison/checkpoints.py),
benchmarked unconditioned, and, for a conditioned run (MeCo / URL), steered
toward each of --sources (default: every configs/benchmarks/sources.yaml source).
The comparison's own pinned / shared checkpoint is left out: the main
comparison (slurm/scale/bench.slurm, slurm/comparison/bench_comparison.slurm) already
benchmarks it under the same label, and the trajectory figures read it from there.

  python scripts/comparison/trajectory_jobs.py --config configs/comparison_3b.yaml
  python scripts/comparison/trajectory_jobs.py --config configs/comparison_3b.yaml --count
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.comparison.checkpoints import milestones, shared_checkpoint
from src.config import load_config

MODES = {"baseline": None, "meco": "meco", "url": "text"}


def jobs(cmp_cfg, sources, include_shared=False):
    shared = shared_checkpoint(cmp_cfg, ROOT)["label"]
    out = []
    for name, run_dir in cmp_cfg["runs"].items():
        mode = MODES.get(name)
        for label in milestones(os.path.join(ROOT, run_dir)):
            if label == shared and not include_shared:
                continue
            out.append(f"{run_dir} {label} none")
            if mode:
                out += [f"{run_dir} {label} {mode} {s}" for s in sources]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "configs/comparison.yaml"))
    p.add_argument("--sources-config", default=os.path.join(ROOT, "configs/benchmarks/sources.yaml"))
    p.add_argument("--sources", nargs="+", default=None)
    p.add_argument("--include-shared", action="store_true")
    p.add_argument("--count", action="store_true", help="Print only the number of jobs.")
    args = p.parse_args()
    sources = args.sources or list(load_config(args.sources_config)["sources"])
    js = jobs(load_config(args.config), sources, args.include_shared)
    print(len(js) if args.count else "\n".join(js))


if __name__ == "__main__":
    main()
