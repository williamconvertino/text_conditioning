#!/usr/bin/env python
"""Table + figures for the shared-checkpoint benchmark comparison
(slurm/comparison/bench_comparison.slurm): Baseline / MeCo / URL-encoded,
unconditioned and steered toward each named source. --config
configs/comparison_<scale>.yaml for another scale.

Writes to outputs/results/<size>/comparison/<ckpt label>/ (src/utils/paths.py):
  tables/benchmark_comparison.{csv,md,tex}
  figures/comparison_unconditioned_by_task.{png,pdf}
  figures/comparison_sources_mean.{png,pdf}
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.comparison.checkpoints import shared_checkpoint
from src.comparison.report import comparison_rows, shared_records, plot_comparison, tasks_with_data, write_table
from src.config import load_config
from src.utils import paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "configs/comparison.yaml"))
    p.add_argument("--tasks-config", default=os.path.join(ROOT, "configs/benchmarks/tasks.yaml"))
    p.add_argument("--sources-config", default=os.path.join(ROOT, "configs/benchmarks/sources.yaml"))
    p.add_argument("--out-dir", default=None, help="Default: outputs/results/<size>/comparison/<ckpt label>")
    args = p.parse_args()

    cmp_cfg = load_config(args.config)
    pin = shared_checkpoint(cmp_cfg, ROOT)
    tasks = load_config(args.tasks_config)["tasks"]
    sources = list(load_config(args.sources_config)["sources"])
    out_dir = args.out_dir or paths.results_dir(paths.comparison_size(cmp_cfg), "comparison", pin["label"])

    records = shared_records(pin)
    if not records:
        raise SystemExit(f"no results under the checkpoint label {pin['label']!r} yet -- "
                         f"run slurm/comparison/bench_comparison.slurm first")
    tasks = tasks_with_data(records.values(), tasks)
    rows, cells, _ = comparison_rows(pin, records, tasks, sources)
    write_table(["Run", "Conditioning"] + tasks, rows, cells, os.path.join(out_dir, "tables"),
                "benchmark_comparison")
    plot_comparison(pin, records, tasks, sources, os.path.join(out_dir, "figures"))
    print(f"{len(rows)} result rows at {pin['label']} -> {out_dir}")


if __name__ == "__main__":
    main()
