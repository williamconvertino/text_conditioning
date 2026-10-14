#!/usr/bin/env python
"""Regenerate comparison tables from saved results -- no evaluation/benchmark
rerun needed, mirrors scripts/make_figures.py's "reproducible from saved
metrics" design (README "Final Evaluation Table" / "Presentation-Ready
Outputs"). Writes two independent tables per model size, each skipped (not an
error) if its source data isn't there yet:

  final_conditioning.{csv,md,tex} (+ domain_breakdown.csv) -- the one-shot
    final-checkpoint perplexity conditioning eval (scripts/evaluate.py --mode
    final), from each run's outputs/experiments/<size>/eval/<run>/
    final_conditioning.json. -> outputs/results/<size>/final_eval/tables/

  benchmark_results.{csv,md,tex} -- lm-eval-harness downstream-task results
    (scripts/benchmarks/run_benchmarks.py), from every
    outputs/experiments/<size>/benchmarks/<run>/<tag>/results.json
    (src/benchmarks/results.py). Unlike the final-conditioning table, this
    isn't restricted to runs with a final_conditioning.json when
    auto-discovering -- the two come from separate scripts (README
    "Downstream Benchmarking"). -> outputs/results/<size>/benchmarks/tables/

With no arguments, covers every run under outputs/checkpoints/:
  python scripts/make_table.py
  python scripts/make_table.py --sizes medium

Run dirs can still be given explicitly, which also restricts the benchmark
table to those runs' results; --out-dir writes both tables for every given
run to one directory instead:
  python scripts/make_table.py outputs/checkpoints/baseline_medium \\
      outputs/checkpoints/meco_p0.9_medium --out-dir /tmp/tables
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.analysis.tables import write_all_tables
from src.benchmarks.results import discover_results
from src.benchmarks.tables import write_all_benchmark_tables
from src.utils import paths


def has_final_conditioning(run_dir):
    return os.path.isfile(paths.final_conditioning_path(run_dir))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="*",
                    help="Defaults to every run under outputs/checkpoints/ "
                         "(and restricts the benchmark table to these runs, if given).")
    p.add_argument("--sizes", nargs="+", default=None, help="Only these model sizes (e.g. medium 3b).")
    p.add_argument("--out-dir", default=None,
                   help="Write both tables here, for every given run together. Default: "
                        "outputs/results/<size>/{final_eval,benchmarks}/tables/, per model size.")
    return p.parse_args()


def write_size(size, run_dirs, explicit, out_dir=None):
    """Both tables for one model size (size None: every given run, into out_dir).
    Returns whether anything was written."""
    wrote_any = False
    final_dirs = [d for d in run_dirs if has_final_conditioning(d)]
    if final_dirs:
        table_out = out_dir or paths.results_dir(size, "final_eval", "tables")
        _, domain_rows = write_all_tables(final_dirs, table_out)
        print(f"[{size or 'all'}] wrote final-conditioning table "
              f"({', '.join(paths.run_name(d) for d in final_dirs)}) to {table_out}")
        if domain_rows:
            print(f"[{size or 'all'}] wrote per-domain breakdown to {os.path.join(table_out, 'domain_breakdown.csv')}")
        wrote_any = True

    run_names = {paths.run_name(d) for d in run_dirs} if explicit else None
    records = discover_results(run_names=run_names, sizes=[size] if size else None)
    if records:
        table_out = out_dir or paths.results_dir(size, "benchmarks", "tables")
        write_all_benchmark_tables(records, table_out)
        print(f"[{size or 'all'}] wrote benchmark-results table ({len(records)} results) to {table_out}")
        wrote_any = True
    return wrote_any


def main():
    args = parse_args()
    explicit = bool(args.run_dirs)
    run_dirs = args.run_dirs or paths.discover_runs(sizes=args.sizes, require=None)

    if args.out_dir:
        wrote_any = write_size(None, run_dirs, explicit, args.out_dir)
    else:
        wrote_any = False
        for size, group in paths.group_by_size(run_dirs).items():
            if args.sizes and size not in args.sizes:
                continue
            wrote_any |= write_size(size, group, explicit)

    if not wrote_any:
        sys.exit(f"nothing to write: no final_conditioning.json or benchmark results found under "
                 f"{paths.EXPERIMENTS_DIR} (run `scripts/evaluate.py --run-dir <run> --mode final` "
                 f"or scripts/benchmarks/run_benchmarks.py first)")


if __name__ == "__main__":
    main()
