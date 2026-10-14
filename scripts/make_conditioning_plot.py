#!/usr/bin/env python
"""Regenerate the final-conditioning comparison plot from saved
final_conditioning.json files -- no evaluation rerun needed, mirrors
scripts/make_figures.py's "reproducible from saved metrics" design but for
the one-shot final-checkpoint conditioning evaluation (scripts/evaluate.py
--mode final: none / self / same_domain / domain_mean / cross_domain) rather
than the throughout-training metrics.jsonl curves.

Writes a grouped bar chart -- one group per conditioning mode, one bar per
model, using each run's "top_domains" (Top-N) scope since that's the one
restricted document set every mode is directly comparable on (README "Final
Evaluation Table").

One figure set per model size: with no arguments, covers every run under
outputs/checkpoints/ with a final_conditioning.json (outputs/experiments/
<size>/eval/<run>/) and writes final_conditioning_{loss,ppl}.{png,pdf} to
outputs/results/<size>/final_eval/figures/:
  python scripts/make_conditioning_plot.py

Run dirs can still be given explicitly (still split by size), and --out-dir
puts every given run in one figure instead:
  python scripts/make_conditioning_plot.py outputs/checkpoints/meco_p0.9_medium \\
      outputs/checkpoints/text_conditioned_medium --out-dir /tmp/figs
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.analysis.figures import plot_final_conditioning_loss, plot_final_conditioning_ppl
from src.utils import paths


def discover_run_dirs(sizes=None):
    """Every run that's had `scripts/evaluate.py --mode final` run against it."""
    return [d for d in paths.discover_runs(sizes=sizes, require=None)
            if os.path.isfile(paths.final_conditioning_path(d))]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="*",
                    help="Defaults to every run with a final_conditioning.json.")
    p.add_argument("--sizes", nargs="+", default=None, help="Only these model sizes (e.g. medium 3b).")
    p.add_argument("--out-dir", default=None,
                   help="Default: outputs/results/<size>/final_eval/figures/, per model size.")
    return p.parse_args()


def main():
    args = parse_args()
    run_dirs = args.run_dirs or discover_run_dirs(args.sizes)
    if not run_dirs:
        sys.exit(f"no runs with a final_conditioning.json under {paths.EXPERIMENTS_DIR} "
                 f"-- run `scripts/evaluate.py --run-dir <run> --mode final` first")

    groups = {"all": run_dirs} if args.out_dir else paths.group_by_size(run_dirs)
    for size, group in groups.items():
        if args.sizes and size != "all" and size not in args.sizes:
            continue
        out_dir = args.out_dir or paths.results_dir(size, "final_eval", "figures")
        print(f"[{size}] using run dirs: {', '.join(paths.run_name(d) for d in group)}")
        plot_final_conditioning_loss(group, out_dir)
        plot_final_conditioning_ppl(group, out_dir)
        print(f"[{size}] figures written to {out_dir}")


if __name__ == "__main__":
    main()
