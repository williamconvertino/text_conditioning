#!/usr/bin/env python
"""Regenerate all comparison figures from saved metrics -- no training or eval
rerun needed (README: "All figure generation should be reproducible from saved
metrics without needing to rerun evaluation").

With no arguments, auto-discovers every run dir under outputs/ (any
subdirectory containing a metrics.jsonl, named <experiment>_<model_size> --
see scripts/train.py) and writes figures to outputs/figures:
  python scripts/make_figures.py

Run dirs and/or the output dir can still be given explicitly:
  python scripts/make_figures.py outputs/baseline_medium outputs/meco_medium \\
      outputs/text_conditioned_medium --out-dir outputs/figures
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
DEFAULT_OUT_DIR = os.path.join(OUTPUTS_DIR, "figures")

from src.analysis.figures import (
    plot_training_loss,
    plot_training_ppl,
    plot_validation_unconditioned_loss,
    plot_validation_unconditioned_ppl,
    plot_validation_conditioned_loss,
    plot_validation_conditioned_ppl,
    plot_conditioning_curve_loss,
    plot_conditioning_curve_ppl,
    plot_validation_both_loss,
    plot_validation_both_ppl,
)


def discover_run_dirs(outputs_dir):
    """Every subdirectory of outputs_dir that has a metrics.jsonl, i.e. an
    actual completed/in-progress training run rather than e.g. outputs/figures."""
    if not os.path.isdir(outputs_dir):
        return []
    run_dirs = []
    for name in sorted(os.listdir(outputs_dir)):
        run_dir = os.path.join(outputs_dir, name)
        if os.path.isfile(os.path.join(run_dir, "metrics.jsonl")):
            run_dirs.append(run_dir)
    return run_dirs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="*",
                    help="Defaults to every run dir under outputs/ with a metrics.jsonl.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--smooth-window", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    run_dirs = args.run_dirs or discover_run_dirs(OUTPUTS_DIR)
    if not run_dirs:
        sys.exit(f"no run dirs found under {OUTPUTS_DIR} (no metrics.jsonl in any subdirectory)")
    print(f"using run dirs: {', '.join(run_dirs)}")

    plot_training_loss(run_dirs, args.out_dir, smooth_window=args.smooth_window)
    plot_training_ppl(run_dirs, args.out_dir, smooth_window=args.smooth_window)
    plot_validation_unconditioned_loss(run_dirs, args.out_dir)
    plot_validation_unconditioned_ppl(run_dirs, args.out_dir)

    # Conditioned-vs-unconditioned curve, per run: applies to any run whose
    # experiment family has a conditioned eval column (MeCo, text_conditioned*
    # -- see CONDITIONED_PREFIXES), so just try every run dir and skip the ones
    # that don't (baseline, or a run with no conditioned column logged yet).
    for run_dir in run_dirs:
        for plot_fn, stem in [(plot_conditioning_curve_loss, "conditioning_curve_loss"),
                               (plot_conditioning_curve_ppl, "conditioning_curve_ppl")]:
            try:
                plot_fn(run_dir, args.out_dir)
            except ValueError as e:
                print(f"skipping {stem} for {run_dir}: {e}")

    for plot_fn, stem in [(plot_validation_conditioned_loss, "validation_conditioned_loss"),
                           (plot_validation_conditioned_ppl, "validation_conditioned_ppl"),
                           (plot_validation_both_loss, "validation_both_loss"),
                           (plot_validation_both_ppl, "validation_both_ppl")]:
        try:
            plot_fn(run_dirs, args.out_dir)
        except ValueError as e:
            print(f"skipping {stem}: {e}")

    print(f"figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
