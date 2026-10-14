#!/usr/bin/env python
"""Regenerate all comparison figures from saved metrics -- no training or eval
rerun needed (README: "All figure generation should be reproducible from saved
metrics without needing to rerun evaluation").

Figures are made separately for each model size, since runs are only
comparable at the same scale: with no arguments, auto-discovers every run under
outputs/checkpoints/ (any run dir containing a metrics.jsonl, named
<experiment>_<model_size> -- see scripts/train.py), except the
document-text-conditioned runs (SELF_TEXT_FAMILIES; add
--include-text-conditioning to keep them), and writes each size's figures to
outputs/results/<size>/training/figures/ (src/utils/paths.py):
  python scripts/make_figures.py
  python scripts/make_figures.py --sizes 3b

Run dirs can still be given explicitly (still split by size), and --out-dir
puts every given run in one set of figures instead:
  python scripts/make_figures.py outputs/checkpoints/baseline_medium \\
      outputs/checkpoints/meco_p0.9_medium --out-dir /tmp/figs
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.analysis.curves import load_run_config
from src.analysis.figures import (
    _run_base,
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
from src.utils import paths


# Document-text-conditioned families (the encoder reads the document itself),
# left out of auto-discovered comparisons by default: their conditioned loss
# isn't comparable to Baseline / MeCo / URL-encoded conditioning. The
# URL-encoded (text_conditioned_url*) and domain-conditioned families stay in.
SELF_TEXT_FAMILIES = ("text_conditioned", "text_conditioned_param_matched", "text_conditioned_cooldown")


def discover_run_dirs(include_text_conditioning=False, sizes=None):
    """Every run under outputs/checkpoints/ that has a metrics.jsonl, minus the
    SELF_TEXT_FAMILIES runs unless include_text_conditioning."""
    return [d for d in paths.discover_runs(sizes=sizes)
            if include_text_conditioning
            or _run_base(paths.run_name(d), load_run_config(d)) not in SELF_TEXT_FAMILIES]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="*",
                    help="Defaults to every run under outputs/checkpoints/ with a metrics.jsonl.")
    p.add_argument("--sizes", nargs="+", default=None, help="Only these model sizes (e.g. medium 3b).")
    p.add_argument("--out-dir", default=None,
                   help="Write every run's figures here together. Default: "
                        "outputs/results/<size>/training/figures/, one set per model size.")
    p.add_argument("--smooth-window", type=int, default=20)
    p.add_argument("--include-text-conditioning", action="store_true",
                   help="Also auto-discover the document-text-conditioned runs (SELF_TEXT_FAMILIES). "
                        "Explicitly listed run dirs are always used as given.")
    return p.parse_args()


def make_figures(run_dirs, out_dir, smooth_window):
    plot_training_loss(run_dirs, out_dir, smooth_window=smooth_window)
    plot_training_ppl(run_dirs, out_dir, smooth_window=smooth_window)
    plot_validation_unconditioned_loss(run_dirs, out_dir)
    plot_validation_unconditioned_ppl(run_dirs, out_dir)

    # Conditioned-vs-unconditioned curve, per run: applies to any run whose
    # experiment family has a conditioned eval column (MeCo, text_conditioned*
    # -- see CONDITIONED_PREFIXES), so just try every run dir and skip the ones
    # that don't (baseline, or a run with no conditioned column logged yet).
    for run_dir in run_dirs:
        for plot_fn, stem in [(plot_conditioning_curve_loss, "conditioning_curve_loss"),
                               (plot_conditioning_curve_ppl, "conditioning_curve_ppl")]:
            try:
                plot_fn(run_dir, out_dir)
            except ValueError as e:
                print(f"skipping {stem} for {run_dir}: {e}")

    for plot_fn, stem in [(plot_validation_conditioned_loss, "validation_conditioned_loss"),
                           (plot_validation_conditioned_ppl, "validation_conditioned_ppl"),
                           (plot_validation_both_loss, "validation_both_loss"),
                           (plot_validation_both_ppl, "validation_both_ppl")]:
        try:
            plot_fn(run_dirs, out_dir)
        except ValueError as e:
            print(f"skipping {stem}: {e}")


def main():
    args = parse_args()
    run_dirs = args.run_dirs or discover_run_dirs(args.include_text_conditioning, args.sizes)
    if not run_dirs:
        sys.exit(f"no run dirs found under {paths.CHECKPOINTS_DIR} (no metrics.jsonl in any subdirectory)")

    groups = {"all": run_dirs} if args.out_dir else paths.group_by_size(run_dirs)
    for size, group in groups.items():
        if args.sizes and size != "all" and size not in args.sizes:
            continue
        out_dir = args.out_dir or paths.results_dir(size, "training", "figures")
        print(f"[{size}] using run dirs: {', '.join(paths.run_name(d) for d in group)}")
        make_figures(group, out_dir, args.smooth_window)
        print(f"[{size}] figures written to {out_dir}")


if __name__ == "__main__":
    main()
