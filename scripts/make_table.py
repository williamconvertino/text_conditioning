#!/usr/bin/env python
"""Regenerate the final-conditioning comparison table from saved
final_conditioning.json files -- no evaluation rerun needed, mirrors
scripts/make_figures.py's "reproducible from saved metrics" design
(README "Final Evaluation Table" / "Presentation-Ready Outputs") but for the
one-shot final-checkpoint conditioning evaluation (scripts/evaluate.py
--mode final) rather than the throughout-training metrics.jsonl curves.

With no arguments, auto-discovers every run dir under outputs/ (any
subdirectory containing a final_conditioning.json, i.e. one that's had
`scripts/evaluate.py --mode final` run against it) and writes
csv/markdown/latex tables plus a per-domain breakdown csv to outputs/tables:
  python scripts/make_table.py

Run dirs and/or the output dir can still be given explicitly:
  python scripts/make_table.py outputs/baseline_medium outputs/meco_medium \\
      outputs/text_conditioned_medium --out-dir outputs/tables
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
DEFAULT_OUT_DIR = os.path.join(OUTPUTS_DIR, "tables")

from src.analysis.tables import write_all_tables


def discover_run_dirs(outputs_dir):
    """Every subdirectory of outputs_dir with a final_conditioning.json, i.e.
    a run that's had `scripts/evaluate.py --mode final` run against it."""
    if not os.path.isdir(outputs_dir):
        return []
    run_dirs = []
    for name in sorted(os.listdir(outputs_dir)):
        run_dir = os.path.join(outputs_dir, name)
        if os.path.isfile(os.path.join(run_dir, "final_conditioning.json")):
            run_dirs.append(run_dir)
    return run_dirs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="*",
                    help="Defaults to every run dir under outputs/ with a final_conditioning.json.")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    run_dirs = args.run_dirs or discover_run_dirs(OUTPUTS_DIR)
    if not run_dirs:
        sys.exit(f"no run dirs found under {OUTPUTS_DIR} (no final_conditioning.json in any "
                 f"subdirectory -- run `scripts/evaluate.py --run-dir <run> --mode final` first)")
    print(f"using run dirs: {', '.join(run_dirs)}")

    _, domain_rows = write_all_tables(run_dirs, args.out_dir)
    print(f"wrote table to {args.out_dir}")
    if domain_rows:
        print(f"wrote per-domain breakdown to {os.path.join(args.out_dir, 'domain_breakdown.csv')}")


if __name__ == "__main__":
    main()
