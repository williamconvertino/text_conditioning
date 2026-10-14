#!/usr/bin/env python
"""Figures + tables for the fine-tuning experiments (README "Fine-Tuning"),
from saved results only -- safe to run while jobs are still going (curves use
whatever steps exist; tables use finished jobs).

Reads outputs/experiments/<size>/finetune/ and writes to
outputs/results/<size>/finetune/<ckpt label>/{figures,tables}/ (src/utils/paths.py):

  figures/curves_eval            eval acc vs step, every run x method, per task + mean
  figures/curves_select          the same on the select split
  figures/curves_<method>        one method, the runs against each other
  figures/curves_run_<run>       one run, its methods against each other
  figures/curves_joint           the joint (--task all) jobs, if any
  tables/finetune_results        zero-shot and best-on-select eval acc (csv/md/tex)
  tables/finetune_final          the final iterate instead
  tables/finetune_delta          best minus step 0, per method (points)
  tables/finetune_vs_standard    each method minus standard FT, paired by seed
  tables/finetune_best_steps     the step each kept iterate came from
  (+ *_joint tables for the joint jobs)

  python scripts/finetune/make_finetune_report.py
  python scripts/finetune/make_finetune_report.py --comparison-config configs/comparison_1.5b.yaml
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.comparison.checkpoints import shared_checkpoint
from src.comparison.report import run_label
from src.config import load_config
from src.finetune import report as F
from src.utils import paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comparison-config", default=os.path.join(ROOT, "configs/comparison.yaml"))
    p.add_argument("--config", default=os.path.join(ROOT, "configs/finetune/default.yaml"))
    p.add_argument("--tasks-config", default=os.path.join(ROOT, "configs/benchmarks/tasks.yaml"))
    p.add_argument("--label", default=None,
                   help="Checkpoint label the jobs ran at (default: the shared checkpoint's).")
    p.add_argument("--out-root", default=None,
                   help="Fine-tuning job root (default: outputs/experiments/<size>/finetune).")
    p.add_argument("--report-dir", default=None,
                   help="Default: outputs/results/<size>/finetune/<ckpt label>.")
    args = p.parse_args()

    cmp_cfg = load_config(args.comparison_config)
    pin = shared_checkpoint(cmp_cfg, ROOT)
    label = args.label or pin["label"]
    size = paths.comparison_size(cmp_cfg)
    out_root = args.out_root or paths.experiments_dir(size, "finetune")
    cfg = load_config(args.config)
    tasks = cfg.get("tasks") or load_config(args.tasks_config)["tasks"]
    base = args.report_dir or paths.results_dir(size, "finetune", label)
    fig_dir, tab_dir = os.path.join(base, "figures"), os.path.join(base, "tables")

    jobs = F.load_jobs(out_root, pin, label)
    k = F.n_seeds(jobs)
    print(f"{sum(len(v) for v in jobs.values())} fine-tuning jobs ({len(jobs)} run/method/task, "
          f"up to {k} seed(s)) under label {label}")
    if k > 1:
        print("table cells: mean over seeds +- std across seeds; curves: mean, shaded +-1 std")
    found = sorted({m for _, m, _ in jobs})
    methods = [m for m in cfg["methods"] if m in found] + [m for m in found if m not in cfg["methods"]]
    runs = list(pin["runs"])

    F.plot_curves(jobs, pin, tasks, methods, runs, fig_dir, "curves_eval",
                  f"Fine-tuning: eval accuracy vs step ({label})")
    F.plot_curves(jobs, pin, tasks, methods, runs, fig_dir, "curves_select",
                  f"Fine-tuning: select accuracy vs step ({label})", which=0)
    for m in methods:
        F.plot_curves(jobs, pin, tasks, [m], runs, fig_dir, f"curves_{m}",
                      f"{F.method_label(m)}: eval accuracy vs step ({label})")
    for r in runs:
        F.plot_curves(jobs, pin, tasks, methods, [r], fig_dir, f"curves_run_{r}",
                      f"{run_label(r)}: eval accuracy vs step ({label})", color_by="method")
    F.plot_curves(jobs, pin, tasks, methods, runs, fig_dir, "curves_joint",
                  f"Joint fine-tuning on all tasks: eval accuracy vs step ({label})", prefix="all:")

    # single-vector PEFT: one trained vector per run. The baseline's starts at the
    # mean token embedding (peft_1vec); the URL run's sits in its conditioning
    # slot, started at E(condition url) (peft) or at the null vector (peft_null).
    base_1vec = ("baseline", "peft_1vec", "Baseline: 1 vector (mean-embedding init)")
    for init, method, desc in (("url_init", "peft", "URL-encoding init"), ("null_init", "peft_null", "null-vector init")):
        pairs = [base_1vec, ("url", method, f"URL-Encoded: 1 vector ({desc})")]
        for full, suffix in ((False, ""), (True, "_full_step0")):
            F.plot_pair_curves(jobs, pin, tasks, pairs, fig_dir, f"curves_single_vector_{init}{suffix}",
                               f"Single-vector PEFT, {desc}: eval accuracy vs step ({label})", full_step0=full)
            F.plot_pair_curves(jobs, pin, tasks, pairs, fig_dir, f"curves_single_vector_{init}{suffix}_mean",
                               f"Single-vector PEFT, {desc} ({label})", full_step0=full, mean_only=True)

    F.result_tables(jobs, pin, tasks, methods, tab_dir)
    F.result_tables(jobs, pin, tasks, methods, tab_dir, joint=True)
    F.best_steps_table(jobs, pin, tasks, methods, tab_dir)
    print(f"wrote {fig_dir} and {tab_dir}")


if __name__ == "__main__":
    main()
