#!/usr/bin/env python
"""Figures + tables for both steering experiments (and the comparison they
are measured against), from saved results only. Safe to run at any time --
sections whose results don't exist yet are skipped.

Reads outputs/experiments/<size>/{steering,url_analysis}/<ckpt label>/ and
writes to outputs/results/<size>/steering/<ckpt label>/{figures,tables}/:

  Experiment 1 (mixtures)
    figures/mixtures_all_pairs_mean        mean acc vs alpha, every pair
    figures/mixture_<a>+<b>_by_task        per-task curves for one pair
    figures/mixtures_strength_mean         mean acc vs steering strength
    tables/mixtures                        every vector x task (csv/md/tex)
    tables/mixture_neighbors               nearest library domains of each vector

  Experiment 2 (task-optimized conditioning)
    figures/task_vectors_by_task           eval acc per task, every run x method
    figures/task_vectors_training          select-acc curves of the optimizers
    figures/transfer_<run>_<method>        cross-task transfer heatmaps
    figures/task_vectors_by_task_no_opt    the same without the simplex / free-optimized vectors
    tables/task_vectors_eval               per-task jobs (csv/md/tex)
    tables/task_vectors_joint              the joint "all" jobs
    tables/task_vectors_delta_vs_unconditioned   paired differences +- stderr
    tables/task_vectors_domains            chosen domains / simplex weights / nearest domains
    tables/domain_search_rankings          top domains per task, MeCo vs URL
    tables/sanity_scorer_vs_lm_eval        steering scorer vs lm-eval numbers

  python scripts/steering/make_steering_report.py
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.comparison.checkpoints import shared_checkpoint
from src.comparison.report import tasks_with_data
from src.config import load_config
from src.steering import report as S
from src.utils import paths
from src.url_analysis.encode import load_library


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comparison-config", default=os.path.join(ROOT, "configs/comparison.yaml"))
    p.add_argument("--tasks-config", default=os.path.join(ROOT, "configs/benchmarks/tasks.yaml"))
    p.add_argument("--mixtures-config", default=os.path.join(ROOT, "configs/steering/mixtures.yaml"))
    p.add_argument("--task-vectors-config", default=os.path.join(ROOT, "configs/steering/task_vectors.yaml"))
    args = p.parse_args()

    cmp_cfg = load_config(args.comparison_config)
    pin = shared_checkpoint(cmp_cfg, ROOT)
    label = pin["label"]
    size = paths.comparison_size(cmp_cfg)
    tasks = load_config(args.task_vectors_config).get("tasks") or load_config(args.tasks_config)["tasks"]
    base = paths.experiments_dir(size, "steering", label)
    out = paths.results_dir(size, "steering", label)
    fig_dir, tab_dir = os.path.join(out, "figures"), os.path.join(out, "tables")
    lib_path = paths.experiments_dir(size, "url_analysis", label, "library.pt")
    lib = load_library(lib_path) if os.path.exists(lib_path) else None

    # -- experiment 1 ---------------------------------------------------------
    recs, url_recs = S.mixture_records(pin)
    mix_cfg = load_config(args.mixtures_config)
    if any(t.startswith(("mix_", "anchor_", "scale_", "mean_")) for t in url_recs):
        # only the tasks the mixture results cover (they may predate a task-list change)
        mix_tasks = tasks_with_data([r for t, r in url_recs.items()
                                     if t.startswith(("mix_", "anchor_", "scale_", "mean_"))], tasks)
        S.plot_mixtures(pin, recs, url_recs, mix_cfg, mix_tasks, fig_dir)
        S.mixture_table(url_recs, mix_tasks, tab_dir)
        vec_path = os.path.join(base, "mixtures", "vectors.pt")
        if lib is not None and os.path.exists(vec_path):
            S.mixture_neighbors(torch.load(vec_path, weights_only=False), lib, tab_dir)
        print(f"mixtures: {len(url_recs)} URL-run results")
    else:
        print("mixtures: no results yet")

    # -- experiment 2 ---------------------------------------------------------
    tv_dir = os.path.join(base, "task_vectors")
    runs = list(pin["runs"])
    results = S.load_task_results(tv_dir, runs, tasks)
    # leave out tasks no run has a task-vector job for yet
    tasks = [t for t in tasks if any(t in per for per in results.values())]
    if results:
        rows, cells, keys = S.task_vector_rows(results, tasks, runs)
        if rows:
            S.write_table(["Run", "Method"] + tasks, rows, cells, tab_dir, "task_vectors_eval")
            S.plot_task_vectors(pin, rows, cells, keys, tasks, fig_dir)
            S.plot_task_vectors(pin, rows, cells, keys, tasks, fig_dir,
                                methods=("unconditioned", "best_source", "domain_search"),
                                stem="task_vectors_by_task_no_opt", title="Source / domain conditioning")
        jrows, jcells = S.joint_rows(results, tasks, runs)
        if jrows:
            S.write_table(["Run", "Method"] + tasks, jrows, jcells, tab_dir, "task_vectors_joint")
        S.delta_table(results, tasks, runs, tab_dir)
        S.nearest_table(results, tasks, tab_dir)
        S.ranking_table(results, tasks, tab_dir)
        S.plot_histories(results, tasks, fig_dir)
        S.plot_transfer(tv_dir, results, tasks, fig_dir)
        S.sanity_check(pin, recs, results, tasks, tab_dir)
        print("task vectors: " + ", ".join(f"{r}: {len(v)} jobs" for r, v in results.items()))
    else:
        print("task vectors: no results yet")
    print(f"-> {out}/{{figures,tables}}")


if __name__ == "__main__":
    main()
