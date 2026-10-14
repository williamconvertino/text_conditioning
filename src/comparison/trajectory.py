"""Benchmark accuracy across pretraining (scripts/comparison/make_trajectory.py):
the compared runs at every token milestone they share snapshots for,
unconditioned and steered toward named sources. The per-milestone results
come from slurm/comparison/bench_trajectory.slurm (tok<N>M labels, one per
milestone), plus the main comparison's own shared-checkpoint results, which
use the same label scheme.

Figures:
  trajectory_unconditioned_mean    mean accuracy vs tokens, every run unconditioned
  trajectory_unconditioned_by_task the same, one panel per benchmark
  trajectory_conditioned_mean      mean accuracy vs tokens, the conditioned runs
                                   steered toward each source (+ their unconditioned
                                   curve and the Baseline as references)
  trajectory_conditioned_by_task   the same, one panel per benchmark
"""

import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.figures import STYLE, get_color, save_figure
from src.analysis.tables import write_csv
from src.benchmarks.results import discover_results
from src.comparison.report import PREFER, mean_of, run_label, task_values
from src.utils.paths import model_size

# Validated (dataviz validate_palette.js, all pairs, light surface): one hue
# per source, in this fixed order; the references stay neutral.
SOURCE_COLORS = ["#2a78d6", "#d95926", "#1baf7a", "#eda100", "#8a2f8f"]
SOURCE_MARKERS = ["o", "s", "^", "D", "v"]
UNCOND_COLOR = "#1f1f1f"
TAG_PREFIX = {"meco": "meco_", "url": "text_"}
RUN_STYLES = {"url": "-", "meco": (0, (5, 2))}


def _tokens(rec):
    ck = rec.get("checkpoint") or {}
    if ck.get("tokens"):
        return int(ck["tokens"])
    return int(re.fullmatch(r"tok(\d+)M", rec["ckpt_label"]).group(1)) * 1_000_000


def collect(cmp_cfg, root, tasks):
    """{(run name, tag): [(tokens, {task: (value, stderr) | None})...] sorted by
    tokens}. Results missing any of `tasks` (e.g. from an older task list) are
    skipped."""
    out, skipped = {}, 0
    for name, run_dir in cmp_cfg["runs"].items():
        run_name = os.path.basename(os.path.normpath(run_dir))
        size = model_size(os.path.join(root, run_dir))
        for rec in discover_results(run_names=[run_name], ckpt_label="tok*M", sizes=[size]):
            if not set(tasks) <= set(rec["results"]):
                skipped += 1
                continue
            cells = dict(zip(tasks, task_values(rec, tasks)))
            out.setdefault((name, rec["conditioning_tag"]), []).append((_tokens(rec), cells))
    for v in out.values():
        v.sort(key=lambda p: p[0])
    return out, skipped


def rows(series, tasks):
    """Long-format table rows: run, conditioning, tokens, task, accuracy (+ mean)."""
    out = []
    for (name, tag), pts in sorted(series.items()):
        for tok, cells in pts:
            for t in tasks:
                c = cells[t]
                out.append([name, tag, tok, t, "" if c is None else f"{c[0]:.4f}",
                            "" if c is None or c[1] is None else f"{c[1]:.4f}"])
            m = mean_of([cells[t] for t in tasks])
            out.append([name, tag, tok, "mean", "" if m is None else f"{m:.4f}", ""])
    return out


def _xy(pts, tasks):
    """tokens, accuracy (%) -- the mean over `tasks`, or one task."""
    xs, ys = [], []
    for tok, cells in pts:
        v = mean_of([cells[t] for t in tasks])
        if v is not None:
            xs.append(tok / 1e9)
            ys.append(100 * v)
    return xs, ys


def _lines(cmp_cfg, series, sources):
    """[(key, label, style kwargs)] for the conditioned figures: each
    conditioned run steered toward each source (color = source), its
    unconditioned curve, and the Baseline."""
    out = []
    cond_runs = [n for n in cmp_cfg["runs"] if n in TAG_PREFIX]
    multi = len(cond_runs) > 1
    for name in cond_runs:
        ls = RUN_STYLES.get(name, "-")
        for i, s in enumerate(sources):
            key = (name, TAG_PREFIX[name] + s)
            if key in series:
                label = f"{run_label(name)}: {s}" if multi else s
                out.append((key, label, dict(color=SOURCE_COLORS[i % len(SOURCE_COLORS)], ls=ls,
                                             marker=SOURCE_MARKERS[i % len(SOURCE_MARKERS)])))
        if (name, "unconditioned") in series:
            out.append(((name, "unconditioned"), f"{run_label(name)}: unconditioned" if multi else "unconditioned",
                        dict(color=UNCOND_COLOR, ls=ls, marker="x", lw=2.6)))
    if ("baseline", "unconditioned") in series:
        out.append((("baseline", "unconditioned"), "Baseline (unconditioned)",
                    dict(color=get_color(os.path.basename(os.path.normpath(cmp_cfg["runs"]["baseline"]))),
                         ls=(0, (2, 1.5)), marker="o", mfc="white", mew=1.4)))
    return out


def _plot(ax, series, lines, tasks):
    for key, label, kw in lines:
        xs, ys = _xy(series[key], tasks)
        ax.plot(xs, ys, label=label, ms=6, **{"lw": 2, **kw})
    ax.set_xlabel("Tokens (billions)")
    ax.grid(True, alpha=0.35)


def plot_unconditioned_mean(cmp_cfg, series, tasks, out_dir, title_suffix=""):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        drawn = 0
        for name, run_dir in cmp_cfg["runs"].items():
            pts = series.get((name, "unconditioned"))
            if not pts:
                continue
            xs, ys = _xy(pts, tasks)
            color = get_color(os.path.basename(os.path.normpath(run_dir)))
            ax.plot(xs, ys, color=color, marker="o", ms=6, lw=2, label=run_label(name))
            ax.annotate(run_label(name), (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=10, color="#333333")
            drawn += 1
        if not drawn:
            plt.close(fig)
            return
        ax.set_xlabel("Tokens (billions)")
        ax.grid(True, alpha=0.35)
        ax.set_ylabel(f"Mean accuracy over {len(tasks)} tasks (%)")
        ax.set_title(f"Unconditioned Benchmarks During Pretraining{title_suffix}")
        ax.margins(x=0.12)
        ax.legend(loc="lower right")
        save_figure(fig, out_dir, "trajectory_unconditioned_mean")


def plot_unconditioned_by_task(cmp_cfg, series, tasks, out_dir, title_suffix=""):
    """One panel per benchmark, every run unconditioned."""
    runs = [(n, d) for n, d in cmp_cfg["runs"].items() if series.get((n, "unconditioned"))]
    if not runs:
        return
    ncol = 3
    nrow = int(np.ceil(len(tasks) / ncol))
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow + 0.6), squeeze=False)
        for ax, t in zip(axes.flat, tasks):
            for name, run_dir in runs:
                pts = [(tok / 1e9, cells[t]) for tok, cells in series[(name, "unconditioned")] if cells[t]]
                xs = np.array([x for x, _ in pts])
                ys = np.array([100 * c[0] for _, c in pts])
                color = get_color(os.path.basename(os.path.normpath(run_dir)))
                ax.plot(xs, ys, color=color, marker="o", ms=5, lw=2, label=run_label(name))
            ax.set_title(t, fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_xlabel("Tokens (billions)", fontsize=10)
            ax.grid(True, alpha=0.35)
        for ax in axes.flat[len(tasks):]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
                   bbox_to_anchor=(0.5, 1.0), fontsize=11)
        fig.suptitle(f"Unconditioned Benchmarks During Pretraining, by Task{title_suffix}", y=1.04,
                     fontsize=15, fontweight="bold")
        save_figure(fig, out_dir, "trajectory_unconditioned_by_task")


def plot_conditioned_mean(cmp_cfg, series, tasks, sources, out_dir, title_suffix=""):
    lines = _lines(cmp_cfg, series, sources)
    if not lines:
        return
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        _plot(ax, series, lines, tasks)
        ax.set_ylabel(f"Mean accuracy over {len(tasks)} tasks (%)")
        ax.set_title(f"Conditioned Benchmarks During Pretraining{title_suffix}")
        ax.legend(title="Conditioned on", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
        save_figure(fig, out_dir, "trajectory_conditioned_mean")


def plot_conditioned_by_task(cmp_cfg, series, tasks, sources, out_dir, title_suffix=""):
    lines = _lines(cmp_cfg, series, sources)
    if not lines:
        return
    ncol = 3
    nrow = int(np.ceil(len(tasks) / ncol))
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow + 0.9), squeeze=False)
        for ax, t in zip(axes.flat, tasks):
            _plot(ax, series, lines, [t])
            ax.set_title(t, fontsize=12)
            ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_xlabel("Tokens (billions)", fontsize=10)
        for ax in axes.flat[len(tasks):]:
            ax.axis("off")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 7), frameon=False,
                   bbox_to_anchor=(0.5, 1.0), fontsize=10)
        fig.suptitle(f"Conditioned Benchmarks During Pretraining, by Task{title_suffix}", y=1.04,
                     fontsize=15, fontweight="bold")
        save_figure(fig, out_dir, "trajectory_conditioned_by_task")


def make_all(cmp_cfg, root, tasks, sources, out_dir):
    series, skipped = collect(cmp_cfg, root, tasks)
    suffix = f" ({str(cmp_cfg['model_size']).upper()})" if cmp_cfg.get("model_size") else ""
    write_csv(["run", "conditioning", "tokens", "task", "accuracy", "stderr"], rows(series, tasks),
              os.path.join(out_dir, "tables", "trajectory.csv"))
    figs = os.path.join(out_dir, "figures")
    plot_unconditioned_mean(cmp_cfg, series, tasks, figs, suffix)
    plot_unconditioned_by_task(cmp_cfg, series, tasks, figs, suffix)
    plot_conditioned_mean(cmp_cfg, series, tasks, sources, figs, suffix)
    plot_conditioned_by_task(cmp_cfg, series, tasks, sources, figs, suffix)
    return series, skipped
