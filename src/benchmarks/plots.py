"""Benchmark comparison figures (scripts/benchmarks/plot_benchmarks.py):

  i.  conditioning-type comparison, e.g. Baseline vs. MeCo vs. Text
      Conditioning, each at a matched source/conditioning setting.
  ii. source comparison, e.g. Wikipedia vs. Reddit vs. News, all for one
      run's conditioning type.

Both reduce to the same shape: a list of (label, benchmark result record)
pairs (src/benchmarks/results.py's discover_results output) to compare, laid
out per task (`plot_task_breakdown`) plus one macro-averaged summary bar
(`plot_aggregate`). Reuses src/analysis/figures.py's presentation style so
benchmark figures look consistent with the training-curve figures.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.figures import STYLE, save_figure
from src.benchmarks.results import best_metric

_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#555555"]


def _series(labeled_records, tasks=None, prefer=("acc_norm,none", "acc,none")):
    """labeled_records: [(label, record), ...] -> (tasks, {label: [values]}).
    A value is None where that (label, task) combo has no result, so callers
    can tell "missing" apart from "scored zero"."""
    if tasks is None:
        tasks = []
        for _, rec in labeled_records:
            for t in rec["results"]:
                if t not in tasks:
                    tasks.append(t)

    series = {}
    for label, rec in labeled_records:
        values = []
        for t in tasks:
            task_results = rec["results"].get(t)
            value = None if task_results is None else best_metric(task_results, prefer=prefer)[1]
            values.append(value)
        series[label] = values
    return tasks, series


def plot_task_breakdown(labeled_records, out_dir, stem, tasks=None, title=None,
                        prefer=("acc_norm,none", "acc,none")):
    """Grouped bar chart: one group per task, one bar per labeled record.
    Bars for a (label, task) combo with no result are drawn faint at height 0
    rather than skipped, so gaps are visible instead of silently absent."""
    tasks, series = _series(labeled_records, tasks=tasks, prefer=prefer)
    labels = list(series.keys())
    n_groups, n_bars = len(tasks), len(labels)
    width = 0.8 / max(1, n_bars)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(max(7.5, 1.1 * n_groups + 2), 4.8))
        x = np.arange(n_groups)
        for bi, label in enumerate(labels):
            values = series[label]
            offsets = x + (bi - (n_bars - 1) / 2) * width
            heights = [v if v is not None else 0.0 for v in values]
            bars = ax.bar(offsets, heights, width=width * 0.92,
                          color=_PALETTE[bi % len(_PALETTE)], label=label)
            for rect, v in zip(bars, values):
                if v is None:
                    rect.set_alpha(0.15)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=20, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(title or "Benchmark Accuracy by Task")
        ax.legend()
        save_figure(fig, out_dir, stem)


def plot_aggregate(labeled_records, out_dir, stem, tasks=None, title=None,
                   prefer=("acc_norm,none", "acc,none")):
    """Single bar chart: one bar per labeled record, macro-averaged over
    whichever tasks it has results for."""
    tasks, series = _series(labeled_records, tasks=tasks, prefer=prefer)
    labels = list(series.keys())
    means = []
    for label in labels:
        values = [v for v in series[label] if v is not None]
        means.append(sum(values) / len(values) if values else 0.0)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(max(5.0, 1.1 * len(labels) + 1.5), 4.8))
        colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(labels))]
        bars = ax.bar(labels, means, color=colors, width=0.6)
        for rect, m in zip(bars, means):
            ax.annotate(f"{m:.3f}", (rect.get_x() + rect.get_width() / 2, m),
                       textcoords="offset points", xytext=(0, 4), ha="center", fontsize=10)
        ax.set_ylabel("Mean accuracy")
        ax.set_ylim(0, 1)
        ax.set_title(title or "Benchmark Accuracy (macro-average)")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        save_figure(fig, out_dir, stem)
