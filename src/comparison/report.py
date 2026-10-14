"""Tables and figures for the shared-checkpoint benchmark comparison
(slurm/comparison/bench_comparison.slurm -> scripts/comparison/make_comparison.py):
Baseline vs. MeCo vs. URL-encoded conditioning, unconditioned and steered
toward each named source.

Also home to the small helpers the steering reports share (record lookup by
run/tag, per-task accuracy vectors, run colors/labels), so every steering
figure labels and colors the three runs the same way.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.figures import STYLE, get_color, save_figure
from src.analysis.tables import write_csv, write_latex, write_markdown
from src.benchmarks.results import best_metric_with_stderr, discover_results

RUN_LABELS = {"baseline": "Baseline", "meco": "MeCo", "url": "URL-Encoded", "url_strong": "URL-Encoded (Strong)"}
PREFER = ("acc_norm,none", "acc,none")


def run_label(name):
    return RUN_LABELS.get(name, name)


def run_color(pin, name):
    return get_color(os.path.basename(os.path.normpath(pin["runs"][name]["run_dir"])))


def latex_escape(s):
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def shared_records(pin, tags=None):
    """{(run name, conditioning tag): record} for every result saved under the
    shared checkpoint's label for the compared runs."""
    out = {}
    for name, info in pin["runs"].items():
        run_name = os.path.basename(os.path.normpath(info["run_dir"]))
        for rec in discover_results(run_names=[run_name], tags=tags, ckpt_label=pin["label"]):
            out[(name, rec["conditioning_tag"])] = rec
    return out


def task_values(rec, tasks):
    """[(value, stderr) or None per task] -- headline metric as in the benchmark table."""
    cells = []
    for t in tasks:
        tr = rec["results"].get(t)
        if tr is None:
            cells.append(None)
            continue
        _, v, se = best_metric_with_stderr(tr, prefer=PREFER)
        cells.append(None if v is None else (v, se))
    return cells


def tasks_with_data(records, tasks):
    """The tasks (in `tasks` order) at least one of `records` has a result for,
    so figures and tables leave out tasks nothing has been run on yet."""
    records = list(records)
    return [t for t in tasks if any(t in r["results"] for r in records)]


def common_task_idx(cells):
    """Indices of the tasks every row of `cells` has a value for -- what a
    figure's "Mean" panel averages over, so every bar averages the same tasks."""
    if not cells:
        return []
    return [j for j in range(len(cells[0])) if all(c[j] is not None for c in cells)]


def mean_label(n_common, n_tasks):
    return "Mean" if n_common == n_tasks else f"Mean ({n_common} of {n_tasks} tasks)"


def mean_of(cells):
    vals = [c[0] for c in cells if c is not None]
    return sum(vals) / len(vals) if vals else None


def fmt_cell(cell, pm="±"):
    if cell is None:
        return "—"
    v, se = cell
    return f"{v * 100:.1f}" + (f" {pm} {se * 100:.1f}" if se is not None else "")


def write_table(header, rows, cells, out_dir, stem):
    """rows: [[label cols...]], cells: matching [[(v, se) | None, ...]] ->
    csv/md/tex with percentage cells and a macro Mean column."""
    plain, latex = [], []
    for r, cs in zip(rows, cells):
        m = mean_of(cs)
        mean = "—" if m is None else f"{m * 100:.1f}"
        plain.append(list(r) + [fmt_cell(c) for c in cs] + [mean])
        latex.append([latex_escape(x) for x in r] + [fmt_cell(c, r"$\pm$") for c in cs] + [mean])
    n_label = len(rows[0]) if rows else 1
    full_header = list(header) + ["Mean"]
    write_csv(full_header, plain, os.path.join(out_dir, f"{stem}.csv"))
    write_markdown(full_header, plain, os.path.join(out_dir, f"{stem}.md"),
                   align_right=[False] * n_label + [True] * (len(full_header) - n_label))
    write_latex([latex_escape(h) for h in full_header], latex, os.path.join(out_dir, f"{stem}.tex"))


def comparison_rows(pin, records, tasks, sources):
    """Row order: Baseline, then per conditioned run its unconditioned row
    followed by one row per source."""
    rows, cells, keys = [], [], []
    for name in pin["runs"]:
        prefix = {"meco": "meco_", "url": "text_"}.get(name)
        tags = ["unconditioned"] + ([f"{prefix}{s}" for s in sources] if prefix else [])
        for tag in tags:
            rec = records.get((name, tag))
            if rec is None:
                continue
            cond = "none" if tag == "unconditioned" else tag.split("_", 1)[1]
            rows.append([run_label(name), cond])
            cells.append(task_values(rec, tasks))
            keys.append((name, tag))
    return rows, cells, keys


def plot_comparison(pin, records, tasks, sources, out_dir):
    """(1) per-task grouped bars for the three unconditioned runs; (2) mean
    accuracy per conditioning source, MeCo vs. URL-encoded, with the Baseline
    as a reference line."""
    uncond = [(n, records.get((n, "unconditioned"))) for n in pin["runs"]]
    uncond = [(n, r) for n, r in uncond if r is not None]
    if uncond:
        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(tasks) + 2), 4.8))
            x = np.arange(len(tasks) + 1)
            width = 0.8 / len(uncond)
            for i, (name, rec) in enumerate(uncond):
                cs = task_values(rec, tasks)
                vals = [c[0] if c else np.nan for c in cs] + [mean_of(cs) or np.nan]
                errs = [c[1] if c and c[1] is not None else 0 for c in cs] + [0]
                ax.bar(x + (i - (len(uncond) - 1) / 2) * width, vals, width * 0.92, yerr=errs,
                       color=run_color(pin, name), label=run_label(name), capsize=2)
            ax.set_xticks(x)
            ax.set_xticklabels(tasks + ["Mean"], rotation=20, ha="right")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Unconditioned Benchmarks ({pin['label']})")
            ax.legend()
            save_figure(fig, out_dir, "comparison_unconditioned_by_task")

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(sources) + 3), 4.8))
        conds = ["unconditioned"] + list(sources)
        x = np.arange(len(conds))
        runs = [n for n in pin["runs"] if n in ("meco", "url")]
        width = 0.8 / max(1, len(runs))
        for i, name in enumerate(runs):
            prefix = "meco_" if name == "meco" else "text_"
            vals = []
            for c in conds:
                rec = records.get((name, "unconditioned" if c == "unconditioned" else prefix + c))
                vals.append(mean_of(task_values(rec, tasks)) if rec else np.nan)
            ax.bar(x + (i - (len(runs) - 1) / 2) * width, vals, width * 0.92,
                   color=run_color(pin, name), label=run_label(name))
        base = records.get(("baseline", "unconditioned"))
        if base:
            ax.axhline(mean_of(task_values(base, tasks)), color=run_color(pin, "baseline"), ls="--",
                       lw=1.5, label="Baseline")
        ax.set_xticks(x)
        ax.set_xticklabels(["none"] + list(sources), rotation=15, ha="right")
        ax.set_ylabel("Mean accuracy")
        ax.set_title(f"Steering by Source ({pin['label']})")
        lo = np.nanmin([b.get_height() for b in ax.patches] + [ax.get_ylim()[1]])
        ax.set_ylim(max(0, lo - 0.03), None)
        ax.legend()
        save_figure(fig, out_dir, "comparison_sources_mean")
