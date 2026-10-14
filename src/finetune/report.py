"""Figures and tables for the fine-tuning experiments
(scripts/finetune/make_finetune_report.py). Reads only saved outputs,
outputs/experiments/<size>/finetune/<run>/<ckpt label>/<method>/<task>/seed<K>/{metrics.jsonl,
result.json}, so it can be rerun at any time: curves come from metrics.jsonl
(so unfinished jobs show up too), tables from finished jobs' result.json.

Training seeds are aggregated: curves are the mean over seeds (shaded +-1 std
across seeds), and a table cell is the mean over seeds +- the std across
seeds. With a single seed, a cell's +- is instead that run's stderr over eval
documents -- a much smaller kind of uncertainty, so tables say which one
they show.
"""

import json
import math
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.figures import BASE_COLORS, STYLE, base_name, get_color, save_figure
from src.comparison.report import run_label, write_table
from src.steering.report import PALETTE

MODE_STYLE = {"standard": "-", "conditioned": "--", "peft": ":"}
METHOD_LABEL = {"standard": "standard FT", "conditioned": "conditioned FT", "peft": "PEFT (vector)",
                "peft_null": "PEFT (from null)", "peft_1vec": "PEFT (1 vector)",
                "conditioned_encoder": "conditioned FT (+encoder)"}


def method_label(m):
    base, _, suffix = m.partition("_lm")
    return METHOD_LABEL.get(m, METHOD_LABEL.get(base, m) + (" [LM loss]" if m.endswith("_lm") else ""))


def _load_one(d):
    mpath, rpath = os.path.join(d, "metrics.jsonl"), os.path.join(d, "result.json")
    if not os.path.exists(mpath):
        return None
    curve = {}
    for line in open(mpath):
        if line.strip():
            r = json.loads(line)
            if r.get("split") == "eval":
                curve.setdefault(r["task"], {})[r["step"]] = (r["select_acc"], r["eval_acc"])
    return {"curve": curve, "result": json.load(open(rpath)) if os.path.exists(rpath) else None}


def load_jobs(out_root, pin, label):
    """{(run key, method, task): {seed: {"curve": {task: {step: (select, eval)}},
    "result": dict or None}}}."""
    jobs = {}
    for name, info in pin["runs"].items():
        base = os.path.join(out_root, os.path.basename(os.path.normpath(info["run_dir"])), label)
        if not os.path.isdir(base):
            continue
        for method in sorted(os.listdir(base)):
            for task in sorted(os.listdir(os.path.join(base, method))):
                tdir = os.path.join(base, method, task)
                for sd in sorted(os.listdir(tdir)):
                    if sd.startswith("seed") and (job := _load_one(os.path.join(tdir, sd))):
                        jobs.setdefault((name, method, task), {})[int(sd[4:])] = job
    return jobs


def n_seeds(jobs):
    return max((len(v) for v in jobs.values()), default=0)


def _seed_curves(jobs, run, method, task, which):
    """[{step: acc}] per seed, for `task` from its own jobs, or from the joint
    'all' jobs when task == 'all:<t>'. which: 0 = select, 1 = eval curve."""
    if task.startswith("all:"):
        seeds, t = jobs.get((run, method, "all"), {}), task[4:]
    else:
        seeds, t = jobs.get((run, method, task), {}), task
    return [{s: v[which] for s, v in job["curve"].get(t, {}).items()} for job in seeds.values()
            if job["curve"].get(t)]


def _agg(per_seed):
    """Mean and std across seeds at the steps every seed has."""
    if not per_seed:
        return [], [], []
    steps = sorted(set.intersection(*(set(c) for c in per_seed)))
    vals = np.array([[c[s] for s in steps] for c in per_seed])
    return steps, vals.mean(0), (vals.std(0, ddof=1) if len(per_seed) > 1 else np.zeros(len(steps)))


def _curve(jobs, run, method, task, which):
    """(steps, mean acc, std acc across seeds)."""
    return _agg(_seed_curves(jobs, run, method, task, which))


def _mean_curve(jobs, run, method, tasks, which, prefix=""):
    """Mean over tasks (per seed, then mean +- std across seeds) at the steps
    every task and seed has. Seeds missing a task are left out."""
    by_task = [_seed_curves(jobs, run, method, prefix + t, which) for t in tasks]
    if any(not c for c in by_task):
        return [], [], []
    k = min(len(c) for c in by_task)
    per_seed = []
    for i in range(k):
        cs = [c[i] for c in by_task]
        steps = set.intersection(*(set(c) for c in cs))
        per_seed.append({s: sum(c[s] for c in cs) / len(cs) for s in steps})
    return _agg(per_seed)


def _grid(tasks):
    n = len(tasks) + 1
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.6 * rows))
    return fig, axes.ravel()


def plot_curves(jobs, pin, tasks, methods, runs, out_dir, stem, title, which=1, color_by="run", prefix=""):
    """One panel per task + a Mean panel: accuracy vs fine-tuning step for
    every (run, method) given. color_by "run": run colors, method linestyles;
    "method": one color per method (for a single run)."""
    lines = [(r, m) for r in runs for m in methods
             if any(_curve(jobs, r, m, prefix + t, which)[0] for t in tasks)]
    if not lines:
        return
    # linestyles tell methods apart; with only one method every line is solid
    single_method = len({m for _, m in lines}) == 1
    with plt.style.context(STYLE):
        fig, axes = _grid(tasks)
        for j, t in enumerate(tasks + ["Mean"]):
            ax = axes[j]
            for i, (r, m) in enumerate(lines):
                xs, ys, sd = (_mean_curve(jobs, r, m, tasks, which, prefix) if t == "Mean"
                              else _curve(jobs, r, m, prefix + t, which))
                if not xs:
                    continue
                mode = m.split("_")[0]
                if color_by == "run":
                    color, ls = _run_color(pin, r), "-" if single_method else MODE_STYLE.get(mode, "-.")
                else:
                    color, ls = PALETTE[methods.index(m) % len(PALETTE)], "-"
                label = run_label(r) if single_method else f"{run_label(r)}: {method_label(m)}"
                ax.plot(xs, 100 * ys, color=color, ls=ls, lw=2.2, label=label)
                if sd.any():
                    ax.fill_between(xs, 100 * (ys - sd), 100 * (ys + sd), color=color, alpha=0.12, lw=0)
            ax.set_title(t, fontsize=11)
            ax.set_xlabel("Fine-tuning step", fontsize=10)
            ax.set_ylabel(("Select" if which == 0 else "Eval") + " accuracy (%)", fontsize=10)
            ax.grid(True, alpha=0.35)
        for ax in axes[len(tasks) + 1:]:
            ax.axis("off")
        handles, labels = axes[len(tasks)].get_legend_handles_labels()
        ncol = min(len(labels), 6 if single_method else 3)
        fig.legend(handles, labels, loc="upper center", ncol=ncol, fontsize=11 if single_method else 9,
                   frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(title, fontsize=15, weight="bold", y=1.0 + 0.025 * math.ceil(len(labels) / ncol) + 0.015)
        save_figure(fig, out_dir, stem)


def with_full_step0(jobs):
    """A copy of `jobs` whose step-0 eval point is the full-eval accuracy
    (result.json's "base") instead of the curve_eval_docs subset's -- the
    subset is noisy enough to shift the start by a point or more."""
    out = {}
    for key, seeds in jobs.items():
        out[key] = {}
        for sd, job in seeds.items():
            base = (job["result"] or {}).get("base", {})
            curve = {t: dict(c) for t, c in job["curve"].items()}
            for t, c in curve.items():
                if 0 in c and t in base and base[t].get("eval") is not None:
                    c[0] = (c[0][0], base[t]["eval"])
            out[key][sd] = {**job, "curve": curve}
    return out


def plot_pair_curves(jobs, pin, tasks, pairs, out_dir, stem, title, which=1, full_step0=False, mean_only=False):
    """Like plot_curves, for an explicit list of (run, method, label) lines --
    e.g. the single-vector PEFT comparison, where each run uses a different
    method. Solid lines in run colors, one panel per task + Mean. full_step0:
    step 0 is the full-eval accuracy (with_full_step0), drawn as an open marker;
    later points stay on the curve subset. mean_only: just the Mean panel,
    as one standalone figure."""
    if full_step0:
        jobs = with_full_step0(jobs)
    pairs = [(r, m, lab) for r, m, lab in pairs if any(_curve(jobs, r, m, t, which)[0] for t in tasks)]
    if not pairs:
        return
    with plt.style.context(STYLE):
        if mean_only:
            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            axes, panels = [ax], ["Mean"]
        else:
            fig, axes = _grid(tasks)
            panels = tasks + ["Mean"]
        for j, t in enumerate(panels):
            ax = axes[j]
            for r, m, lab in pairs:
                xs, ys, sd = _mean_curve(jobs, r, m, tasks, which) if t == "Mean" else _curve(jobs, r, m, t, which)
                if not xs:
                    continue
                color = _run_color(pin, r)
                ax.plot(xs, 100 * ys, color=color, lw=2.2, label=lab)
                if full_step0 and xs[0] == 0:
                    ax.plot([0], [100 * ys[0]], "o", ms=7, mfc="white", mec=color, mew=2, zorder=5)
                if sd.any():
                    ax.fill_between(xs, 100 * (ys - sd), 100 * (ys + sd), color=color, alpha=0.12, lw=0)
            if mean_only:
                ax.set_xlabel("Fine-tuning step")
                ax.set_ylabel(f"Mean {'select' if which == 0 else 'eval'} accuracy over {len(tasks)} tasks (%)")
                ax.grid(True, alpha=0.35)
                ax.legend(loc="lower right", frameon=False)
                ax.set_title(title)
                continue
            ax.set_title(t, fontsize=12, weight="bold" if t == "Mean" else None)
            ax.set_xlabel("Fine-tuning step", fontsize=10)
            ax.set_ylabel(("Select" if which == 0 else "Eval") + " accuracy (%)", fontsize=10)
            ax.grid(True, alpha=0.35)
        if mean_only:
            save_figure(fig, out_dir, stem)
            return
        for ax in axes[len(tasks) + 1:]:
            ax.axis("off")
        handles, labels = axes[len(tasks)].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), fontsize=11, frameon=False,
                   bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(title, fontsize=15, weight="bold", y=1.04)
        save_figure(fig, out_dir, stem)


def _run_color(pin, name):
    """The run family's color (src/analysis/figures.py), distinct per run even
    for families without one."""
    run_dir = os.path.basename(os.path.normpath(pin["runs"][name]["run_dir"]))
    family = re.sub(r"_p[0-9.]+$", "", base_name(run_dir))      # meco_p0.9_medium -> meco
    if family in BASE_COLORS:
        return BASE_COLORS[family]
    return get_color(run_dir, fallback_index=list(pin["runs"]).index(name))


def _values(seeds, key, task):
    """Full-eval accuracy of each finished seed."""
    return [j["result"][key][task]["eval"] for j in (seeds or {}).values()
            if j["result"] is not None and task in j["result"].get(key, {})]


def _cell(seeds, key, task):
    """(mean over seeds, std across seeds) -- or, for one seed, (acc, its
    stderr over eval docs)."""
    vals = _values(seeds, key, task)
    if not vals:
        return None
    if len(vals) == 1 or key == "base":       # step 0 is the same model for every seed
        r = next(j["result"] for j in seeds.values() if j["result"] is not None)[key][task]
        return (r["eval"], r.get("eval_stderr"))
    return (float(np.mean(vals)), float(np.std(vals, ddof=1)))


def _paired(seeds_a, seeds_b, key_a, key_b, task):
    """Per-seed difference a - b (seeds both have), as (mean, std across seeds)."""
    common = [s for s in (seeds_a or {}) if s in (seeds_b or {})]
    d = [_values({s: seeds_a[s]}, key_a, task)[0] - _values({s: seeds_b[s]}, key_b, task)[0]
         for s in common if _values({s: seeds_a[s]}, key_a, task) and _values({s: seeds_b[s]}, key_b, task)]
    if not d:
        return None
    return (float(np.mean(d)), float(np.std(d, ddof=1)) if len(d) > 1 else None)


def result_tables(jobs, pin, tasks, methods, out_dir, joint=False):
    """finetune_results: zero-shot (the standard job's step 0) and each
    method's best-on-select iterate, on the full eval split.
    finetune_final: the same with the final iterate. finetune_delta: best
    minus the method's own step 0, in points."""
    suffix = "_joint" if joint else ""
    get = (lambda r, m, t: jobs.get((r, m, "all"))) if joint else (lambda r, m, t: jobs.get((r, m, t)))
    for key, stem in (("best", "finetune_results"), ("final", "finetune_final")):
        rows, cells = [], []
        for r in pin["runs"]:
            zs = [_cell(get(r, "standard", t), "base", t) for t in tasks]
            if any(zs):
                rows.append([run_label(r), "zero-shot"])
                cells.append(zs)
            for m in methods:
                cs = [_cell(get(r, m, t), key, t) for t in tasks]
                if any(cs):
                    rows.append([run_label(r), method_label(m)])
                    cells.append(cs)
        if rows:
            write_table(["Run", "Method"] + tasks, rows, cells, out_dir, stem + suffix)
    rows, cells = [], []
    for r in pin["runs"]:
        for m in methods:
            cs = [_paired(get(r, m, t), get(r, m, t), "best", "base", t) for t in tasks]
            if any(cs):
                rows.append([run_label(r), method_label(m)])
                cells.append(cs)
    if rows:
        write_table(["Run", "Method"] + tasks, rows, cells, out_dir, "finetune_delta" + suffix)
    # each method against standard fine-tuning of the same run, paired by seed
    rows, cells = [], []
    for r in pin["runs"]:
        for m in methods:
            if m == "standard":
                continue
            cs = [_paired(get(r, m, t), get(r, "standard", t), "best", "best", t) for t in tasks]
            if any(cs):
                rows.append([run_label(r), f"{method_label(m)} − standard FT"])
                cells.append(cs)
    if rows:
        write_table(["Run", "Comparison"] + tasks, rows, cells, out_dir, "finetune_vs_standard" + suffix)


def best_steps_table(jobs, pin, tasks, methods, out_dir):
    from src.analysis.tables import write_markdown
    rows = []
    for r in pin["runs"]:
        for m in methods:
            cs = [[j["result"]["best_step"] for _, j in sorted((jobs.get((r, m, t)) or {}).items())
                   if j["result"] is not None] for t in tasks]
            if any(cs):
                rows.append([run_label(r), method_label(m)] + ["/".join(map(str, c)) or "—" for c in cs])
    if rows:
        write_markdown(["Run", "Method"] + tasks, rows, os.path.join(out_dir, "finetune_best_steps.md"),
                       align_right=[False, False] + [True] * len(tasks))
