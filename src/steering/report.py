"""Tables and figures for the steering experiments
(scripts/steering/make_steering_report.py). Reads only saved outputs:

  mixtures      outputs/experiments/<size>/benchmarks/<url run>/<label>/<tag>/results.json  (+ vectors.pt)
  task vectors  outputs/experiments/<size>/steering/<label>/task_vectors/<run>/<task>.{json,pt}
  transfer      outputs/experiments/<size>/steering/<label>/task_vectors/<run>/transfer.json
"""

import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import torch

from src.analysis.figures import STYLE, save_figure
from src.analysis.tables import write_csv
from src.comparison.report import (common_task_idx, mean_label, mean_of, shared_records, run_color, run_label,
                                   task_values, write_table)
from src.analysis.tables import write_markdown as _write_markdown
from src.url_analysis.analyses import neighbors

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#8C564B", "#7F7F7F"]


def write_markdown(header, rows, path, align_right=None):
    _write_markdown(header, [[str(c) for c in r] for r in rows], path, align_right=align_right)


def _ref_lines(ax, pin, records, tasks, task=None):
    """Horizontal reference lines: each run's unconditioned accuracy."""
    for name in pin["runs"]:
        rec = records.get((name, "unconditioned"))
        if rec is None:
            continue
        cells = task_values(rec, tasks)
        v = mean_of(cells) if task is None else (cells[tasks.index(task)] or (None,))[0]
        if v is not None:
            ax.axhline(v, color=run_color(pin, name), ls="--", lw=1.2, alpha=0.8,
                       label=f"{run_label(name)} (none)")


# ================================================================================
# Experiment 1: mixtures
# ================================================================================

def mixture_records(pin):
    recs = shared_records(pin)
    url = {tag: r for (n, tag), r in recs.items() if n == "url"}
    return recs, url


def _anchor_or_tag(url_recs, a, b, alpha):
    if alpha == 0:
        return url_recs.get(f"anchor_{a}")
    if alpha == 1:
        return url_recs.get(f"anchor_{b}")
    return url_recs.get(f"mix_{a}+{b}_a{alpha:.2f}")


def plot_mixtures(pin, recs, url_recs, mix_cfg, tasks, out_dir):
    alphas = mix_cfg.get("alphas", [])
    pairs = mix_cfg.get("pairs", [])
    # (a) all pairs, mean accuracy vs alpha
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (a, b) in enumerate(pairs):
            ys = [mean_of(task_values(r, tasks)) if (r := _anchor_or_tag(url_recs, a, b, al)) else np.nan
                  for al in alphas]
            ax.plot(alphas, ys, marker="o", color=PALETTE[i % len(PALETTE)], label=f"{a} → {b}")
        _ref_lines(ax, pin, recs, tasks)
        ax.set_xlabel("α (weight on second anchor)")
        ax.set_ylabel("Mean accuracy")
        ax.set_title("Interpolating URL vectors")
        ax.legend(fontsize=8.5, ncol=2)
        save_figure(fig, out_dir, "mixtures_all_pairs_mean")

    # (b) per pair, per task small multiples
    for a, b in pairs:
        with plt.style.context(STYLE):
            n = len(tasks) + 1
            cols = 3
            fig, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(4.2 * cols, 3.2 * math.ceil(n / cols)))
            axes = axes.ravel()
            for j, t in enumerate(tasks + ["Mean"]):
                ax = axes[j]
                ys, es = [], []
                for al in alphas:
                    r = _anchor_or_tag(url_recs, a, b, al)
                    if r is None:
                        ys.append(np.nan); es.append(0); continue
                    cells = task_values(r, tasks)
                    if t == "Mean":
                        ys.append(mean_of(cells)); es.append(0)
                    else:
                        c = cells[tasks.index(t)]
                        ys.append(c[0] if c else np.nan); es.append(c[1] or 0 if c else 0)
                ax.errorbar(alphas, ys, yerr=es, marker="o", color=run_color(pin, "url"), capsize=2)
                _ref_lines(ax, pin, recs, tasks, None if t == "Mean" else t)
                ax.set_title(t, fontsize=11)
                ax.set_xticks(alphas)
            for ax in axes[n:]:
                ax.axis("off")
            if axes[0].get_legend_handles_labels()[0]:
                axes[0].legend(fontsize=7)
            fig.suptitle(f"{a} (α=0) → {b} (α=1)", fontsize=14, weight="bold")
            save_figure(fig, out_dir, f"mixture_{a}+{b}_by_task")

    # (c) steering strength
    sc = mix_cfg.get("scale")
    if sc:
        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(8, 5))
            for i, a in enumerate(sc["anchors"]):
                xs, ys = [], []
                for s in sc["strengths"]:
                    r = url_recs.get(f"anchor_{a}" if s == 1 else f"scale_{a}_s{s:.2f}")
                    if r:
                        xs.append(s); ys.append(mean_of(task_values(r, tasks)))
                ax.plot(xs, ys, marker="o", color=PALETTE[i % len(PALETTE)], label=a)
            _ref_lines(ax, pin, recs, tasks)
            ax.axvline(1, color="gray", lw=0.8, ls=":")
            ax.set_xlabel(f"steering strength s  (z = ref + s·(z_anchor − ref), ref = {sc.get('reference', 'mean')})")
            ax.set_ylabel("Mean accuracy")
            ax.set_title("Steering strength")
            ax.legend(fontsize=9)
            save_figure(fig, out_dir, "mixtures_strength_mean")


def mixture_table(url_recs, tasks, out_dir):
    order = sorted(url_recs, key=lambda t: (t.split("_")[0] != "unconditioned", t))
    rows = [[t] for t in order]
    cells = [task_values(url_recs[t], tasks) for t in order]
    write_table(["Vector"] + tasks, rows, cells, out_dir, "mixtures")


def mixture_neighbors(vectors, lib, out_dir, k=8):
    tags = list(vectors)
    z = torch.stack([vectors[t]["z"] for t in tags])
    nb = neighbors(z, tags, lib, k)
    rows = [[t, f"{float(vectors[t]['z'].norm()):.1f}", ", ".join(f"{n['domain']} ({n['cos']:.2f})" for n in nb[t])]
            for t in tags]
    write_markdown(["Vector", "Norm", f"Nearest {k} library domains (centered cos)"], rows,
                   os.path.join(out_dir, "mixture_neighbors.md"), align_right=[False, True, False])
    write_csv(["vector", "norm", "neighbors"], rows, os.path.join(out_dir, "mixture_neighbors.csv"))


# ================================================================================
# Experiment 2: task vectors
# ================================================================================

def load_task_results(tv_dir, runs, tasks):
    """{run: {task_or_all: (json, pt or None)}}"""
    out = {}
    for run in runs:
        for t in tasks + ["all"]:
            j = os.path.join(tv_dir, run, f"{t}.json")
            if os.path.exists(j):
                ptp = os.path.join(tv_dir, run, f"{t}.pt")
                pt = torch.load(ptp, weights_only=False) if os.path.exists(ptp) else None
                out.setdefault(run, {})[t] = (json.load(open(j)), pt)
    return out


def with_best_source(res):
    """Add a derived "best_source" method: the named source with the best
    select accuracy (chosen on select, so it is a fair method, not an oracle)."""
    ms = res["methods"]
    srcs = [m for m in ms if m.startswith("source:")]
    if srcs:
        best = max(srcs, key=lambda m: ms[m]["mean_select"])
        ms["best_source"] = {**ms[best], "chosen": best.split(":", 1)[1]}
    return res


METHOD_ORDER = ["unconditioned", "best_source", "domain_search", "opt_simplex", "opt_free"]
METHOD_LABEL = {"unconditioned": "none", "best_source": "best source", "domain_search": "best domain",
                "opt_simplex": "simplex-optimized", "opt_free": "free-optimized"}


def task_vector_rows(results, tasks, runs):
    """Per-task jobs: one row per (run, method), each task's cell from that
    task's own job. Returns (rows, cells, pairs=[(run, method)])."""
    rows, cells, keys = [], [], []
    for run in runs:
        per = results.get(run, {})
        for m in METHOD_ORDER:
            cs, any_ = [], False
            for t in tasks:
                if t not in per:
                    cs.append(None); continue
                res = with_best_source(per[t][0])
                mm = res["methods"].get(m)
                if mm is None:
                    cs.append(None); continue
                any_ = True
                tr = mm["tasks"][t]
                cs.append((tr["eval"], tr["eval_stderr"]))
            if any_:
                rows.append([run_label(run), METHOD_LABEL[m]])
                cells.append(cs)
                keys.append((run, m))
    return rows, cells, keys


def joint_rows(results, tasks, runs):
    rows, cells = [], []
    for run in runs:
        if "all" not in results.get(run, {}):
            continue
        res = with_best_source(results[run]["all"][0])
        for m in METHOD_ORDER:
            mm = res["methods"].get(m)
            if mm:
                rows.append([run_label(run), METHOD_LABEL[m]])
                cells.append([(mm["tasks"][t]["eval"], mm["tasks"][t]["eval_stderr"]) if t in mm["tasks"] else None
                              for t in tasks])
    return rows, cells


def paired_delta(results, run, task, method):
    """Mean paired eval difference (method - unconditioned) on `task`, and its
    stderr, from the per-doc correctness saved in <task>.pt."""
    per = results.get(run, {}).get(task)
    if per is None or per[1] is None:
        return None
    pd = per[1]["per_doc"]
    if method == "best_source":
        best = with_best_source(per[0])["methods"].get("best_source")
        if best is None:
            return None
        method = "source:" + best["chosen"]
    base_m = method
    if base_m not in pd or "unconditioned" not in pd:
        return None
    d = np.array(pd[base_m][task], dtype=float) - np.array(pd["unconditioned"][task], dtype=float)
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))


def delta_table(results, tasks, runs, out_dir):
    rows = []
    for run in runs:
        for m in METHOD_ORDER[1:]:
            cells = [paired_delta(results, run, t, m) for t in tasks]
            if all(c is None for c in cells):
                continue
            fmt = ["—" if c is None else f"{100 * c[0]:+.1f} ± {100 * c[1]:.1f}" for c in cells]
            vals = [c[0] for c in cells if c]
            rows.append([run_label(run), METHOD_LABEL[m]] + fmt + [f"{100 * np.mean(vals):+.1f}" if vals else "—"])
    header = ["Run", "Method"] + tasks + ["Mean"]
    write_markdown(header, rows, os.path.join(out_dir, "task_vectors_delta_vs_unconditioned.md"),
                   align_right=[False, False] + [True] * (len(tasks) + 1))
    write_csv(header, rows, os.path.join(out_dir, "task_vectors_delta_vs_unconditioned.csv"))


def plot_task_vectors(pin, rows, cells, keys, tasks, out_dir, methods=None, stem="task_vectors_by_task",
                      title="Task-optimized conditioning"):
    """methods: only these METHOD_ORDER keys (default: all of them)."""
    if methods is not None:
        keep = [i for i, k in enumerate(keys) if k[1] in methods]
        rows, cells, keys = [rows[i] for i in keep], [cells[i] for i in keep], [keys[i] for i in keep]
    if not rows:
        return
    labels = [f"{r[0]}: {r[1]}" for r in rows]
    shades = {}
    for run in pin["runs"]:
        idx = [i for i, k in enumerate(keys) if k[0] == run]
        base = matplotlib.colors.to_rgb(run_color(pin, run))
        for j, i in enumerate(idx):
            f = 0.35 + 0.65 * (j + 1) / len(idx)
            shades[i] = tuple(1 - f * (1 - c) for c in base)
    common = common_task_idx(cells)
    with plt.style.context(STYLE):
        n = len(tasks) + 1
        cols = 3
        fig, axes = plt.subplots(math.ceil(n / cols), cols, figsize=(5.2 * cols, 3.6 * math.ceil(n / cols)))
        axes = axes.ravel()
        for j, t in enumerate(tasks + ["Mean"]):
            ax = axes[j]
            if t == "Mean":
                # over the tasks every row has, so each bar averages the same tasks
                vals = [mean_of([c[k] for k in common]) or np.nan for c in cells]
                t = mean_label(len(common), len(tasks))
            else:
                vals = [(c[tasks.index(t)] or (np.nan,))[0] for c in cells]
            errs = [0 if j == len(tasks) else ((c[j] or (0, 0))[1] or 0) for c in cells]
            x = np.arange(len(vals))
            ax.bar(x, vals, yerr=errs, color=[shades[i] for i in range(len(vals))], capsize=2)
            finite = [v for v in vals if v is not None and not np.isnan(v)]
            if finite:
                ax.set_ylim(max(0, min(finite) - 0.04), max(finite) + 0.03)
            ax.set_xticks([])
            ax.set_title(t, fontsize=11)
        for ax in axes[n:]:
            ax.axis("off")
        handles = [plt.Rectangle((0, 0), 1, 1, color=shades[i]) for i in range(len(labels))]
        fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), fontsize=9,
                   bbox_to_anchor=(0.5, -0.02 - 0.03 * math.ceil(len(labels) / 5)))
        fig.suptitle(f"{title} (eval split, {pin['label']})", fontsize=14, weight="bold")
        save_figure(fig, out_dir, stem)


def nearest_table(results, tasks, out_dir, k=8):
    rows = []
    for t in tasks + ["all"]:
        url = results.get("url", {}).get(t)
        meco = results.get("meco", {}).get(t)
        um = url[0]["methods"] if url else {}
        row = [t,
               meco[0]["methods"].get("domain_search", {}).get("chosen", "—") if meco else "—",
               um.get("domain_search", {}).get("chosen", "—"),
               ", ".join(f"{w['domain']} ({w['weight']:.2f})" for w in um.get("opt_simplex", {}).get("top_weights", [])[:k]) or "—",
               ", ".join(f"{n['domain']} ({n['cos']:.2f})" for n in um.get("opt_free", {}).get("nearest", [])[:k]) or "—"]
        rows.append(row)
    header = ["Task", "MeCo best domain", "URL best domain", "Simplex top weights",
              "Free vector: nearest domains (centered cos)"]
    write_markdown(header, rows, os.path.join(out_dir, "task_vectors_domains.md"), align_right=[False] * 5)
    write_csv(header, rows, os.path.join(out_dir, "task_vectors_domains.csv"))


def ranking_table(results, tasks, out_dir, k=10):
    lines = []
    for t in tasks + ["all"]:
        lines.append(f"## {t}\n")
        lines.append("| Rank | MeCo domain | select acc | URL domain | select acc |")
        lines.append("| ---: | --- | ---: | --- | ---: |")
        rk = {run: results.get(run, {}).get(t, ({"methods": {}},))[0]["methods"].get("domain_search", {}).get("ranking", [])
              for run in ("meco", "url")}
        cell = lambda r: f"{r['name']} | {100 * r['acc']:.1f}" if r else "— | —"
        for i in range(k):
            m = rk["meco"][i] if i < len(rk["meco"]) else None
            u = rk["url"][i] if i < len(rk["url"]) else None
            lines.append(f"| {i + 1} | {cell(m)} | {cell(u)} |")
        lines.append("")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "domain_search_rankings.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_histories(results, tasks, out_dir):
    url = results.get("url", {})
    if not url:
        return
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
        for ax, m in zip(axes, ("opt_simplex", "opt_free")):
            for i, t in enumerate(tasks + ["all"]):
                mm = url.get(t, ({"methods": {}},))[0]["methods"].get(m)
                if not mm:
                    continue
                h = [r for r in mm["history"] if "select_acc" in r]
                base = h[0]["select_acc"]
                ax.plot([r["step"] for r in h], [r["select_acc"] - base for r in h], marker=".",
                        color=PALETTE[i % len(PALETTE)], label=t, lw=1.6)
            ax.axhline(0, color="gray", lw=0.8)
            ax.set_xlabel("step")
            ax.set_ylabel("select acc − init")
            ax.set_title(METHOD_LABEL[m])
        axes[0].legend(fontsize=8, ncol=2)
        save_figure(fig, out_dir, "task_vectors_training")


def plot_transfer(tv_dir, results, tasks, out_dir):
    for run in ("url", "meco"):
        path = os.path.join(tv_dir, run, "transfer.json")
        if not os.path.exists(path):
            continue
        tr = json.load(open(path))
        base = {t: results.get(run, {}).get(t, ({"methods": {}},))[0]["methods"].get("unconditioned", {})
                .get("tasks", {}).get(t, {}).get("eval") for t in tasks}
        for m, mat in tr["acc"].items():
            srcs = [s for s in tasks + ["all"] if s in mat]
            if not srcs:
                continue
            D = np.array([[100 * (mat[s][t] - base[t]) if base[t] is not None else np.nan for t in tasks]
                          for s in srcs])
            lim = np.nanmax(np.abs(D)) if np.isfinite(D).any() else 1
            with plt.style.context(STYLE):
                fig, ax = plt.subplots(figsize=(1.0 * len(tasks) + 3, 0.6 * len(srcs) + 2))
                im = ax.imshow(D, cmap="RdBu", vmin=-lim, vmax=lim)
                for i in range(len(srcs)):
                    for j in range(len(tasks)):
                        if np.isfinite(D[i, j]):
                            ax.text(j, i, f"{D[i, j]:+.1f}", ha="center", va="center", fontsize=8)
                ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, rotation=30, ha="right")
                ax.set_yticks(range(len(srcs))); ax.set_yticklabels(srcs)
                ax.set_xlabel("evaluated on"); ax.set_ylabel("vector fit on")
                ax.grid(False)
                fig.colorbar(im, ax=ax, fraction=0.04, label="Δ acc vs unconditioned (pts)")
                ax.set_title(f"Transfer: {run_label(run)} {METHOD_LABEL.get(m, m)}")
                save_figure(fig, out_dir, f"transfer_{run}_{m}")


def sanity_check(pin, recs, results, tasks, out_dir):
    """Our scorer's unconditioned eval accuracy vs. the lm-eval harness
    result for the same run/checkpoint (should match to rounding on tasks
    whose eval split is the full lm-eval split)."""
    rows = []
    for run in pin["runs"]:
        rec = recs.get((run, "unconditioned"))
        for t in tasks:
            per = results.get(run, {}).get(t)
            if rec is None or per is None or t not in rec["results"]:
                continue
            info = per[0]["splits"][t]
            ours = per[0]["methods"].get("unconditioned", {}).get("tasks", {}).get(t, {}).get("eval")
            cell = task_values(rec, [t])[0]
            if ours is None or cell is None:
                continue
            rows.append([run_label(run), t, f"{100 * cell[0]:.2f}", f"{100 * ours:.2f}",
                         "subset" if info["eval_is_subset"] else f"{100 * (ours - cell[0]):+.2f}"])
    if rows:
        write_markdown(["Run", "Task", "lm-eval", "steering scorer", "diff"], rows,
                       os.path.join(out_dir, "sanity_scorer_vs_lm_eval.md"))
