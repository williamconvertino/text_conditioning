"""Figures and tables for the URL-encoder analysis, drawn only from the saved
results directory (scripts/url_analysis/plot_encoder_analysis.py), so they
can be restyled and regenerated without re-encoding anything.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis.figures import STYLE, save_figure
from src.analysis.tables import write_csv
from src.analysis.tables import write_markdown as _write_markdown

# Okabe-Ito + extensions: categorical colors for up to ~12 groups, then gray.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442",
           "#000000", "#8C564B", "#7F7F7F", "#BCBD22", "#17BECF"]
OTHER = "#C8C8C8"


def write_markdown(header, rows, path, align_right=None):
    _write_markdown(header, [[str(c) for c in r] for r in rows], path, align_right=align_right)


def _color_map(labels, order=None):
    order = order or sorted({l for l in labels if not str(l).startswith("(") and l != "other"},
                            key=lambda l: -labels.count(l))
    cmap = {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(order[:len(PALETTE)])}
    return cmap, [cmap.get(l, OTHER) for l in labels]


def _annotate(ax, xy, names, which, fontsize=7):
    for i in which:
        ax.annotate(names[i], xy[i], xytext=(3, 3), textcoords="offset points", fontsize=fontsize,
                    color="#222222", alpha=0.9)


# -- global map ------------------------------------------------------------------

def plot_domain_map(dm, out_dir, n_label=40, highlight=(), coords="tsne"):
    xy = np.array(dm[coords])
    names, counts = dm["domains"], np.array(dm["counts"])
    size = 6 + 30 * (np.log(counts) - np.log(counts).min()) / max(1e-9, np.ptp(np.log(counts)))
    label_idx = set(range(min(n_label, len(names)))) | {i for i, n in enumerate(names) if n in set(highlight)}

    for color_by in ("category", "cluster"):
        labels = [str(c) for c in dm[color_by]]
        if color_by == "category":
            cmap, colors = _color_map(labels)
        else:
            uniq = sorted(set(labels), key=int)
            cmap = {l: plt.get_cmap("tab20")(i % 20) for i, l in enumerate(uniq)}
            colors = [cmap[l] for l in labels]
        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(11, 9))
            ax.scatter(xy[:, 0], xy[:, 1], s=size, c=colors, alpha=0.75, linewidths=0)
            _annotate(ax, xy, names, sorted(label_idx))
            if color_by == "category":
                for l, c in cmap.items():
                    ax.scatter([], [], c=[c], s=30, label=l)
                ax.scatter([], [], c=[OTHER], s=30, label="other")
                ax.legend(loc="best", fontsize=9, markerscale=1.2, frameon=True)
            else:
                for l in cmap:
                    pts = xy[np.array(labels) == l]
                    ax.text(*np.median(pts, 0), l, fontsize=11, weight="bold", ha="center",
                            color="black", alpha=0.6)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"URL-encoder domain map ({coords.upper()}, top {len(names)} domains, "
                         f"colored by {color_by})")
            save_figure(fig, out_dir, f"domain_map_{coords}_{color_by}")


def write_cluster_table(dm, out_dir):
    rows = [[c, dm["cluster_sizes"][c], ", ".join(m[:12])] for c, m in dm["cluster_members"].items()]
    write_markdown(["Cluster", "Domains", "Most frequent members"], rows, os.path.join(out_dir, "clusters.md"))
    write_csv(["cluster", "n_domains", "members"], rows, os.path.join(out_dir, "clusters.csv"))


# -- neighbors ---------------------------------------------------------------------

def write_neighbor_tables(nbrs, out_dir, stem, k=10, counts=None):
    rows = []
    for q, lst in nbrs.items():
        c = "" if counts is None else counts.get(q, "")
        rows.append([q, c, ", ".join(f"{n['domain']} ({n['cos']:.2f})" for n in lst[:k])])
    write_markdown(["Query", "Freq.", f"Nearest {k} domains (centered cos)"], rows,
                   os.path.join(out_dir, f"{stem}.md"), align_right=[False, True, False])
    long = [[q, i + 1, n["domain"], f"{n['cos']:.4f}"] for q, lst in nbrs.items() for i, n in enumerate(lst)]
    write_csv(["query", "rank", "domain", "centered_cos"], long, os.path.join(out_dir, f"{stem}.csv"))


def plot_similarity_heatmap(labels, sims, out_dir, stem="query_similarity"):
    if not labels:
        return
    sims = np.array(sims)
    with plt.style.context(STYLE):
        n = len(labels)
        fig, ax = plt.subplots(figsize=(0.45 * n + 3, 0.45 * n + 2))
        im = ax.imshow(sims, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=8); ax.set_yticklabels(labels, fontsize=8)
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.04, label="centered cosine")
        ax.set_title("Similarity between domain vectors")
        save_figure(fig, out_dir, stem)


# -- global statistics ------------------------------------------------------------

def plot_spectrum(spectra, out_dir):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots()
        for name, sp in spectra.items():
            cum = np.cumsum(sp["explained_variance_ratio"])
            ax.plot(np.arange(1, len(cum) + 1), cum, label=f"{name} (PR={sp['participation_ratio']:.1f})")
        ax.set_xlabel("Principal components")
        ax.set_ylabel("Cumulative explained variance")
        ax.set_ylim(0, 1)
        ax.set_title("Effective dimensionality of URL vectors")
        ax.legend()
        save_figure(fig, out_dir, "pca_spectrum")


def plot_stats(stats, out_dir):
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        ax = axes[0]
        ax.hist(stats["page_norms_sample"], bins=60, color=PALETTE[0], alpha=0.8, label="page vectors")
        ax.axvline(stats["norm"]["null"], color=PALETTE[1], ls="--", label="null vector")
        ax.axvline(stats["norm"]["root"]["p50"], color=PALETTE[2], ls=":", label="median root vector")
        ax.set_xlabel("L2 norm"); ax.set_ylabel("Pages"); ax.set_title("Vector norms"); ax.legend()
        ax = axes[1]
        ax.hist(stats["root_vs_centroid_per_domain"], bins=60, color=PALETTE[2], alpha=0.8)
        ax.set_xlabel("centered cos(E(root URL), mean E(page URLs))")
        ax.set_ylabel("Domains")
        ax.set_title("Is the root URL a good domain vector?")
        save_figure(fig, out_dir, "vector_stats")


def plot_retrieval(ret, var, out_dir):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ks = [k for k in ret if k.startswith("top") and k != "top1_per_domain"]
        vals = [ret[k] for k in ks]
        bars = ax.bar(ks, vals, color=PALETTE[0], width=0.6)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center")
        ax.axhline(ret["chance_top1"], color="gray", ls="--", lw=1, label=f"chance (1/{ret['n_domains']})")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Page → own-domain retrieval\n(domain explains {100 * var['r2_domain']:.0f}% "
                     f"of page-vector variance)")
        ax.legend()
        save_figure(fig, out_dir, "domain_retrieval")


def plot_perturbations(pert, out_dir):
    kinds = [k for k in pert["kinds"] if k != "original"]
    vals = [pert["mean"][k] for k in kinds]
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(kinds[::-1], vals[::-1], color=PALETTE[0])
        ref = pert["reference"]
        if ref["same_domain_pages"] is not None:
            ax.axvline(ref["same_domain_pages"], color=PALETTE[2], ls="--", label="two pages, same domain")
        ax.axvline(ref["different_domain_pages"], color=PALETTE[1], ls=":", label="two pages, different domains")
        ax.set_xlabel("centered cos to E(original URL)")
        ax.set_xlim(min(0, min(vals + [ref["different_domain_pages"]]) - 0.05), 1)
        ax.set_title(f"Sensitivity to surface edits (mean over {len(pert['urls'])} URLs)")
        ax.legend(loc="lower left", fontsize=9)
        save_figure(fig, out_dir, "url_perturbations")


def plot_year_probe(probe, out_dir):
    if probe.get("skipped"):
        return
    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        y, p = np.array(probe["test_year"]), np.array(probe["test_pred"])
        ax.scatter(y + np.random.default_rng(0).uniform(-0.3, 0.3, len(y)), p, s=4, alpha=0.3, color=PALETTE[0])
        lo, hi = y.min(), y.max()
        ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1)
        ax.set_xlabel("Year in URL"); ax.set_ylabel("Predicted from URL vector")
        ax.set_title(f"Linear year probe (R²={probe['r2']:.2f}, MAE {probe['mae']:.1f} "
                     f"vs {probe['mae_predict_mean']:.1f})", fontsize=12)
        save_figure(fig, out_dir, "year_probe")


# -- in-domain -------------------------------------------------------------------------

def _label_counts(labels):
    """(top-label share, n "(other)", n "(none)") of one rule's plotted labels."""
    n_other = sum(l == "(other)" for l in labels)
    n_none = sum(l == "(none)" for l in labels)
    return 1 - (n_other + n_none) / max(1, len(labels)), n_other, n_none


def plot_in_domain(name, res, out_dir):
    """One t-SNE panel per coloring: KMeans cluster, then each label rule.
    Every page is plotted; pages outside a rule's top labels are gray, with
    their own legend entries ("other": a rarer label, "none": the rule found
    no label), and the panel title gives the share the colored labels cover."""
    xy = np.array(res["tsne"])
    panels = [("KMeans cluster", [str(c) for c in res["cluster"]], None)]
    for rule, r in res["rules"].items():
        share, _, _ = _label_counts(r["labels"])
        title = f"{rule}: top {len(r['top_labels'])} labels = {100 * share:.0f}% of pages"
        if r["silhouette"] is not None:
            title += f"\n(silhouette {r['silhouette']:.2f} among them)"
        panels.append((title, r["labels"], r["top_labels"]))
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.9))
        axes = np.atleast_1d(axes)
        for ax, (title, labels, order) in zip(axes, panels):
            cmap, colors = _color_map(labels, order)
            gray = np.array([c == OTHER for c in colors])
            # gray first, so the colored labels are drawn on top
            ax.scatter(xy[gray, 0], xy[gray, 1], s=5, c=OTHER, alpha=0.5, linewidths=0)
            ax.scatter(xy[~gray, 0], xy[~gray, 1], s=6, c=[c for c in colors if c != OTHER],
                       alpha=0.8, linewidths=0)
            for l, c in cmap.items():
                ax.scatter([], [], c=[c], s=25, label=f"{l[:26]} ({labels.count(l)})")
            if order is not None:
                _, n_other, n_none = _label_counts(labels)
                if n_other:
                    ax.scatter([], [], c=[OTHER], s=25, label=f"other labels ({n_other})")
                if n_none:
                    ax.scatter([], [], c=[OTHER], s=25, label=f"no label ({n_none})")
            ax.legend(fontsize=7.5, loc="best", frameon=True, markerscale=1.1)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title, fontsize=11)
        fig.suptitle(f"{name}: {res['n_pages']} page URLs (t-SNE)", fontsize=14, weight="bold")
        save_figure(fig, out_dir, f"in_domain_{name}")


def write_in_domain_tables(results, out_dir):
    rows = []
    for name, res in results.items():
        for rule, r in res["rules"].items():
            rows.append([name, rule, res["n_pages"], f"{100 * r['coverage']:.0f}%",
                         f"{100 * _label_counts(r['labels'])[0]:.0f}%",
                         "—" if r["silhouette"] is None else f"{r['silhouette']:.3f}",
                         "—" if r["within_label_cos"] is None else f"{r['within_label_cos']:.3f}",
                         f"{r['overall_cos']:.3f}", ", ".join(r["top_labels"][:6])])
    write_markdown(["Group", "Label rule", "Pages", "Labeled", "In top labels", "Silhouette (top labels)",
                    "Within-label cos", "Overall cos", "Top labels"], rows,
                   os.path.join(out_dir, "in_domain_summary.md"),
                   align_right=[False, False, True, True, True, True, True, True, False])
    lines = []
    for name, res in results.items():
        lines.append(f"## {name}\n")
        for c, urls in res["cluster_exemplars"].items():
            lines.append(f"**Cluster {c}** ({res['cluster'].count(int(c))} pages)\n")
            lines += [f"- {u}" for u in urls]
            lines.append("")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "in_domain_cluster_exemplars.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def write_summary(summary_rows, out_dir):
    write_markdown(["Statistic", "Value"], summary_rows, os.path.join(out_dir, "summary.md"),
                   align_right=[False, True])
