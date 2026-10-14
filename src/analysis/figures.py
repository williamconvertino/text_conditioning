"""Figure generation for training curves and conditioning comparisons (README
"Training Curves" / "Presentation-Ready Outputs"). Consumes only saved
metrics.jsonl files (src/analysis/curves.py), never checkpoints, so figures can
be regenerated at any time via scripts/make_figures.py.

Every comparison has both a loss and a perplexity version (same data, same
layout, different y-metric) -- see the *_loss/*_ppl pairs below.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.analysis.curves import load_run_config, load_run_metrics, smooth
from src.analysis.tables import DOMAIN_MODE_LABELS, load_final_results
from src.utils.paths import final_conditioning_path

DISPLAY_NAMES = {
    "baseline": "Baseline",
    "meco": "MeCo",
    "meco_cooldown": "MeCo (Cooldown)",
    "text_conditioned": "Text Conditioning",
    "text_conditioned_param_matched": "Text Conditioning (Param Matched)",
    "text_conditioned_cooldown": "Text Conditioning (Cooldown)",
    "text_conditioned_domain_mean": "Text Conditioning (Domain-Mean)",
    "text_conditioned_domain_random": "Text Conditioning (Same-Domain)",
    "text_conditioned_url": "URL-Encoded Conditioning",
    "text_conditioned_url_cooldown": "URL-Encoded Conditioning (Cooldown)",
    "text_conditioned_url_full": "URL-Encoded Conditioning (Full URL)",
    "text_conditioned_url_full_strong": "URL-Encoded Conditioning (Full URL, Strong)",
}
# configs/data/*.yaml corpora other than the default FineWeb, by `dataset`
DATASET_LABELS = {"fineweb_edu": "FineWeb-Edu"}
# their run-dir tags (scripts/train.py appends _<run_tag> before _<model_size>)
DATA_RUN_TAGS = ("fwedu",)

# Run dirs are named <experiment>_<model_size> (scripts/train.py), e.g.
# outputs/checkpoints/baseline_medium -- strip a known model_size suffix before looking up
# DISPLAY_NAMES, then note the size, so plots/tables stay readable regardless
# of which model_size an experiment was trained at.
MODEL_SIZES = ("small", "medium", "full", "1.5b", "3b", "8b")

# Column-name prefix (before "_ppl"/"_loss"), legend label, and per-run
# conditioning-curve figure stem (plot_conditioning_curve below) for each
# experiment family's "conditioned" eval column -- MeCo conditions on a
# prefix of the document itself, text conditioning on a (possibly synthetic)
# text summary. Both support a conditioning_schedule (src/training/
# schedules.py); the figure stem is shared across every text_conditioned*
# family (pre-existing "text_conditioning_*" filenames), one per this dict's
# distinct family key otherwise. The domain-conditioned families' conditioned
# column is the condition they're trained under (in-domain partner documents,
# README "Domain-Conditioned Training"); they also log self_conditioned_*,
# which plot_conditioning_curve adds as a third line. The URL-encoded
# families' conditioned column is the encoded-URL condition (README
# "URL-Encoded Conditioning"); they never see document text, so have no
# self-conditioned column.
CONDITIONED_PREFIXES = {
    "meco": ("meco_conditioned", "MeCo-conditioned", "meco_conditioning"),
    "meco_cooldown": ("meco_conditioned", "MeCo-conditioned", "meco_conditioning"),
    "text_conditioned": ("self_conditioned", "self-conditioned", "text_conditioning"),
    "text_conditioned_param_matched": ("self_conditioned", "self-conditioned", "text_conditioning"),
    "text_conditioned_cooldown": ("self_conditioned", "self-conditioned", "text_conditioning"),
    "text_conditioned_domain_mean": ("domain_conditioned", "domain-mean-conditioned",
                                     "text_domain_mean_conditioning"),
    "text_conditioned_domain_random": ("domain_conditioned", "same-domain-conditioned",
                                       "text_domain_random_conditioning"),
    "text_conditioned_url": ("url_conditioned", "URL-conditioned", "text_url_conditioning"),
    "text_conditioned_url_cooldown": ("url_conditioned", "URL-conditioned",
                                      "text_url_cooldown_conditioning"),
    "text_conditioned_url_full": ("url_conditioned", "full-URL-conditioned",
                                  "text_url_full_conditioning"),
}

# Fixed y-axis caps used by every loss/ppl figure -- well above the converged
# range for any run so far, but well below the early-training blowup (loss
# ~11 / ppl in the hundreds), so plots stay zoomed in on the region that's
# actually informative to compare.
LOSS_YMAX = 4
PPL_YMAX = 50

# Presentation style: matplotlib's built-in seaborn-flavored theme (clean
# white background, light gridlines, standard fonts/sizing) plus a couple of
# small overrides -- deliberately not a lot of hand-tuned rcParams, so plots
# stay looking like "normal" seaborn/matplotlib output rather than a bespoke
# theme. `plt.style.context` accepts a style name followed by a dict of
# overrides applied on top of it.
STYLE = [
    "seaborn-v0_8-whitegrid",
    {
        "figure.figsize": (7.5, 4.8),
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.titlepad": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 10.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 2.2,
        "lines.markeredgewidth": 0,
    },
]

# Okabe-Ito colorblind-safe palette. Colors are assigned per *experiment
# family* (not per run) so e.g. MeCo is always the same blue across every
# figure, regardless of run order or model size.
BASE_COLORS = {
    "baseline": "#555555",
    "meco": "#0072B2",
    "meco_cooldown": "#56B4E9",
    "text_conditioned": "#D55E00",
    "text_conditioned_param_matched": "#009E73",
    "text_conditioned_cooldown": "#CC79A7",
    "text_conditioned_domain_mean": "#E69F00",
    "text_conditioned_domain_random": "#882255",
    # URL-encoded = vector-space MeCo: indigo, dark/light like meco/meco_cooldown
    "text_conditioned_url": "#332288",
    "text_conditioned_url_cooldown": "#8878D0",
    "text_conditioned_url_full": "#44AA99",
    "text_conditioned_url_full_strong": "#117733",
}
_FALLBACK_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def base_name(run_name):
    """Strip a trailing _<model_size> suffix, then a data tag, e.g.
    "meco_medium" -> "meco", "baseline_fwedu_1.5b" -> "baseline"."""
    for size in MODEL_SIZES:
        if run_name.endswith(f"_{size}"):
            run_name = run_name[: -(len(size) + 1)]
            break
    for tag in DATA_RUN_TAGS:
        if run_name.endswith(f"_{tag}"):
            return run_name[: -(len(tag) + 1)]
    return run_name


def schedule_label(cfg):
    """Human-readable conditioning_schedule summary for a run's resolved_config
    (e.g. "p=0.7", "linear cooldown 0.9→0.1 @0.7-1.0"), or None for runs whose
    variant doesn't schedule conditioning at all (currently "text" and "meco",
    see src/training/schedules.py), runs with no saved config, or a run left
    at the always-conditioned default (constant p_condition=1.0 -- nothing
    schedule-related to distinguish it by). Lets a p_condition sweep (or a
    mix of schedule kinds) show up distinctly in figure legends instead of
    every run in a family reading identically."""
    if not cfg or cfg.get("variant") not in ("text", "meco"):
        return None
    sched = cfg.get("conditioning_schedule") or {}
    kind = sched.get("kind", "constant")

    def fmt(x):
        return f"{x:g}" if isinstance(x, (int, float)) else str(x)

    if kind == "constant":
        p = sched.get("p_condition", 1.0)
        return None if p >= 1.0 else f"p={fmt(p)}"
    if kind in ("linear_cooldown", "cosine_cooldown"):
        shape = "linear" if kind == "linear_cooldown" else "cosine"
        return (f"{shape} cooldown {fmt(sched.get('p_start'))}→{fmt(sched.get('p_end'))} "
                f"@{fmt(sched.get('cooldown_start_frac'))}-{fmt(sched.get('cooldown_end_frac'))}")
    if kind == "step_cooldown":
        return (f"step cooldown {fmt(sched.get('p_start'))}→{fmt(sched.get('p_end'))} "
                f"@{fmt(sched.get('step_frac'))}")
    return kind


def schedule_sort_key(cfg):
    """A numeric conditioning-strength proxy used to order/shade runs within
    an experiment family (see _resolve_runs) -- the constant p_condition value,
    or a cooldown schedule's starting p_condition. None when not applicable,
    so those runs just keep run-list order instead of being sorted."""
    if not cfg:
        return None
    sched = cfg.get("conditioning_schedule") or {}
    kind = sched.get("kind", "constant")
    if kind == "constant":
        return sched.get("p_condition")
    return sched.get("p_start")


def _run_base(run_name, cfg):
    """The experiment family a run belongs to, e.g. "text_conditioned" --
    preferably the config's own `name` field (set once from the experiment
    yaml's basename and untouched by any --p-condition/--conditioning-schedule
    run-dir tag, see scripts/train.py), falling back to stripping a
    known _<model_size> suffix off the run dir name when no resolved_config.yaml
    is available."""
    if cfg and cfg.get("name"):
        return cfg["name"]
    return base_name(run_name)


def display_name(run_name, cfg=None):
    if cfg and cfg.get("name"):
        base = cfg["name"]
        label = DISPLAY_NAMES.get(base, base)
        size = cfg.get("model_size")
    else:
        base = base_name(run_name)
        label = DISPLAY_NAMES.get(base, run_name)
        size = run_name.rsplit("_", 1)[-1]
    extras = [size] if size in MODEL_SIZES else []
    if cfg and cfg.get("dataset") in DATASET_LABELS:
        extras.append(DATASET_LABELS[cfg["dataset"]])
    sched = schedule_label(cfg)
    if sched:
        extras.append(sched)
    return f"{label} ({', '.join(extras)})" if extras else label


def get_color(run_name, fallback_index=0):
    base = base_name(run_name)
    if base in BASE_COLORS:
        return BASE_COLORS[base]
    return _FALLBACK_PALETTE[fallback_index % len(_FALLBACK_PALETTE)]


def _resolve_runs(run_dirs):
    """Pre-pass shared by every plot_* function below: load each run's name +
    resolved_config.yaml once, and assign a display label + color.

    Colors are assigned per *experiment family* as before (get_color), but
    when a family has more than one run in this particular figure -- e.g. a
    conditioning_schedule sweep at several p_condition values -- those runs
    are shaded from light to full-saturation across that family's base color
    (ordered by schedule_sort_key, so a higher p_condition/starting
    p_condition reads as more saturated) instead of all overlapping in the
    exact same color."""
    descs = []
    for i, run_dir in enumerate(run_dirs):
        name = os.path.basename(run_dir.rstrip("/"))
        cfg = load_run_config(run_dir)
        base = _run_base(name, cfg)
        descs.append({
            "run_dir": run_dir, "name": name, "cfg": cfg, "base": base,
            "label": display_name(name, cfg), "sort_key": schedule_sort_key(cfg),
            "color": BASE_COLORS.get(base, get_color(name, i)),
        })

    groups = {}
    for d in descs:
        groups.setdefault(d["base"], []).append(d)

    for base, members in groups.items():
        base_color = BASE_COLORS.get(base)
        if len(members) < 2 or base_color is None:
            continue
        ordered = sorted(members, key=lambda d: (d["sort_key"] is None, d["sort_key"]))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("family", ["#FFFFFF", base_color])
        n = len(ordered)
        for rank, d in enumerate(ordered):
            frac = 0.45 + 0.55 * (rank / max(1, n - 1))
            d["color"] = matplotlib.colors.to_hex(cmap(frac))
    return descs


def _metric_label(metric):
    return "Perplexity" if metric == "ppl" else "Loss"


def _format_token_axis(ax):
    """Tokens are counted in the billions here, and matplotlib's default
    scientific-notation offset ("1e9" tucked in the corner) is more confusing
    than helpful -- show plain "0.5", "1.0", ... billion-token ticks instead."""
    ax.set_xlabel("Tokens (billions)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e9:g}"))


def _crop_to_cap(ax, series, ymax):
    """series: list of (x, y) pandas Series pairs already plotted. When ymax
    clips off the (much larger) early-training values -- perplexity in
    particular can start in the tens of thousands, dwarfing the ymax=50 cap --
    two things need fixing up:

    1. matplotlib's default y-autoscale margin is computed from that huge
       early range, so a naive `ax.set_ylim(top=ymax)` leaves `bottom` deep
       in negative territory (empty space filling most of the figure).
    2. the x-axis otherwise still spans token 0, so the entire (capped-off,
       invisible) early region is blank space on the left.

    Both are fixed by deriving xlim/ylim only from the portion of each
    series that's actually within the cap, rather than from the full data
    range matplotlib autoscaled to before we knew about the cap."""
    if ymax is None:
        return
    crossings, visible_mins = [], []
    for x, y in series:
        visible = y[y <= ymax]
        if not visible.empty:
            crossings.append(x[y <= ymax].min())
            visible_mins.append(visible.min())
    if crossings:
        ax.set_xlim(left=min(crossings))
    if visible_mins:
        bottom = min(visible_mins)
        pad = max((ymax - bottom) * 0.05, 1e-6)
        ax.set_ylim(bottom=max(0, bottom - pad), top=ymax)
    else:
        ax.set_ylim(top=ymax)


def save_figure(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_training(run_dirs, out_dir, metric="loss", smooth_window=20, ymax=None):
    """Training loss/perplexity for every run. Only raw "loss" is logged per
    training step, so the perplexity version is derived as exp(loss)."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots()
        plotted = []
        for d in _resolve_runs(run_dirs):
            df = load_run_metrics(d["run_dir"])
            tr = df[df["split"] == "train"]
            if tr.empty:
                continue
            raw = tr["loss"] if metric == "loss" else np.exp(tr["loss"])
            values = smooth(raw, smooth_window)
            ax.plot(tr["tokens"], values, color=d["color"], label=d["label"])
            plotted.append((tr["tokens"], values))
        ax.set_ylabel(f"Train {_metric_label(metric).lower()}")
        ax.set_title(f"Training {_metric_label(metric)}")
        if ymax is not None:
            _crop_to_cap(ax, plotted, ymax)
        _format_token_axis(ax)
        ax.legend()
        save_figure(fig, out_dir, f"training_{metric}")


def plot_training_loss(run_dirs, out_dir, smooth_window=20, ymax=LOSS_YMAX):
    plot_training(run_dirs, out_dir, metric="loss", smooth_window=smooth_window, ymax=ymax)


def plot_training_ppl(run_dirs, out_dir, smooth_window=20, ymax=PPL_YMAX):
    plot_training(run_dirs, out_dir, metric="ppl", smooth_window=smooth_window, ymax=ymax)


def plot_validation_unconditioned(run_dirs, out_dir, metric="ppl", ymax=None):
    """Unconditioned validation loss/perplexity across runs -- the
    unconditioned-only counterpart to plot_validation_conditioned below.
    Falls back to the plain "loss"/"ppl" column for any run that doesn't
    have an unconditioned_* column (e.g. the baseline, which is always
    unconditioned)."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots()
        plotted = []
        for d in _resolve_runs(run_dirs):
            df = load_run_metrics(d["run_dir"])
            va = df[df["split"] == "val"]
            field = f"unconditioned_{metric}"
            col = field if field in va.columns and va[field].notna().any() else metric
            v = va.dropna(subset=[col])
            if v.empty:
                continue
            ax.plot(v["tokens"], v[col], marker="o", markersize=4, markeredgecolor="white",
                    color=d["color"], label=d["label"])
            plotted.append((v["tokens"], v[col]))
        ax.set_ylabel(f"Validation {_metric_label(metric).lower()}")
        ax.set_title(f"Unconditioned Validation {_metric_label(metric)}")
        if ymax is not None:
            _crop_to_cap(ax, plotted, ymax)
        _format_token_axis(ax)
        ax.legend()
        save_figure(fig, out_dir, f"validation_unconditioned_{metric}")


def plot_validation_unconditioned_ppl(run_dirs, out_dir, ymax=PPL_YMAX):
    plot_validation_unconditioned(run_dirs, out_dir, metric="ppl", ymax=ymax)


def plot_validation_unconditioned_loss(run_dirs, out_dir, ymax=LOSS_YMAX):
    plot_validation_unconditioned(run_dirs, out_dir, metric="loss", ymax=ymax)


def plot_validation_conditioned(run_dirs, out_dir, metric="ppl", ymax=None, include_baseline=True):
    """Conditioned validation loss/perplexity across runs -- the
    conditioned-only counterpart to plot_validation_unconditioned above,
    using each run's experiment-family conditioned column (CONDITIONED_PREFIXES:
    self_conditioned_* for text conditioning, meco_conditioned_* for MeCo).
    The baseline, which has nothing to condition on, is drawn as a dashed,
    "(unconditioned)"-labeled reference line from its ordinary validation
    curve (include_baseline=False leaves it out); other runs from a family
    with no conditioned column are skipped rather than plotted as flat lines."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots()
        plotted = []
        for d in _resolve_runs(run_dirs):
            prefix, _, _ = CONDITIONED_PREFIXES.get(d["base"], (None, None, None))
            df = load_run_metrics(d["run_dir"])
            va = df[df["split"] == "val"]
            if prefix is None:
                if not (include_baseline and d["base"] == "baseline") or metric not in va.columns:
                    continue
                v = va.dropna(subset=[metric])
                if v.empty:
                    continue
                ax.plot(v["tokens"], v[metric], linestyle="--", marker="o", markersize=4,
                        markeredgecolor="white", color=d["color"],
                        label=(d["label"][:-1] + ", unconditioned)" if d["label"].endswith(")")
                               else f"{d['label']} (unconditioned)"))
                plotted.append((v["tokens"], v[metric]))
                continue
            col = f"{prefix}_{metric}"
            if col not in va.columns:
                continue
            v = va.dropna(subset=[col])
            if v.empty:
                continue
            ax.plot(v["tokens"], v[col], marker="^", markersize=4, markeredgecolor="white",
                    color=d["color"], label=d["label"])
            plotted.append((v["tokens"], v[col]))
        if not plotted:
            raise ValueError("no runs with a conditioned validation column found")
        ax.set_ylabel(f"Validation {_metric_label(metric).lower()}")
        ax.set_title(f"Conditioned Validation {_metric_label(metric)}")
        if ymax is not None:
            _crop_to_cap(ax, plotted, ymax)
        _format_token_axis(ax)
        ax.legend()
        save_figure(fig, out_dir, f"validation_conditioned_{metric}")


def plot_validation_conditioned_ppl(run_dirs, out_dir, ymax=PPL_YMAX, include_baseline=True):
    plot_validation_conditioned(run_dirs, out_dir, metric="ppl", ymax=ymax, include_baseline=include_baseline)


def plot_validation_conditioned_loss(run_dirs, out_dir, ymax=LOSS_YMAX, include_baseline=True):
    plot_validation_conditioned(run_dirs, out_dir, metric="loss", ymax=ymax, include_baseline=include_baseline)


def plot_conditioning_curve(run_dir, out_dir, metric="ppl", ymax=None):
    """Conditioned vs unconditioned loss/perplexity over training, for a
    single run from any experiment family with a CONDITIONED_PREFIXES entry
    (currently MeCo and every text_conditioned* family) -- generalizes what
    used to be text-conditioning-only (plot_text_conditioning) now that MeCo
    also has a conditioning_schedule (src/training/schedules.py). Figure stem
    comes from CONDITIONED_PREFIXES so every text_conditioned* family keeps
    writing to the pre-existing "text_conditioning_{metric}" filename."""
    name = os.path.basename(run_dir.rstrip("/"))
    cfg = load_run_config(run_dir)
    base = _run_base(name, cfg)
    prefix, cond_label, stem = CONDITIONED_PREFIXES.get(base, (None, None, None))
    if prefix is None:
        raise ValueError(f"{run_dir}'s experiment family ({base!r}) has no conditioned "
                         f"eval column -- is this a MeCo or text-conditioned run?")

    with plt.style.context(STYLE):
        fig, ax = plt.subplots()
        df = load_run_metrics(run_dir)
        cond_col, uncond_col = f"{prefix}_{metric}", f"unconditioned_{metric}"
        va = df[df["split"] == "val"].dropna(subset=[cond_col, uncond_col])
        if va.empty:
            raise ValueError(f"{run_dir} has no {cond_col}/{uncond_col} logged")
        color = get_color(name)
        ax.plot(va["tokens"], va[uncond_col], color=color, linestyle="-",
                 marker="o", markersize=4, markeredgecolor="white", label="Unconditioned")
        ax.plot(va["tokens"], va[cond_col], color=color, linestyle="--",
                 marker="^", markersize=4, markeredgecolor="white",
                 label=cond_label[:1].upper() + cond_label[1:])
        plotted = [(va["tokens"], va[uncond_col]), (va["tokens"], va[cond_col])]
        # Domain-conditioned runs also log self-conditioning (the encoder
        # seeing the target document itself), which they never train on.
        self_col = f"self_conditioned_{metric}"
        if prefix != "self_conditioned" and self_col in va.columns and va[self_col].notna().any():
            sc = va.dropna(subset=[self_col])
            ax.plot(sc["tokens"], sc[self_col], color=color, linestyle=":",
                     marker="s", markersize=3.5, markeredgecolor="white", label="Self-conditioned")
            plotted.append((sc["tokens"], sc[self_col]))
        ax.set_ylabel(f"Validation {_metric_label(metric).lower()}")
        ax.set_title(display_name(name, cfg))
        if ymax is not None:
            _crop_to_cap(ax, plotted, ymax)
        _format_token_axis(ax)
        ax.legend()
        save_figure(fig, out_dir, f"{stem}_{metric}")


def plot_conditioning_curve_ppl(run_dir, out_dir, ymax=PPL_YMAX):
    plot_conditioning_curve(run_dir, out_dir, metric="ppl", ymax=ymax)


def plot_conditioning_curve_loss(run_dir, out_dir, ymax=LOSS_YMAX):
    plot_conditioning_curve(run_dir, out_dir, metric="loss", ymax=ymax)


def plot_validation_both(run_dirs, out_dir, metric="ppl", ymax=None):
    """All models & conditionings, overlaid: unconditioned vs. conditioned
    (MeCo- or self-conditioned) validation loss/perplexity for every run --
    the combined counterpart to plot_validation_unconditioned/
    plot_validation_conditioned above (same two series, one figure).
    Color encodes experiment family and stays consistent with the other
    figures; line style encodes the eval condition (solid = unconditioned,
    dashed = conditioned), so e.g. MeCo, Text Conditioning (conditioned
    eval), and Text Conditioning (unconditioned eval) are each visually
    distinct at a glance."""
    with plt.style.context(STYLE):
        fig, ax = plt.subplots()
        plotted = []
        for d in _resolve_runs(run_dirs):
            base, color, label = d["base"], d["color"], d["label"]
            df = load_run_metrics(d["run_dir"])
            va = df[df["split"] == "val"]

            uncond_field = f"unconditioned_{metric}"
            uncond_col = (uncond_field if uncond_field in va.columns
                          and va[uncond_field].notna().any() else metric)
            uncond = va.dropna(subset=[uncond_col])
            if not uncond.empty:
                ax.plot(uncond["tokens"], uncond[uncond_col], color=color, linestyle="-",
                        marker="o", markersize=4, markeredgecolor="white",
                        label=f"{label} — unconditioned")
                plotted.append((uncond["tokens"], uncond[uncond_col]))

            prefix, cond_label, _ = CONDITIONED_PREFIXES.get(base, (None, None, None))
            cond_col = f"{prefix}_{metric}" if prefix else None
            if cond_col and cond_col in va.columns:
                cond = va.dropna(subset=[cond_col])
                if not cond.empty:
                    ax.plot(cond["tokens"], cond[cond_col], color=color, linestyle="--",
                            marker="^", markersize=4, markeredgecolor="white",
                            label=f"{label} — {cond_label}")
                    plotted.append((cond["tokens"], cond[cond_col]))

        if not plotted:
            raise ValueError("no runs with plottable validation data found")

        ax.set_ylabel(f"Validation {_metric_label(metric).lower()}")
        ax.set_title("All Models & Conditionings")
        if ymax is not None:
            _crop_to_cap(ax, plotted, ymax)
        _format_token_axis(ax)
        ax.legend(loc="upper right", framealpha=0.9)
        save_figure(fig, out_dir, f"validation_both_{metric}")


def plot_validation_both_ppl(run_dirs, out_dir, ymax=PPL_YMAX):
    plot_validation_both(run_dirs, out_dir, metric="ppl", ymax=ymax)


def plot_validation_both_loss(run_dirs, out_dir, ymax=LOSS_YMAX):
    plot_validation_both(run_dirs, out_dir, metric="loss", ymax=ymax)


def plot_final_conditioning(run_dirs, out_dir, metric="ppl"):
    """Grouped bar chart of the final (post-training) conditioning-mode
    comparison from each run's final_conditioning.json (src/eval/conditioning.py,
    scripts/evaluate.py --mode final): one group per conditioning mode (No
    Cond. / Self / Same-Domain / Domain-Mean / Cross-Domain), one bar per
    model within each group. Colors match every other figure (get_color /
    _resolve_runs).

    Always reads the "top_domains" (Top-N) scope, never "overall": that's the
    one restricted document set every mode is directly comparable on (README
    "Final Evaluation Table" -- same_domain/domain_mean/cross_domain only
    ever have a Top-N value in the first place). Runs with no
    final_conditioning.json, or whose final_conditioning.json has no
    top_domains data (no domain had enough validation documents), are
    skipped entirely rather than mixed in at a different scope."""
    idx = 1 if metric == "ppl" else 0
    bars = []
    for d in _resolve_runs(run_dirs):
        path = final_conditioning_path(d["run_dir"])
        if not os.path.exists(path):
            continue
        top_domains = load_final_results(path).get("top_domains")
        if top_domains is None:
            continue
        values = {mode: val[idx] for mode, _ in DOMAIN_MODE_LABELS
                  if (val := top_domains["aggregate"].get(mode)) is not None}
        if values:
            bars.append((d["label"], d["color"], values))
    if not bars:
        raise ValueError("no runs with a top_domains final_conditioning.json found")

    modes = [mode for mode, _ in DOMAIN_MODE_LABELS if any(mode in values for _, _, values in bars)]
    mode_ticklabels = [label.replace(" PPL", "") for mode, label in DOMAIN_MODE_LABELS if mode in modes]

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(modes))
        width = 0.8 / len(bars)
        for i, (label, color, values) in enumerate(bars):
            xs = [x[j] - 0.4 + width / 2 + i * width for j, mode in enumerate(modes) if mode in values]
            ys = [values[mode] for mode in modes if mode in values]
            ax.bar(xs, ys, width=width * 0.92, color=color, label=label,
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(mode_ticklabels)
        ax.set_ylabel(f"Validation {_metric_label(metric).lower()} (Top-N)")
        ax.set_title("Final Conditioning Evaluation")
        ax.legend()
        save_figure(fig, out_dir, f"final_conditioning_{metric}")


def plot_final_conditioning_ppl(run_dirs, out_dir):
    plot_final_conditioning(run_dirs, out_dir, metric="ppl")


def plot_final_conditioning_loss(run_dirs, out_dir):
    plot_final_conditioning(run_dirs, out_dir, metric="loss")
