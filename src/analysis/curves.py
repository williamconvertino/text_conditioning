"""Load metrics.jsonl files into pandas DataFrames for plotting
(src/analysis/figures.py) and table generation (src/analysis/tables.py).
Analysis always reads saved metrics, never checkpoints, so figures can be
regenerated without rerunning anything.
"""

import os

import pandas as pd
import yaml

from src.utils.logging import load_jsonl


def load_run_metrics(run_dir):
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no metrics.jsonl in {run_dir}")
    return pd.DataFrame(load_jsonl(path))


def load_run_config(run_dir):
    """The run's resolved_config.yaml (src/training/trainer.py's `run()` saves
    one per run dir), used by src/analysis/figures.py to label/color runs by
    e.g. their conditioning_schedule. Returns None if a run predates this file
    or was otherwise never trained (figures should still degrade gracefully)."""
    path = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.exists(path):
        return None
    return yaml.safe_load(open(path)) or {}


def smooth(series, window):
    """Simple rolling mean; window <= 1 returns the series unchanged. The raw
    series should always also be kept/plotted -- this is for visual
    presentation only (README "Presentation-Ready Outputs")."""
    if window is None or window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()
