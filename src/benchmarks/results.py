"""Save/discover/organize lm-eval-harness benchmark results
(scripts/benchmarks/run_benchmarks.py) so every (run, conditioning)
combination lands in one predictable, greppable place and can be found again
later -- consumed by scripts/benchmarks/plot_benchmarks.py.

Layout (src/utils/paths.py; next to the perplexity final-conditioning eval's
outputs/experiments/<size>/eval/<run>/final_conditioning.json):

    outputs/experiments/<size>/benchmarks/<run_name>/<conditioning_tag>/results.json

Results for the checkpoint shared by a cross-run comparison
(src/comparison/checkpoints.py) go one level deeper, under its label,
so they never mix with (or overwrite) the default ckpt_best.pt results:

    outputs/experiments/<size>/benchmarks/<run_name>/<ckpt_label>/<conditioning_tag>/results.json
"""

import glob
import json
import os
import time

from src.utils import paths
from src.utils.paths import benchmarks_dir

_STDERR_MARKERS = ("_stderr", "alias")


def result_dir(run_dir, tag, ckpt_label=None):
    if ckpt_label:
        return os.path.join(benchmarks_dir(run_dir), ckpt_label, tag)
    return os.path.join(benchmarks_dir(run_dir), tag)


def result_path(run_dir, tag, ckpt_label=None):
    return os.path.join(result_dir(run_dir, tag, ckpt_label), "results.json")


def save_result(run_dir, tag, *, run_name, variant, conditioning_kind, conditioning_meta,
                tasks_cfg, lm_eval_results, ckpt_label=None, checkpoint=None):
    """checkpoint: optional provenance for the evaluated weights, e.g.
    {"path", "step", "tokens"} of the shared checkpoint."""
    path = result_path(run_dir, tag, ckpt_label)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "run_dir": run_dir,
        "run_name": run_name,
        "variant": variant,
        "conditioning_tag": tag,
        "conditioning_kind": conditioning_kind,
        "conditioning_meta": conditioning_meta,
        "tasks_cfg": tasks_cfg,
        "ckpt_label": ckpt_label,
        "checkpoint": checkpoint,
        "timestamp": time.time(),
        "results": lm_eval_results.get("results", {}),
        "n_shot": lm_eval_results.get("n-shot", {}),
        "n_samples": lm_eval_results.get("n-samples", {}),
        "versions": lm_eval_results.get("versions", {}),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)
    return path


def load_result(path):
    return json.load(open(path))


def discover_results(run_names=None, tags=None, ckpt_label=None, sizes=None):
    """Every outputs/experiments/<size>/benchmarks/<run>/<tag>/results.json
    (or, with ckpt_label, every .../<run>/<ckpt_label>/<tag>/results.json),
    optionally filtered to specific run directory names / conditioning tags /
    model sizes."""
    parts = ["benchmarks", "*"] + ([ckpt_label] if ckpt_label else []) + ["*", "results.json"]
    found = []
    for size in (sizes or ["*"]):
        found += glob.glob(os.path.join(paths.EXPERIMENTS_DIR, size, *parts))
    records = []
    for path in sorted(found):
        rec = load_result(path)
        if run_names is not None and rec["run_name"] not in run_names:
            continue
        if tags is not None and rec["conditioning_tag"] not in tags:
            continue
        records.append(rec)
    return records


def find_result(run_name, tag, ckpt_label=None):
    matches = discover_results(run_names=[run_name], tags=[tag], ckpt_label=ckpt_label)
    if not matches:
        raise FileNotFoundError(f"no benchmark result for run_name={run_name!r} tag={tag!r} "
                                f"under {paths.EXPERIMENTS_DIR} -- run scripts/benchmarks/run_benchmarks.py first")
    return matches[0]


def best_metric(task_results, prefer=("acc_norm,none", "acc,none")):
    """Pick a task's headline metric: the first of `prefer` that's present,
    else the first non-stderr/non-alias metric found. Returns (metric_key, value)
    or (None, None) if nothing usable is present."""
    for key in prefer:
        if key in task_results:
            return key, task_results[key]
    for key, value in task_results.items():
        if key == "alias" or any(m in key for m in _STDERR_MARKERS):
            continue
        if isinstance(value, (int, float)):
            return key, value
    return None, None


def stderr_key_for(metric_key):
    """lm-eval-harness's stderr naming convention: "acc_norm,none" ->
    "acc_norm_stderr,none" (metric name gets "_stderr", the ",<filter>" suffix
    is unchanged) -- see _STDERR_MARKERS above."""
    if "," not in metric_key:
        return metric_key + "_stderr"
    metric, rest = metric_key.split(",", 1)
    return f"{metric}_stderr,{rest}"


def best_metric_with_stderr(task_results, prefer=("acc_norm,none", "acc,none")):
    """best_metric, plus that metric's stderr (None if not present or not
    numeric -- e.g. bootstrapping was disabled). Returns (metric_key, value,
    stderr), all None if nothing usable is present."""
    key, value = best_metric(task_results, prefer=prefer)
    if key is None:
        return None, None, None
    stderr = task_results.get(stderr_key_for(key))
    if not isinstance(stderr, (int, float)):
        stderr = None
    return key, value, stderr


def task_metric_rows(records, prefer=("acc_norm,none", "acc,none")):
    """records (src/benchmarks/results.py's discover_results output) -> long
    format rows [{run_name, conditioning_tag, variant, task, metric, value}, ...]
    for plotting/tabulating (src/benchmarks/plots.py)."""
    rows = []
    for rec in records:
        for task, task_results in rec["results"].items():
            metric, value = best_metric(task_results, prefer=prefer)
            if metric is None:
                continue
            rows.append({
                "run_name": rec["run_name"], "conditioning_tag": rec["conditioning_tag"],
                "variant": rec["variant"], "task": task, "metric": metric, "value": value,
            })
    return rows
