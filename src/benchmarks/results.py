"""Save/discover/organize lm-eval-harness benchmark results
(scripts/benchmarks/run_benchmarks.py) so every (run, conditioning)
combination lands in one predictable, greppable place and can be found again
later -- consumed by scripts/benchmarks/plot_benchmarks.py.

Layout (mirrors outputs/<run>/final_conditioning.json for the perplexity
final-conditioning eval):

    outputs/<run_name>/benchmarks/<conditioning_tag>/results.json
"""

import glob
import json
import os
import time

_STDERR_MARKERS = ("_stderr", "alias")


def result_dir(run_dir, tag):
    return os.path.join(run_dir, "benchmarks", tag)


def result_path(run_dir, tag):
    return os.path.join(result_dir(run_dir, tag), "results.json")


def save_result(run_dir, tag, *, run_name, variant, conditioning_kind, conditioning_meta,
                tasks_cfg, lm_eval_results):
    path = result_path(run_dir, tag)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "run_dir": run_dir,
        "run_name": run_name,
        "variant": variant,
        "conditioning_tag": tag,
        "conditioning_kind": conditioning_kind,
        "conditioning_meta": conditioning_meta,
        "tasks_cfg": tasks_cfg,
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


def discover_results(outputs_dir, run_names=None, tags=None):
    """Every outputs/<run>/benchmarks/<tag>/results.json under outputs_dir,
    optionally filtered to specific run directory names / conditioning tags."""
    pattern = os.path.join(outputs_dir, "*", "benchmarks", "*", "results.json")
    records = []
    for path in sorted(glob.glob(pattern)):
        rec = load_result(path)
        if run_names is not None and rec["run_name"] not in run_names:
            continue
        if tags is not None and rec["conditioning_tag"] not in tags:
            continue
        records.append(rec)
    return records


def find_result(outputs_dir, run_name, tag):
    matches = discover_results(outputs_dir, run_names=[run_name], tags=[tag])
    if not matches:
        raise FileNotFoundError(f"no benchmark result for run_name={run_name!r} tag={tag!r} "
                                f"under {outputs_dir} -- run scripts/benchmarks/run_benchmarks.py first")
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
