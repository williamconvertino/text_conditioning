"""Where everything under outputs/ lives (README "Outputs Layout"). Every
script and library module resolves output paths through here, so the layout
is defined in one place.

    outputs/
      checkpoints/<run>/            (i) one dir per training run: ckpt_*.pt,
                                        checkpoints/ snapshots, metrics.jsonl,
                                        resolved_config.yaml -- what
                                        scripts/train.py writes
      experiments/<size>/           (ii) intermediate results, by model size
        eval/<run>/final_conditioning.json
        benchmarks/<run>/[<ckpt label>/]<tag>/results.json
        finetune/<run>/<ckpt label>/<method>/<task>/seed<K>/
        url_analysis/<ckpt label>/  corpus.pt, library.pt, results/*.json, ...
        steering/<ckpt label>/      task_vectors/, mixtures/
      results/<size>/               (iii) presentation-ready figures/tables
        training/figures/           training curves (scripts/make_figures.py)
        final_eval/{figures,tables}/
        benchmarks/{figures,tables}/
        comparison/<ckpt label>/{figures,tables}/
        finetune/<ckpt label>/{figures,tables}/
        steering/<ckpt label>/{figures,tables}/
        url_analysis/<ckpt label>/{figures,tables}/
      archive/                      superseded runs and tables, kept for reference

<size> is a run's model_size (configs/model/*.yaml: medium, 1.5b, 3b, ...), so
each scale's results stay apart.

The outputs root defaults to <repo>/outputs; TC_OUTPUTS_DIR overrides it (e.g.
for a smoke test), and set_outputs_dir() does so in-process (tests/conftest.py).
Read the directories through this module at call time (paths.CHECKPOINTS_DIR,
not `from ... import CHECKPOINTS_DIR`) so an override is seen everywhere.
"""

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def set_outputs_dir(path):
    global OUTPUTS_DIR, CHECKPOINTS_DIR, EXPERIMENTS_DIR, RESULTS_DIR, ARCHIVE_DIR
    OUTPUTS_DIR = os.path.abspath(path)
    CHECKPOINTS_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
    EXPERIMENTS_DIR = os.path.join(OUTPUTS_DIR, "experiments")
    RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")
    ARCHIVE_DIR = os.path.join(OUTPUTS_DIR, "archive")


set_outputs_dir(os.environ.get("TC_OUTPUTS_DIR") or os.path.join(ROOT, "outputs"))

# Known model_size values, also the run-dir name suffix (<experiment>_<size>,
# scripts/train.py), used when a run has no resolved_config.yaml.
MODEL_SIZES = ("small", "medium", "full", "1.5b", "3b", "8b")


def run_name(run_dir):
    return os.path.basename(os.path.normpath(run_dir))


def resolve_run_dir(run_dir):
    """A run dir as given (absolute, or relative to the cwd or the repo root),
    else the run of the same name under outputs/checkpoints/ -- so paths
    recorded before the outputs/ reorganization (e.g. outputs/baseline_medium
    in old results.json files) still resolve."""
    for cand in (run_dir, os.path.join(ROOT, run_dir)):
        if os.path.isdir(cand):
            return cand
    return os.path.join(CHECKPOINTS_DIR, run_name(run_dir))


def model_size(run_dir):
    """The run's model_size: its resolved_config.yaml's, else the run-dir name's
    _<size> suffix. Raises ValueError if neither is known."""
    cfg_path = os.path.join(resolve_run_dir(run_dir), "resolved_config.yaml")
    if os.path.exists(cfg_path):
        size = (yaml.safe_load(open(cfg_path)) or {}).get("model_size")
        if size:
            return str(size)
    name = run_name(run_dir)
    for size in MODEL_SIZES:
        if name.endswith(f"_{size}"):
            return size
    raise ValueError(f"cannot tell the model size of {run_dir}: no model_size in its "
                     f"resolved_config.yaml and no _<size> suffix in its name")


def comparison_size(comparison_cfg):
    """Model size of a configs/comparison*.yaml run set: its `model_size` key,
    else the (single) size shared by its runs."""
    if comparison_cfg.get("model_size"):
        return str(comparison_cfg["model_size"])
    sizes = {model_size(d) for d in comparison_cfg["runs"].values()}
    if len(sizes) != 1:
        raise ValueError(f"comparison runs span several model sizes ({sorted(sizes)}); "
                         f"set model_size in the comparison config")
    return sizes.pop()


def experiments_dir(size, *parts):
    return os.path.join(EXPERIMENTS_DIR, size, *parts)


def results_dir(size, *parts):
    return os.path.join(RESULTS_DIR, size, *parts)


def final_conditioning_path(run_dir):
    """scripts/evaluate.py --mode final's output for a run."""
    return experiments_dir(model_size(run_dir), "eval", run_name(run_dir), "final_conditioning.json")


def benchmarks_dir(run_dir):
    """Root of a run's lm-eval results (src/benchmarks/results.py)."""
    return experiments_dir(model_size(run_dir), "benchmarks", run_name(run_dir))


def discover_runs(sizes=None, require="metrics.jsonl"):
    """Every run dir under outputs/checkpoints/ that has `require` (relative to
    the run dir; None for any), optionally restricted to some model sizes."""
    if not os.path.isdir(CHECKPOINTS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(CHECKPOINTS_DIR)):
        d = os.path.join(CHECKPOINTS_DIR, name)
        if not os.path.isdir(d) or (require and not os.path.isfile(os.path.join(d, require))):
            continue
        if sizes is not None:
            try:
                if model_size(d) not in sizes:
                    continue
            except ValueError:
                continue
        out.append(d)
    return out


def group_by_size(run_dirs):
    """{size: [run_dir, ...]} in first-seen order."""
    groups = {}
    for d in run_dirs:
        groups.setdefault(model_size(d), []).append(d)
    return groups
