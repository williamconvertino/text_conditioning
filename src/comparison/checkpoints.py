"""Pick the furthest checkpoint a set of runs all share, so a comparison
between them (configs/comparison.yaml) never gives one run more training than
another.

Candidates per run (src/training/trainer.py writes both):
  checkpoints/ckpt_step<step>_tok<N>M.pt   standing eval snapshots, one per
                                           ckpt_save_interval_tokens tokens
  ckpt_last.pt                             the resume checkpoint (its step is
                                           read from the file itself)

Resolution, in order:
  1. The largest *step* every run has a checkpoint at (in practice the final
     step, once every run has finished: ckpt_last.pt at max_steps).
  2. Otherwise the largest token milestone every run has a snapshot for.
     Snapshots are written the first step past each milestone, so the runs'
     steps there differ by a handful (e.g. 19813 vs 19825); the result records
     each run's exact step and token count so that gap is visible.

A comparison config can instead pin the checkpoint explicitly with
`checkpoint: tok<N>M` (that token milestone's snapshot in every run) or
`checkpoint: step<N>` -- e.g. to evaluate one run of a scale now, at a
milestone the other runs will reach later, so their results line up.

ckpt_best.pt is never a candidate: it is chosen per run by that run's own val
loss, so it is not a matched training budget.

Every script resolves this itself at startup (shared_checkpoint), so jobs can
be submitted manually in any order. Results are stored under the resolved
label (e.g. step20000), so if a run is still training and the answer moves,
results from different checkpoints land in different directories rather
than mixing -- run the experiments once every run has finished.
"""

import glob
import json
import os
import re
import time

import torch

from src.config import load_config

SNAPSHOT_RE = re.compile(r"ckpt_step(\d+)_tok(\d+)M\.pt$")
DEFAULT_TOKEN_INTERVAL = 250_000_000


def list_checkpoints(run_dir):
    """[{path (relative to run_dir), step, tokens, kind}] for every candidate
    checkpoint in run_dir, sorted by step."""
    out = []
    for path in glob.glob(os.path.join(run_dir, "checkpoints", "ckpt_step*_tok*M.pt")):
        m = SNAPSHOT_RE.search(os.path.basename(path))
        if m:
            out.append({"path": os.path.relpath(path, run_dir), "step": int(m.group(1)),
                        "tokens": int(m.group(2)) * 1_000_000, "kind": "snapshot"})
    last = os.path.join(run_dir, "ckpt_last.pt")
    if os.path.exists(last):
        # mmap: only the pickled metadata is read, not the ~5GB of tensors.
        ckpt = torch.load(last, map_location="cpu", mmap=True, weights_only=False)
        out.append({"path": "ckpt_last.pt", "step": int(ckpt["step"]),
                    "tokens": int(ckpt["tokens_seen"]), "kind": "last"})
    return sorted(out, key=lambda c: (c["step"], c["kind"] == "last"))


def _token_interval(run_dir):
    cfg_path = os.path.join(run_dir, "resolved_config.yaml")
    if os.path.exists(cfg_path):
        return int(load_config(cfg_path).get("ckpt_save_interval_tokens", DEFAULT_TOKEN_INTERVAL))
    return DEFAULT_TOKEN_INTERVAL


def _describe(cands):
    return "\n".join(f"  {name}: " + (", ".join(f"step {c['step']} ({c['tokens'] / 1e6:.0f}M tok, {c['kind']})"
                                               for c in cs) or "no checkpoints")
                     for name, cs in cands.items())


def _pinned(runs, cands, pin):
    """The checkpoint each run has at an explicit `pin` (tok<N>M or step<N>)."""
    m = re.fullmatch(r"(tok|step)(\d+)M?", pin)
    if not m or (m.group(1) == "tok") != pin.endswith("M"):
        raise ValueError(f"checkpoint pin {pin!r} must look like tok<N>M or step<N>")
    chosen = {}
    for name, cs in cands.items():
        if m.group(1) == "step":
            match = [c for c in cs if c["step"] == int(m.group(2))]
        else:
            interval = _token_interval(runs[name])
            milestone = int(m.group(2)) * 1_000_000 / interval
            match = [c for c in cs if c["kind"] == "snapshot" and round(c["tokens"] / interval) == milestone]
        if not match:
            raise ValueError(f"run {name} has no checkpoint at {pin}:\n" + _describe({name: cs}))
        chosen[name] = match[0]
    return chosen


def resolve_shared_checkpoint(runs, pin=None):
    """runs: {name: run_dir}. Returns the shared-checkpoint dict (see module docstring):
    {label, rule, runs: {name: {run_dir, path, step, tokens}}}. Raises
    ValueError, listing every run's candidates, when nothing is shared (or a
    `pin` -- tok<N>M / step<N> -- is missing from some run)."""
    cands = {name: list_checkpoints(run_dir) for name, run_dir in runs.items()}
    if pin:
        chosen = _pinned(runs, cands, pin)
        return {
            "label": pin, "rule": "pinned", "resolved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runs": {name: {"run_dir": runs[name], "path": c["path"], "step": c["step"], "tokens": c["tokens"]}
                     for name, c in chosen.items()},
        }

    step_sets = [{c["step"] for c in cs} for cs in cands.values()]
    shared_steps = set.intersection(*step_sets) if step_sets else set()
    if shared_steps:
        step = max(shared_steps)
        # prefer the snapshot over ckpt_last when both exist at this step
        # (identical weights; the snapshot is smaller to load).
        chosen = {name: next(c for c in cs if c["step"] == step) for name, cs in cands.items()}
        label, rule = f"step{step}", "shared_step"
    else:
        milestones = {}
        for name, cs in cands.items():
            interval = _token_interval(runs[name])
            milestones[name] = {round(c["tokens"] / interval): c for c in cs if c["kind"] == "snapshot"}
        shared = set.intersection(*(set(m) for m in milestones.values())) if milestones else set()
        if not shared:
            raise ValueError("no checkpoint step or token milestone is shared by every run:\n"
                             + _describe(cands))
        m = max(shared)
        chosen = {name: ms[m] for name, ms in milestones.items()}
        interval = _token_interval(next(iter(runs.values())))
        label, rule = f"tok{m * interval // 1_000_000}M", "shared_token_milestone"

    return {
        "label": label,
        "rule": rule,
        "resolved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": {name: {"run_dir": runs[name], "path": c["path"], "step": c["step"], "tokens": c["tokens"]}
                 for name, c in chosen.items()},
    }


def shared_checkpoint(comparison_cfg, root="."):
    """Resolve the shared checkpoint for configs/comparison.yaml's runs, now
    (or the config's `checkpoint` pin). run_dir entries in the result are kept
    as written in the config. Results are kept apart per scale by the
    outputs/{experiments,results}/<size>/ directories (src/utils/paths.py);
    an optional `label_prefix` is still prepended to the label if set."""
    runs = {name: os.path.join(root, d) for name, d in comparison_cfg["runs"].items()}
    shared = resolve_shared_checkpoint(runs, pin=comparison_cfg.get("checkpoint"))
    if comparison_cfg.get("label_prefix"):
        shared["label"] = f"{comparison_cfg['label_prefix']}-{shared['label']}"
    shared["runs"] = {name: {**info, "run_dir": comparison_cfg["runs"][name]}
                      for name, info in shared["runs"].items()}
    return shared


PIN_RE = re.compile(r"(tok\d+M|step\d+)")


def is_pin(s):
    return bool(PIN_RE.fullmatch(str(s)))


def run_checkpoint(run_dir, pin):
    """One run's checkpoint at a tok<N>M / step<N> pin, without a comparison
    config -- e.g. the per-milestone benchmarks of a training trajectory
    (scripts/comparison/make_trajectory.py). Returns (path relative to
    run_dir, {path, step, tokens})."""
    c = _pinned({"run": run_dir}, {"run": list_checkpoints(run_dir)}, pin)["run"]
    return c["path"], {"path": c["path"], "step": c["step"], "tokens": c["tokens"]}


def milestones(run_dir):
    """Every tok<N>M label run_dir has a snapshot for, in training order."""
    interval = _token_interval(run_dir)
    labels = {f"tok{round(c['tokens'] / interval) * interval // 1_000_000}M": c["step"]
              for c in list_checkpoints(run_dir) if c["kind"] == "snapshot"}
    return sorted(labels, key=labels.get)


def checkpoint_for(pin, run_dir):
    """(checkpoint path relative to run_dir, entry) for the shared-checkpoint run
    whose run_dir matches `run_dir` (compared by basename, so
    outputs/checkpoints/x_medium and ./outputs/checkpoints/x_medium/ both match)."""
    key = os.path.basename(os.path.normpath(run_dir))
    for name, info in pin["runs"].items():
        if os.path.basename(os.path.normpath(info["run_dir"])) == key:
            return info["path"], {"name": name, **info}
    raise KeyError(f"{run_dir} is not one of the compared runs "
                   f"({', '.join(i['run_dir'] for i in pin['runs'].values())})")
