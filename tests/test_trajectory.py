"""Benchmarks across pretraining (src/comparison/trajectory.py,
scripts/comparison/trajectory_jobs.py): per-run milestone pins, the job list,
and the figures built from per-milestone results."""

import os
import sys

import pytest

from src.benchmarks.results import save_result
from src.comparison.checkpoints import milestones, run_checkpoint
from src.comparison.trajectory import collect, make_all
from tests.test_steering import _run

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from comparison.trajectory_jobs import jobs  # noqa: E402

TASKS = ["piqa", "sciq_closed"]


def _save(run_dir, tag, label, tokens, acc, tasks=TASKS):
    name = os.path.basename(run_dir)
    save_result(run_dir, tag, run_name=name, variant="text", conditioning_kind="text", conditioning_meta={},
                tasks_cfg={}, lm_eval_results={"results": {t: {"acc_norm,none": acc} for t in tasks}},
                ckpt_label=label, checkpoint={"path": "x", "step": 1, "tokens": tokens})


def test_run_checkpoint_and_milestones(tmp_path):
    a = _run(tmp_path, "a_medium", [(860, 250), (1720, 500)], last=(1900, 550_000_000))
    assert milestones(a) == ["tok250M", "tok500M"]
    path, info = run_checkpoint(a, "tok500M")
    assert path.endswith("ckpt_step0001720_tok500M.pt") and info["step"] == 1720
    with pytest.raises(ValueError):
        run_checkpoint(a, "tok750M")


def test_jobs_skip_the_shared_checkpoint(tmp_path, monkeypatch):
    b = _run(tmp_path, "b_medium", [(860, 250), (1720, 500)])
    u = _run(tmp_path, "u_medium", [(861, 250), (1721, 500)])
    cfg = {"checkpoint": "tok500M", "runs": {"baseline": b, "url": u}}
    js = jobs(cfg, ["wikipedia", "reddit"])
    assert js == [f"{b} tok250M none", f"{u} tok250M none", f"{u} tok250M text wikipedia",
                  f"{u} tok250M text reddit"]
    assert len(jobs(cfg, ["wikipedia"], include_shared=True)) == 2 + 4


def test_trajectory_series_and_figures(tmp_path):
    b, u = str(tmp_path / "baseline_medium"), str(tmp_path / "url_medium")
    for i, tok in enumerate([250, 500]):
        _save(b, "unconditioned", f"tok{tok}M", tok * 10**6, 0.5 + i / 10)
        _save(u, "unconditioned", f"tok{tok}M", tok * 10**6, 0.52 + i / 10)
        _save(u, "text_wikipedia", f"tok{tok}M", tok * 10**6, 0.55 + i / 10)
    _save(u, "text_reddit", "tok250M", 250 * 10**6, 0.4, tasks=["piqa"])     # older task list: skipped
    cfg = {"model_size": "medium", "runs": {"baseline": b, "url": u}}
    series, skipped = collect(cfg, "/", TASKS)
    assert skipped == 1
    assert [t for t, _ in series[("url", "text_wikipedia")]] == [250 * 10**6, 500 * 10**6]
    assert series[("baseline", "unconditioned")][1][1]["piqa"][0] == pytest.approx(0.6)
    out = tmp_path / "figs"
    make_all(cfg, "/", TASKS, ["wikipedia", "reddit"], str(out))
    for stem in ("trajectory_unconditioned_mean", "trajectory_unconditioned_by_task",
                 "trajectory_conditioned_mean", "trajectory_conditioned_by_task"):
        assert (out / "figures" / f"{stem}.png").exists()
    assert (out / "tables" / "trajectory.csv").exists()
