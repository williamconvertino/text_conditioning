"""One lm-evaluation-harness pass for an already-loaded run under one
ConditioningSpec (src/benchmarks/conditioning.py), shared by
scripts/benchmarks/run_benchmarks.py (one spec per invocation) and the
steering experiments (scripts/steering/, many specs per loaded model), so
every benchmark number in the repo comes from the same adapter and settings.
"""

from lm_eval import simple_evaluate

from src.benchmarks.lm_wrapper import CtrlTokensLM
from src.benchmarks.results import save_result
from src.benchmarks.tasks import task_manager


def benchmark_settings(tasks_cfg, tasks=None, limit=None, num_fewshot=None, batch_size=None):
    """Resolve per-invocation overrides against configs/benchmarks/tasks.yaml."""
    return {
        "tasks": list(tasks or tasks_cfg["tasks"]),
        "num_fewshot": num_fewshot if num_fewshot is not None else tasks_cfg.get("num_fewshot", 0),
        "limit": limit if limit is not None else tasks_cfg.get("limit"),
        "batch_size": batch_size or tasks_cfg.get("batch_size", 16),
    }


def evaluate_spec(model, tokenizer, cfg, spec, settings, device=None):
    """Run lm-eval on `model` conditioned per `spec`. Returns lm-eval's results
    dict (simple_evaluate's return value)."""
    lm = CtrlTokensLM(model, tokenizer, cfg["variant"], spec,
                      block_size=cfg["block_size"], meco_max_tokens=cfg.get("meco_max_tokens", 48),
                      batch_size=settings["batch_size"], device=device)
    seed = cfg.get("seed", 1234)
    return simple_evaluate(model=lm, tasks=settings["tasks"], num_fewshot=settings["num_fewshot"],
                           limit=settings["limit"], log_samples=False, task_manager=task_manager(),
                           random_seed=seed,
                           numpy_random_seed=seed, torch_random_seed=seed)


def evaluate_and_save(model, tokenizer, cfg, spec, settings, run_dir, *, tag=None, ckpt_label=None,
                      checkpoint=None, device=None):
    """evaluate_spec + save_result. Returns (results, results.json path)."""
    results = evaluate_spec(model, tokenizer, cfg, spec, settings, device=device)
    run_name = run_dir.rstrip("/").split("/")[-1]
    path = save_result(run_dir, tag or spec.tag, run_name=run_name, variant=cfg["variant"],
                       conditioning_kind=spec.kind, conditioning_meta=spec.meta,
                       tasks_cfg={k: settings[k] for k in ("tasks", "num_fewshot", "limit")},
                       lm_eval_results=results, ckpt_label=ckpt_label, checkpoint=checkpoint)
    return results, path
