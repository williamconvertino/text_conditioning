#!/usr/bin/env python
"""Run the lm-evaluation-harness benchmark suite (configs/benchmarks/tasks.yaml)
against a trained run, under one conditioning setting.

This is a separate script/module tree from scripts/evaluate.py and src/eval/
(perplexity-based evaluation) on purpose: benchmarking is downstream-task
accuracy via lm-eval-harness, an unrelated evaluation axis from the
train-time/perplexity metrics (README "Evaluation": "Downstream LM-Eval tasks
can be added once the basic conditioning effect has been established").

Conditioning (src/benchmarks/conditioning.py):
  --mode none                          any variant's own unconditioned mode
  --mode meco   --source <name>        MeCo domain prefix, from configs/benchmarks/sources.yaml
  --mode meco   --domain <string>      MeCo domain prefix, ad hoc
  --mode text   --source <name>        text-conditioning vector, averaged over
                                        n_docs_per_source matching training docs
  --mode text   --url <url>            text-conditioning vector, from one
                                        specific training document

Results are written to outputs/<run_name>/benchmarks/<tag>/results.json
(src/benchmarks/results.py) so every (run, conditioning) combination lives in
its own predictable place -- see scripts/benchmarks/plot_benchmarks.py to
compare them, or `python scripts/benchmarks/plot_benchmarks.py list` to see
what's already been run.

Examples:
  python scripts/benchmarks/run_benchmarks.py --run-dir outputs/baseline_medium --mode none

  python scripts/benchmarks/run_benchmarks.py --run-dir outputs/meco_medium \\
      --mode meco --source wikipedia

  python scripts/benchmarks/run_benchmarks.py --run-dir outputs/text_conditioned_medium \\
      --mode text --source wikipedia

  python scripts/benchmarks/run_benchmarks.py --run-dir outputs/text_conditioned_medium \\
      --mode text --url https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)

  # fast smoke test across every default task
  python scripts/benchmarks/run_benchmarks.py --run-dir outputs/baseline_medium --mode none --limit 50
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from lm_eval import simple_evaluate

from src.config import load_config
from src.data.shards import split_shards
from src.training.checkpointing import load_run
from src.benchmarks.conditioning import resolve_meco, resolve_none, resolve_text
from src.benchmarks.lm_wrapper import CtrlTokensLM
from src.benchmarks.results import best_metric, save_result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="ckpt_best.pt")
    p.add_argument("--mode", choices=["none", "meco", "text"], required=True)
    p.add_argument("--source", default=None,
                   help="Named source from --sources-config (e.g. wikipedia, reddit).")
    p.add_argument("--domain", default=None,
                   help="Ad hoc MeCo domain/URL string (--mode meco only).")
    p.add_argument("--url", default=None,
                   help="Ad hoc single training-document URL to condition on (--mode text only).")
    p.add_argument("--tag", default=None, help="Override the auto-generated result tag.")
    p.add_argument("--tasks-config", default=os.path.join(ROOT, "configs/benchmarks/tasks.yaml"))
    p.add_argument("--sources-config", default=os.path.join(ROOT, "configs/benchmarks/sources.yaml"))
    p.add_argument("--tasks", nargs="+", default=None, help="Override tasks.yaml's task list.")
    p.add_argument("--limit", type=float, default=None,
                   help="Docs per task (see tasks.yaml); overrides tasks.yaml.")
    p.add_argument("--num-fewshot", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-docs-per-source", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    tasks_cfg = load_config(args.tasks_config)
    sources_cfg = load_config(args.sources_config) if args.mode in ("meco", "text") else None

    model, tokenizer, cfg = load_run(args.run_dir, args.checkpoint)
    device = next(model.parameters()).device
    variant = cfg["variant"]

    if args.mode == "none":
        spec = resolve_none()
    elif args.mode == "meco":
        if variant != "meco":
            raise SystemExit(f"--mode meco requires a MeCo-trained run (this run's variant={variant!r})")
        spec = resolve_meco(source=args.source, domain=args.domain, sources_cfg=sources_cfg)
    else:
        if variant != "text":
            raise SystemExit(f"--mode text requires a text-conditioned run (this run's variant={variant!r})")
        train_stems, _ = split_shards(cfg["tok_dir"], cfg["n_val_shards"])
        spec = resolve_text(
            model.conditioner, tokenizer, cfg["tok_dir"], train_stems,
            cfg.get("encoder_max_tokens", 512), device,
            source=args.source, url=args.url, sources_cfg=sources_cfg,
            n_docs=args.n_docs_per_source or sources_cfg.get("n_docs_per_source", 20),
            seed=args.seed if args.seed is not None else sources_cfg.get("seed", 1234))

    tag = args.tag or spec.tag

    lm = CtrlTokensLM(
        model, tokenizer, variant, spec,
        block_size=cfg["block_size"], meco_max_tokens=cfg.get("meco_max_tokens", 48),
        batch_size=args.batch_size or tasks_cfg.get("batch_size", 16), device=device)

    tasks = args.tasks or tasks_cfg["tasks"]
    num_fewshot = args.num_fewshot if args.num_fewshot is not None else tasks_cfg.get("num_fewshot", 0)
    limit = args.limit if args.limit is not None else tasks_cfg.get("limit")
    seed = cfg.get("seed", 1234)

    run_name = os.path.basename(args.run_dir.rstrip("/"))
    print(f"[{run_name}] mode={args.mode} tag={tag} tasks={tasks} "
         f"num_fewshot={num_fewshot} limit={limit}", flush=True)
    if spec.meta:
        print(f"  conditioning: {spec.meta}", flush=True)

    results = simple_evaluate(model=lm, tasks=tasks, num_fewshot=num_fewshot, limit=limit,
                              log_samples=False, random_seed=seed, numpy_random_seed=seed,
                              torch_random_seed=seed)

    out_path = save_result(args.run_dir, tag, run_name=run_name, variant=variant,
                           conditioning_kind=spec.kind, conditioning_meta=spec.meta,
                           tasks_cfg={"tasks": tasks, "num_fewshot": num_fewshot, "limit": limit},
                           lm_eval_results=results)
    print(f"wrote {out_path}")
    for task, task_results in results["results"].items():
        metric, value = best_metric(task_results)
        print(f"  {task:20s} {metric or '--':16s} {'--' if value is None else f'{value:.4f}'}")


if __name__ == "__main__":
    main()
