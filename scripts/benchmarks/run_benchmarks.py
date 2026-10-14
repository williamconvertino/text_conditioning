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

URL-encoded text runs (text_conditioning_source: url) encode URL metadata
instead of documents, so they can be steered with any string in vector space:
  --mode text   --source <name>        E(the source's meco_domain)
  --mode text   --domain <string>      E(any ad hoc domain string)
  --mode text   --url <url>            E(that URL's domain / full URL); need not
                                        be a training document

Results are written to outputs/experiments/<size>/benchmarks/<run_name>/<tag>/results.json
(src/benchmarks/results.py) so every (run, conditioning) combination lives in
its own predictable place -- see scripts/benchmarks/plot_benchmarks.py to
compare them, or `python scripts/benchmarks/plot_benchmarks.py list` to see
what's already been run.

Examples:
  python scripts/benchmarks/run_benchmarks.py --run-dir outputs/checkpoints/baseline_medium --mode none

  python scripts/benchmarks/run_benchmarks.py --run-dir outputs/checkpoints/meco_medium \\
      --mode meco --source wikipedia

  python scripts/benchmarks/run_benchmarks.py --run-dir outputs/checkpoints/text_conditioned_medium \\
      --mode text --source wikipedia

  python scripts/benchmarks/run_benchmarks.py --run-dir outputs/checkpoints/text_conditioned_medium \\
      --mode text --url https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)

  # fast smoke test across every default task
  python scripts/benchmarks/run_benchmarks.py --run-dir outputs/checkpoints/baseline_medium --mode none --limit 50

  # at the furthest checkpoint shared by the compared runs (configs/comparison.yaml,
  # scripts/comparison/show_checkpoint.py); results go to
  # outputs/experiments/<size>/benchmarks/<run>/<ckpt label>/<tag>/results.json
  python scripts/benchmarks/run_benchmarks.py --run-dir outputs/checkpoints/meco_p0.9_medium \\
      --checkpoint shared --mode meco --source wikipedia
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.comparison.checkpoints import checkpoint_for, is_pin, run_checkpoint, shared_checkpoint
from src.config import load_config
from src.data.shards import split_shards
from src.training.checkpointing import load_run
from src.benchmarks.conditioning import resolve_meco, resolve_none, resolve_text, resolve_url_text
from src.data.url_conditioning import build_url_source
from src.benchmarks.results import best_metric, load_result, result_path
from src.benchmarks.runner import benchmark_settings, evaluate_and_save


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="ckpt_best.pt",
                   help="Checkpoint file in --run-dir, or 'shared' for the furthest checkpoint shared by "
                        "--comparison-config's runs (results then go under its label, e.g. step20000), or "
                        "a tok<N>M / step<N> milestone of this run alone (results go under that label).")
    p.add_argument("--comparison-config", default=os.path.join(ROOT, "configs/comparison.yaml"))
    p.add_argument("--mode", choices=["none", "meco", "text"], required=True)
    p.add_argument("--source", default=None,
                   help="Named source from --sources-config (e.g. wikipedia, reddit).")
    p.add_argument("--domain", default=None,
                   help="Ad hoc domain/URL string: --mode meco, or --mode text for a URL-encoded run.")
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
    p.add_argument("--skip-existing", action="store_true",
                   help="Do nothing if this (run, checkpoint, tag) already has results for every task.")
    return p.parse_args()


def main():
    args = parse_args()
    tasks_cfg = load_config(args.tasks_config)
    sources_cfg = load_config(args.sources_config) if args.mode in ("meco", "text") else None

    checkpoint, ckpt_label, ckpt_info = args.checkpoint, None, None
    if args.checkpoint == "shared":
        pin = shared_checkpoint(load_config(args.comparison_config), ROOT)
        checkpoint, ckpt_info = checkpoint_for(pin, args.run_dir)
        ckpt_label = pin["label"]
        print(f"shared checkpoint {ckpt_label}: {checkpoint} (step {ckpt_info['step']})", flush=True)
    elif is_pin(args.checkpoint):
        checkpoint, ckpt_info = run_checkpoint(args.run_dir, args.checkpoint)
        ckpt_label = args.checkpoint
        print(f"checkpoint {ckpt_label}: {checkpoint} (step {ckpt_info['step']})", flush=True)

    model, tokenizer, cfg = load_run(args.run_dir, checkpoint)
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
        url_source = build_url_source(cfg, tokenizer)
        if url_source is not None:
            spec = resolve_url_text(model.conditioner, tokenizer, url_source, device,
                                    source=args.source, domain=args.domain, url=args.url,
                                    sources_cfg=sources_cfg)
        elif args.domain is not None:
            raise SystemExit("--mode text --domain needs a URL-encoded run (text_conditioning_source: url); "
                             "document-encoded runs take --source or --url")
        else:
            train_stems, _ = split_shards(cfg["tok_dir"], cfg["n_val_shards"])
            spec = resolve_text(
                model.conditioner, tokenizer, cfg["tok_dir"], train_stems,
                cfg.get("encoder_max_tokens", 512), device,
                source=args.source, url=args.url, sources_cfg=sources_cfg,
                n_docs=args.n_docs_per_source or sources_cfg.get("n_docs_per_source", 20),
                seed=args.seed if args.seed is not None else sources_cfg.get("seed", 1234))

    tag = args.tag or spec.tag
    settings = benchmark_settings(tasks_cfg, tasks=args.tasks, limit=args.limit,
                                  num_fewshot=args.num_fewshot, batch_size=args.batch_size)

    run_name = os.path.basename(args.run_dir.rstrip("/"))
    existing = result_path(args.run_dir, tag, ckpt_label)
    if args.skip_existing and os.path.exists(existing) and \
            set(settings["tasks"]) <= set(load_result(existing)["results"]) and settings["limit"] is None:
        print(f"[{run_name}] {existing} already has every task; skipping")
        return
    print(f"[{run_name}] mode={args.mode} tag={tag} tasks={settings['tasks']} "
         f"num_fewshot={settings['num_fewshot']} limit={settings['limit']}", flush=True)
    if spec.meta:
        print(f"  conditioning: {spec.meta}", flush=True)

    results, out_path = evaluate_and_save(model, tokenizer, cfg, spec, settings, args.run_dir, tag=tag,
                                          ckpt_label=ckpt_label, checkpoint=ckpt_info, device=device)
    print(f"wrote {out_path}")
    for task, task_results in results["results"].items():
        metric, value = best_metric(task_results)
        print(f"  {task:20s} {metric or '--':16s} {'--' if value is None else f'{value:.4f}'}")


if __name__ == "__main__":
    main()
