#!/usr/bin/env python
"""Evaluate a trained run.

Quick mode (--mode quick): validation loss/ppl under the run's natural
conditioning, same computation as training-time validation -- a fast sanity
check that doesn't touch the encoder-heavy modes.

Final mode (--mode final, the default): the full README "Final Conditioning
Evaluation" -- none / self / same_domain / domain_mean (/ cross_domain) -- run
once on the final checkpoint and saved to <run-dir>/final_conditioning.json.
Each mode is reported at two scopes (src/eval/conditioning.py): "overall" (the
fixed, domain-independent validation sample) and "top_domains" (a fixed set of
domains with enough validation documents to support same-domain/domain-mean
conditioning at all -- see that module's docstring), both as a combined
aggregate and a per-domain breakdown.

Given several runs' final_conditioning.json (via --table), also builds the
comparison table (README "Final Evaluation Table") as csv/md/tex, plus a
per-domain breakdown CSV.

Usage (run dirs are named <experiment>_<model_size>, e.g. outputs/baseline_medium
-- see scripts/train.py):
  python scripts/evaluate.py --run-dir outputs/baseline_medium
  python scripts/evaluate.py --run-dir outputs/meco_medium
  python scripts/evaluate.py --run-dir outputs/text_conditioned_medium
  python scripts/evaluate.py --table outputs/baseline_medium outputs/meco_medium \\
      outputs/text_conditioned_medium --table-out outputs/tables
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_config
from src.data.collator import load_val_batches
from src.data.shards import split_shards
from src.eval.conditioning import evaluate_final
from src.eval.perplexity import evaluate_loss
from src.training.checkpointing import load_run
from src.analysis.tables import write_all_tables


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", help="Single run to evaluate.")
    p.add_argument("--checkpoint", default="ckpt_best.pt")
    p.add_argument("--mode", choices=["quick", "final"], default="final")
    p.add_argument("--eval-config", default=os.path.join(ROOT, "configs/eval/conditioning.yaml"))
    p.add_argument("--table", nargs="+", default=None, help="Run dirs to build the comparison table from.")
    p.add_argument("--table-out", default=None)
    return p.parse_args()


def evaluate_one(run_dir, args):
    model, tokenizer, cfg = load_run(run_dir, args.checkpoint)
    _, val_stems = split_shards(cfg["tok_dir"], cfg["n_val_shards"])

    if args.mode == "quick":
        val_batches = load_val_batches(cfg["tok_dir"], val_stems, tokenizer, cfg["variant"],
                                       cfg["block_size"], cfg["meco_max_tokens"],
                                       cfg.get("encoder_max_tokens", 512), cfg["val_docs"],
                                       cfg["val_batch_size"])
        loss, ppl = evaluate_loss(model, val_batches)
        print(f"{run_dir}: loss={loss:.4f} ppl={ppl:.2f}")
        return {"none": (loss, ppl)}

    eval_cfg = dict(load_config(args.eval_config))
    eval_cfg.setdefault("block_size", cfg["block_size"])
    eval_cfg.setdefault("meco_max_tokens", cfg["meco_max_tokens"])
    eval_cfg.setdefault("encoder_max_tokens", cfg.get("encoder_max_tokens", 512))
    results = evaluate_final(model, tokenizer, cfg["tok_dir"], val_stems, eval_cfg)

    out_path = os.path.join(run_dir, "final_conditioning.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_path}")
    _print_results(results)
    return results


def _print_mode_table(modes_dict, indent="  "):
    for mode, val in modes_dict.items():
        print(f"{indent}{mode:12s} {'--' if val is None else f'loss={val[0]:.4f} ppl={val[1]:.2f}'}")


def _print_results(results):
    print("overall (fixed validation sample):")
    _print_mode_table(results["overall"])

    td = results.get("top_domains")
    if td is None:
        print("top_domains: no domain had enough validation documents to support "
             "same-domain/domain-mean conditioning (see configs/eval/conditioning.yaml)")
        return

    print(f"top_domains ({td['n_domains']} domains, aggregate):")
    _print_mode_table(td["aggregate"])
    print(f"  see final_conditioning.json's top_domains.per_domain for the "
         f"per-domain breakdown across {td['n_domains']} domains")


def main():
    args = parse_args()
    if args.run_dir:
        evaluate_one(args.run_dir, args)

    if args.table:
        table_out = args.table_out or os.path.join(os.path.dirname(args.table[0].rstrip("/")), "tables")
        try:
            _, domain_rows = write_all_tables(args.table, table_out)
        except FileNotFoundError as e:
            raise SystemExit(str(e))
        print(f"wrote table to {table_out}")
        if domain_rows:
            print(f"wrote per-domain breakdown to {os.path.join(table_out, 'domain_breakdown.csv')}")


if __name__ == "__main__":
    main()
