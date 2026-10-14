#!/usr/bin/env python
"""Cross-task transfer of task-optimized conditioning: evaluate the vector
(or, for MeCo, the domain) each per-task run of scripts/steering/run_task_vectors.py
found for task A on every task B's eval split. A strong diagonal with a weak
off-diagonal means the vectors are task-specific; uniform gains mean they
found a generally better "benchmark-like" conditioning.

  python scripts/steering/run_task_transfer.py --run url
  python scripts/steering/run_task_transfer.py --run meco --methods domain_search

Output: outputs/experiments/<size>/steering/<ckpt label>/task_vectors/<run>/transfer.json
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.benchmarks.conditioning import resolve_meco
from src.config import load_config
from src.steering.scoring import make_lm, score_docs
from src.steering.task_data import build_splits, load_tasks
from src.steering.vectors import load_compared_run, text_spec
from src.utils import paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--methods", nargs="+", default=None,
                   help="Default: opt_free opt_simplex domain_search (url); domain_search (meco).")
    p.add_argument("--config", default=os.path.join(ROOT, "configs/steering/task_vectors.yaml"))
    p.add_argument("--comparison-config", default=os.path.join(ROOT, "configs/comparison.yaml"))
    p.add_argument("--tasks-config", default=os.path.join(ROOT, "configs/benchmarks/tasks.yaml"))
    p.add_argument("--checkpoint", default="shared")
    p.add_argument("--smoke", action="store_true", help="Read/write run_task_vectors.py --smoke results.")
    args = p.parse_args()

    cfg = load_config(args.config)
    R = load_compared_run(ROOT, args.comparison_config, args.run, args.checkpoint)
    variant = R["cfg"]["variant"]
    methods = args.methods or (["opt_free", "opt_simplex", "domain_search"] if variant == "text" else ["domain_search"])
    task_names = cfg.get("tasks") or load_config(args.tasks_config)["tasks"]
    label = R["ckpt_label"] + ("-smoke" if args.smoke else "")
    base = paths.experiments_dir(R["size"], "steering", label, "task_vectors", R["run_key"])

    sources = {}                      # (method, source task) -> spec
    for t in task_names + ["all"]:
        jpath, ppath = os.path.join(base, f"{t}.json"), os.path.join(base, f"{t}.pt")
        if not os.path.exists(jpath):
            print(f"missing {jpath}; skipping source task {t}", flush=True)
            continue
        res = json.load(open(jpath))
        side = torch.load(ppath, weights_only=False)
        for m in methods:
            if m not in res["methods"]:
                continue
            if variant == "meco":
                sources[(m, t)] = resolve_meco(domain=res["methods"][m]["chosen"])
            else:
                sources[(m, t)] = text_spec(f"{m}_{t}", side["vectors"][m], {"method": m, "fit_task": t})

    sp = cfg["splits"]
    tasks = load_tasks(task_names)
    if args.smoke:
        sp = {**sp, "n_fit": 64, "n_select": 48}
    splits = {t: build_splits(t, tasks[t], sp["n_fit"], sp["n_select"], sp["seed"], sp.get("holdout_frac", 0.3),
                              n_eval=100 if args.smoke else None) for t in task_names}
    out = {"run": R["run_key"], "ckpt_label": R["ckpt_label"], "tasks": task_names, "methods": methods,
           "acc": {}, "stderr": {}}
    for (m, src), spec in sources.items():
        lm = make_lm(R["model"], R["tokenizer"], R["cfg"], spec, cfg["eval_batch_size"])
        for t, s in splits.items():
            r = score_docs(lm, s, s.eval)
            out["acc"].setdefault(m, {}).setdefault(src, {})[t] = r["acc"]
            out["stderr"].setdefault(m, {}).setdefault(src, {})[t] = r["stderr"]
        print(f"  {m:14s} fit on {src:15s} -> mean eval {sum(out['acc'][m][src].values()) / len(splits):.4f}",
              flush=True)
        with open(os.path.join(base, "transfer.json"), "w") as f:
            json.dump(out, f, indent=1)
    print(f"wrote {os.path.join(base, 'transfer.json')}")


if __name__ == "__main__":
    main()
