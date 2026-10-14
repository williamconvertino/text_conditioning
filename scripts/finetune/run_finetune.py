#!/usr/bin/env python
"""Fine-tune a pretrained run on a benchmark task and record accuracy against
step (configs/finetune/default.yaml, src/finetune/, README "Fine-Tuning").

  python scripts/finetune/run_finetune.py --model baseline --method standard --task arc_easy
  python scripts/finetune/run_finetune.py --model meco --method conditioned --task piqa
  python scripts/finetune/run_finetune.py --model url --method peft --task all     # joint over every task

  # another scale: its comparison config names its runs
  python scripts/finetune/run_finetune.py --comparison-config configs/comparison_1.5b.yaml \\
      --model url --method peft --task boolq

  # full fine-tuning at 8B: shard across 4 GPUs
  torchrun --standalone --nproc_per_node=4 scripts/finetune/run_finetune.py \\
      --comparison-config configs/comparison_8b.yaml --model meco --method standard --task piqa --set fsdp=true

--model is a key of the comparison config (baseline | meco | url) or a run
directory. (Not --run: under torchrun that abbreviates torchrun's own
--run-path.) The model starts from --checkpoint: "shared" (default) is the
furthest checkpoint every compared run has, i.e. the final one once all have
finished (src/comparison/checkpoints.py); a file name inside the run dir
(e.g. ckpt_last.pt) picks one directly.

--method is a key of the config's `methods` (standard | conditioned | peft |
...). --set key=value overrides any config entry (value parsed as YAML;
dotted keys reach into nested entries, e.g. --set condition.url=https://arxiv.org/).

--seed (or --set seed=K) changes the training-data order only: the splits
and the eval-curve subset are fixed by splits.seed, so seeds are directly
comparable and the report averages over them.

Output: outputs/experiments/<size>/finetune/<run dir name>/<ckpt label>/<method>/<task>/seed<K>/
  metrics.jsonl  train loss, and select / eval-curve accuracy per eval step
  result.json    step-0, final and best-on-select accuracy on the full eval
                 split, the curve, and the conditioning used
  prefix.pt      per-doc eval correctness and the learned vectors (peft)
Figures and tables: scripts/finetune/make_finetune_report.py.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import yaml

from src.comparison.checkpoints import checkpoint_for, shared_checkpoint
from src.config import load_config, save_config
from src.data.tokenizer_utils import load_tokenizer
from src.finetune.data import TaskData
from src.finetune.loop import Dist, run_finetune
from src.finetune.model import build_finetune_model, load_base_model
from src.steering.task_data import build_splits, load_tasks
from src.training.checkpointing import save_eval_checkpoint
from src.utils import paths, sharding
from src.utils.distributed import ddp_cleanup, ddp_setup


def apply_overrides(cfg, pairs):
    for pair in pairs:
        key, _, value = pair.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def resolve_run(run, comparison_config, checkpoint):
    """(absolute run dir, checkpoint file relative to it, checkpoint label)."""
    cmp_cfg = load_config(comparison_config)
    run_dir = cmp_cfg["runs"].get(run, run)
    if checkpoint == "shared":
        pin = shared_checkpoint(cmp_cfg, ROOT)
        ckpt, _ = checkpoint_for(pin, run_dir)
        label = pin["label"]
    else:
        ckpt, label = checkpoint, os.path.splitext(os.path.basename(checkpoint))[0]
    return os.path.join(ROOT, run_dir), ckpt, label


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", dest="run", required=True,
                   help="Comparison-config key (baseline | meco | url) or a run dir.")
    p.add_argument("--method", required=True, help="A key of the config's `methods`.")
    p.add_argument("--task", required=True, help="An lm-eval task from the suite, or 'all'.")
    p.add_argument("--config", default=os.path.join(ROOT, "configs/finetune/default.yaml"))
    p.add_argument("--comparison-config", default=os.path.join(ROOT, "configs/comparison.yaml"))
    p.add_argument("--tasks-config", default=os.path.join(ROOT, "configs/benchmarks/tasks.yaml"))
    p.add_argument("--checkpoint", default="shared")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a config entry.")
    p.add_argument("--seed", type=int, default=None, help="Training seed (default: the config's).")
    p.add_argument("--out-root", default=None,
                   help="Default: outputs/experiments/<run's model size>/finetune (src/utils/paths.py).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny splits/steps (results under '<label>-smoke').")
    args = p.parse_args()

    rank, world, local_rank, device, is_ddp = ddp_setup()
    log = print if rank == 0 else (lambda *a, **k: None)

    cfg = apply_overrides(load_config(args.config), args.set)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.method not in cfg["methods"]:
        raise SystemExit(f"unknown --method {args.method!r}; known: {list(cfg['methods'])}")
    method = cfg["methods"][args.method]
    mode = method["mode"]
    if args.smoke:
        cfg["splits"].update(n_fit=64, n_select=32)
        cfg.update(steps=4, eval_every=2, curve_eval_docs=32, log_every=1, warmup_steps=1)
    fsdp = bool(cfg.get("fsdp")) and is_ddp
    torch.manual_seed(cfg["seed"])

    task_names = cfg.get("tasks") or load_config(args.tasks_config)["tasks"]
    task_names = task_names if args.task == "all" else [args.task]
    cond = cfg["condition"]
    condition_url = cond.get("per_task", {}).get(args.task, cond["url"])

    run_dir, ckpt, label = resolve_run(args.run, args.comparison_config, args.checkpoint)
    run_cfg = load_config(os.path.join(run_dir, "resolved_config.yaml"))
    label += "-smoke" if args.smoke else ""
    tag = args.method + ("" if cfg["objective"] == "choice_ce" else f"_{cfg['objective']}")
    out_root = args.out_root or paths.experiments_dir(paths.model_size(run_dir), "finetune")
    out_dir = os.path.join(out_root, os.path.basename(os.path.normpath(run_dir)), label, tag, args.task,
                           f"seed{cfg['seed']}")
    log(f"[finetune] {run_dir} ({ckpt}) -> {out_dir}")

    tokenizer = load_tokenizer(run_cfg["tokenizer"])
    base = load_base_model(run_dir, ckpt, tokenizer, run_cfg)
    ft, info = build_finetune_model(base, run_cfg, tokenizer, mode, condition_url=condition_url,
                                    peft_init=method.get("init", "condition"),
                                    n_vectors=method.get("n_vectors", 1),
                                    train_encoder=method.get("train_encoder", False))
    info.update(run_dir=run_dir, checkpoint=ckpt, ckpt_label=label, method=args.method, method_cfg=method)
    log(f"[finetune] {mode}: {info['n_trainable'] / 1e6:.3f}M trainable params  "
        + "  ".join(f"{k}={info[k]!r}" for k in ("condition_url", "prefix", "encoder_input", "peft_init")
                    if info.get(k) is not None))

    if fsdp:
        wrapped = ft = sharding.shard_model(ft, device, world)
    else:
        ft = ft.to(device)
        wrapped = ft
        if is_ddp:
            kw = {"device_ids": [local_rank]} if device.type == "cuda" else {}
            wrapped = torch.nn.parallel.DistributedDataParallel(ft, **kw)
    D = Dist(rank, world, device, is_ddp, fsdp)

    tasks = load_tasks(task_names)
    sp = cfg["splits"]
    data = []
    for t in task_names:
        splits = build_splits(t, tasks[t], sp["n_fit"], sp["n_select"], sp["seed"], sp.get("holdout_frac", 0.3),
                              n_eval=64 if args.smoke else None)
        log(f"{t}: metric {splits.metric}  " + "  ".join(f"{k}={v}" for k, v in splits.info.items()))
        # the curve subset follows the split seed, so every training seed scores the same docs
        data.append(TaskData(splits, tokenizer, run_cfg["block_size"], ft.prefix_ids, cfg["curve_eval_docs"],
                             sp["seed"]))

    result = run_finetune(ft, wrapped, data, mode, cfg, D, out_dir, tokenizer.pad_token_id, info, log=log)
    t = result["tasks"]
    log(f"[finetune] best step {result['best_step']}: eval "
        + "  ".join(f"{k} {result['base'][k]['eval']:.4f} -> {result['best'][k]['eval']:.4f}" for k in t))

    if cfg.get("save_model"):
        # the kept (best-on-select) iterate, as an ordinary run checkpoint
        state = sharding.full_model_state(ft) if fsdp else ft.state_dict()
        if rank == 0:
            state = {k[len("base."):]: v for k, v in state.items() if k.startswith("base.")}
            save_eval_checkpoint(os.path.join(out_dir, "model.pt"), state, result["best_step"], None, None,
                                 {**run_cfg, "finetune": info})
            save_config(run_cfg, os.path.join(out_dir, "resolved_config.yaml"))
    ddp_cleanup(is_ddp)


if __name__ == "__main__":
    main()
