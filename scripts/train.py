#!/usr/bin/env python
"""Train one experiment. The experiment config determines the variant (baseline
/ meco / text-conditioned) -- there is no separate training script per variant.

Single-GPU:  python scripts/train.py --experiment configs/experiment/baseline.yaml --out-dir outputs
Multi-GPU:   torchrun --standalone --nproc_per_node=4 scripts/train.py \\
                 --experiment configs/experiment/text_conditioned.yaml --out-dir outputs

Resume is automatic: rerun the same command and it picks up ckpt_last.pt.

Standing, never-overwritten eval snapshots are written to
outputs/<run>/checkpoints/ every ckpt_save_interval_tokens tokens (configs/
train/default.yaml, default 250M) independent of ckpt_last.pt/ckpt_best.pt --
point scripts/evaluate.py's --checkpoint at any of them to evaluate mid-training
progress without running the full schedule.

Conditioning-schedule sweeps (e.g. trying several constant p_condition values,
see src/training/schedules.py) apply to both variant: text and variant: meco
runs -- pass --p-condition (or --conditioning-schedule for non-constant
schedules) instead of hand-editing/copying the experiment yaml. The override
is folded into the run dir name so each value in the sweep gets its own run
dir, e.g.:
  python scripts/train.py --experiment configs/experiment/text_conditioned.yaml \\
      --p-condition 0.7
  -> outputs/text_conditioned_p0.7_medium
  python scripts/train.py --experiment configs/experiment/meco.yaml --p-condition 0.7
  -> outputs/meco_p0.7_medium
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_config
from src.training.schedules import schedule_tag
from src.training.trainer import run


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, help="Path to configs/experiment/<name>.yaml")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--max-steps", type=int, default=None, help="Override the config's max_steps.")
    p.add_argument("--p-condition", type=float, default=None,
                   help="Override conditioning_schedule with a constant schedule at this "
                        "p_condition value (only meaningful for variant: text or meco).")
    p.add_argument("--conditioning-schedule", type=str, default=None,
                   help="Override conditioning_schedule with this JSON dict, e.g. "
                        "'{\"kind\": \"linear_cooldown\", \"p_start\": 1.0, \"p_end\": 0.0, "
                        "\"cooldown_start_frac\": 0.7, \"cooldown_end_frac\": 1.0}'. Takes "
                        "precedence over --p-condition.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.experiment)
    cfg.setdefault("name", os.path.splitext(os.path.basename(args.experiment))[0])
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.no_resume:
        cfg["no_resume"] = True

    schedule_override = None
    if args.conditioning_schedule is not None:
        schedule_override = json.loads(args.conditioning_schedule)
    elif args.p_condition is not None:
        schedule_override = {"kind": "constant", "p_condition": args.p_condition}
    if schedule_override is not None:
        cfg["conditioning_schedule"] = schedule_override

    # Suffix the run dir with the backbone's model_size (configs/model/*.yaml)
    # so that pointing an experiment's `include:` at a different-sized model
    # (e.g. llama_medium.yaml -> llama3_3b.yaml) starts a fresh run dir instead
    # of resuming/overwriting a smaller model's checkpoints and logs. Likewise,
    # a --p-condition/--conditioning-schedule override gets its own tag so a
    # sweep run doesn't resume/overwrite the yaml's own default-schedule run.
    default_run_name = cfg["name"]
    if schedule_override is not None:
        default_run_name += f"_{schedule_tag(schedule_override)}"
    default_run_name += f"_{cfg.get('model_size', 'default')}"
    run_dir = os.path.join(args.out_dir, args.run_name or default_run_name)
    run(cfg, run_dir)


if __name__ == "__main__":
    main()
