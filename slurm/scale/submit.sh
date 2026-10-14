#!/bin/bash
# Pretrain one run at a Llama 3 scale on one 8-GPU node (README "Scaling Up"):
#
#   bash slurm/scale/submit.sh 3b text_conditioned_url_full 60      # up to 60h
#   bash slurm/scale/submit.sh 3b meco 60 0.9                       # MeCo at p_condition 0.9
#   DEPENDENCY=afterany:123 bash slurm/scale/submit.sh 1.5b baseline 72
#   DATA=fineweb_edu bash slurm/scale/submit.sh 1.5b text_conditioned_url_full_strong 62   # FineWeb-Edu
#
# Runs are trained one at a time, each for as long as the budget allows, and
# compared at matching token milestones (the eval snapshots every
# ckpt_save_interval_tokens; src/comparison/checkpoints.py picks the largest
# one all compared runs share). The job stops itself cleanly shortly before
# its time limit, and resubmitting the same command continues the run.
# Afterwards: slurm/scale/eval.slurm, slurm/scale/bench.slurm and
# slurm/finetune/submit_all.sh <scale> evaluate the compared runs.
set -euo pipefail
cd /work/wac20/ctrl_tokens/text_conditioned

SCALE=${1:?"usage: bash slurm/scale/submit.sh <1.5b|3b|8b> <experiment> <hours> [p_condition]"}
EXPERIMENT=${2:?"experiment name, e.g. text_conditioned_url_full"}
HOURS=${3:?"time limit in hours"}
P=${4:-}
DEP=()
[[ -n "${DEPENDENCY:-}" ]] && DEP=(--dependency="$DEPENDENCY")
DATA=${DATA:-}
sbatch --parsable "${DEP[@]}" --time="${HOURS}:00:00" --job-name="tc_train_${EXPERIMENT}${DATA:+_$DATA}_${SCALE}" \
  --export=ALL,SCALE="$SCALE",EXPERIMENT="$EXPERIMENT"${P:+,P_CONDITION=$P}${DATA:+,DATA=$DATA} slurm/scale/train.slurm
