#!/bin/bash
#SBATCH --job-name=tc_steer_task_vectors
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-29

# Steering experiment 2: task-optimized conditioning
# (configs/steering/task_vectors.yaml). One array task per (run, task) for
# the three compared runs x (9 benchmark tasks + "all", the joint vector).
# Needs outputs/experiments/<size>/url_analysis/<ckpt label>/library.pt (domain candidates and
# the simplex library). Then: slurm/steering/task_transfer.slurm.
# Run with: sbatch slurm/steering/task_vectors.slurm
# Another scale (configs/comparison_<SCALE>.yaml), only some runs (array size
# = runs x 9):
#   sbatch --array=0-9 --export=ALL,SCALE=3b,RUNS=url slurm/steering/task_vectors.slurm

set -euo pipefail

cd /work/wac20/ctrl_tokens/text_conditioned
mkdir -p logs

CONDA_ENV=ctrl_tokens
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export HF_HOME=/work/wac20/.cache/huggingface
export HF_TOKEN=${HF_TOKEN:?"set HF_TOKEN in .env (needs accepted Llama-3.2-3B license)"}

COMPARISON=configs/comparison.yaml
[[ -n "${SCALE:-}" ]] && COMPARISON=configs/comparison_${SCALE}.yaml
read -r -a RUNS <<< "${RUNS:-baseline meco url}"
TASKS=(hellaswag piqa arc_easy arc_challenge winogrande boolq openbookqa sciq_closed mmlu_cont all)
RUN=${RUNS[$((SLURM_ARRAY_TASK_ID / ${#TASKS[@]}))]}
TASK=${TASKS[$((SLURM_ARRAY_TASK_ID % ${#TASKS[@]}))]}

python scripts/steering/run_task_vectors.py --comparison-config "$COMPARISON" --run "$RUN" --task "$TASK"
