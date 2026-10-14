#!/bin/bash
#SBATCH --job-name=tc_steer_transfer
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-1

# Cross-task transfer of the task-optimized conditioning found by
# slurm/steering/task_vectors.slurm (array task 0: URL-encoded vectors,
# 1: MeCo's best domains). Run after every task_vectors job has finished.
# Run with: sbatch slurm/steering/task_transfer.slurm
# Another scale / only the URL run:
#   sbatch --array=0 --export=ALL,SCALE=3b,RUNS=url slurm/steering/task_transfer.slurm

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
read -r -a RUNS <<< "${RUNS:-url meco}"
python scripts/steering/run_task_transfer.py --comparison-config "$COMPARISON" --run "${RUNS[$SLURM_ARRAY_TASK_ID]}"
