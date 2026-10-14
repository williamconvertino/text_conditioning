#!/bin/bash
#SBATCH --job-name=tc_steer_mixtures
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-5

# Steering experiment 1: lm-eval on combinations of URL vectors
# (configs/steering/mixtures.yaml), split over NUM_SHARDS array tasks (keep in
# sync with --array). Needs outputs/experiments/<size>/url_analysis/<ckpt label>/library.pt
# (slurm/url_analysis/encoder_analysis.slurm) for the strength sweep's
# reference. Finished vectors are skipped, so resubmitting resumes.
# Run with: sbatch slurm/steering/mixtures.slurm
# Smoke:    sbatch --export=ALL,LIMIT=100 slurm/steering/mixtures.slurm
# Another scale: sbatch --export=ALL,SCALE=3b slurm/steering/mixtures.slurm

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
NUM_SHARDS=6
ARGS=(--comparison-config "$COMPARISON" --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$NUM_SHARDS")
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "${LIMIT}")
python scripts/steering/run_mixtures.py "${ARGS[@]}"
