#!/bin/bash
#SBATCH --job-name=tc_train_text_conditioned_url_full
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

# Trains the "text_conditioned_url_full" experiment (configs/experiment/text_conditioned_url_full.yaml)
# -> outputs/checkpoints/text_conditioned_url_full_medium.
# Run with: sbatch slurm/url_conditioned/train_text_conditioned_url_full.slurm
#
# Other conditioning schedules, e.g. a p_condition sweep or a cooldown, get
# their own run dir via --p-condition / --conditioning-schedule (see
# scripts/train.py) -- pass EXTRA_ARGS through --export, e.g.:
#   sbatch --export=ALL,EXTRA_ARGS="--p-condition 0.7" slurm/url_conditioned/train_text_conditioned_url_full.slurm

set -euo pipefail

cd /work/wac20/ctrl_tokens/text_conditioned
mkdir -p logs

CONDA_ENV=ctrl_tokens
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Secrets (HF_TOKEN, etc.) live in .env, which is gitignored -- see
# .env.example for the template. Never hardcode a real token in this file.
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export HF_HOME=/work/wac20/.cache/huggingface
export HF_TOKEN=${HF_TOKEN:?"set HF_TOKEN in .env (needs accepted Llama-3.2-3B license)"}

# EXTRA_ARGS is intentionally unquoted so it splits into separate flags.
torchrun --standalone --nproc_per_node=4 scripts/train.py \
  --experiment "configs/experiment/text_conditioned_url_full.yaml" \
  --out-dir outputs/checkpoints ${EXTRA_ARGS:-}
