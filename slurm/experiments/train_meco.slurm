#!/bin/bash
#SBATCH --job-name=tc_train_meco
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out

# Trains the "meco" experiment (configs/experiment/meco.yaml) with a constant
# conditioning_schedule of p_condition=0.9 (see src/training/schedules.py),
# aligned with text_conditioned / text_conditioned_url. --p-condition tags the
# run dir -> outputs/checkpoints/meco_p0.9_medium, so this doesn't resume the finished
# p_condition=1.0 run in outputs/checkpoints/meco_medium.
# Run with: sbatch slurm/experiments/train_meco.slurm
#
# Other conditioning schedules get their own run dir the same way -- pass
# EXTRA_ARGS through --export (a later --p-condition overrides this one), e.g.:
#   sbatch --export=ALL,EXTRA_ARGS="--p-condition 0.7" slurm/experiments/train_meco.slurm

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
  --experiment "configs/experiment/meco.yaml" \
  --p-condition 0.9 \
  --out-dir outputs/checkpoints ${EXTRA_ARGS:-}
