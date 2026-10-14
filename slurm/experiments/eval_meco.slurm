#!/bin/bash
#SBATCH --job-name=tc_eval_meco
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# Final conditioning evaluation for the "meco" run (outputs/checkpoints/meco_p0.9_medium,
# trained by slurm/experiments/train_meco.slurm). Set RUN_DIR via --export to
# evaluate another run instead, e.g. the original p_condition=1.0 MeCo run:
#   sbatch --export=ALL,RUN_DIR=outputs/checkpoints/meco_medium slurm/experiments/eval_meco.slurm
# Run with: sbatch slurm/experiments/eval_meco.slurm

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

RUN_DIR=${RUN_DIR:-outputs/checkpoints/meco_p0.9_medium}

python scripts/evaluate.py --run-dir "$RUN_DIR" --mode final
