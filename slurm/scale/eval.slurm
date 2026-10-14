#!/bin/bash
#SBATCH --job-name=tc_scale_eval
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Final perplexity/conditioning evaluation (scripts/evaluate.py --mode final)
# of one scaled run. One GPU is enough at every scale (8B in fp32 is ~32GB).
#   sbatch --export=ALL,RUN_DIR=outputs/checkpoints/meco_p0.9_3b slurm/scale/eval.slurm

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

RUN_DIR=${RUN_DIR:?"set RUN_DIR (e.g. outputs/checkpoints/baseline_1b)"}
python scripts/evaluate.py --run-dir "$RUN_DIR" --mode final ${EXTRA_ARGS:-}
