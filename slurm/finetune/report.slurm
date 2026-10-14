#!/bin/bash
#SBATCH --job-name=tc_finetune_report
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out

# Fine-tuning figures and tables (scripts/finetune/make_finetune_report.py),
# CPU only, from saved results; safe to rerun at any time.
#   sbatch slurm/finetune/report.slurm
#   sbatch --export=ALL,SCALE=8b slurm/finetune/report.slurm

set -euo pipefail

cd /work/wac20/ctrl_tokens/text_conditioned
mkdir -p logs

CONDA_ENV=ctrl_tokens
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

COMPARISON=configs/comparison.yaml
[[ -n "${SCALE:-}" ]] && COMPARISON=configs/comparison_${SCALE}.yaml
python scripts/finetune/make_finetune_report.py --comparison-config "$COMPARISON"
