#!/bin/bash
#SBATCH --job-name=tc_steer_report
#SBATCH --partition=common
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out

# Every table/figure for the shared-checkpoint comparison, the encoder analysis and both
# steering experiments (CPU only, reads saved results). Safe to rerun anytime.
# Run with: sbatch slurm/comparison/report.slurm
# Another scale: sbatch --export=ALL,SCALE=3b slurm/comparison/report.slurm

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
COMPARISON=configs/comparison.yaml
[[ -n "${SCALE:-}" ]] && COMPARISON=configs/comparison_${SCALE}.yaml
python scripts/comparison/make_comparison.py --config "$COMPARISON" || true
python scripts/comparison/make_trajectory.py --config "$COMPARISON" || true
python scripts/url_analysis/plot_encoder_analysis.py --comparison-config "$COMPARISON" || true
python scripts/steering/make_steering_report.py --comparison-config "$COMPARISON"
