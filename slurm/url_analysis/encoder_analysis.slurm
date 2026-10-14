#!/bin/bash
#SBATCH --job-name=tc_encoder_analysis
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out

# URL-encoder latent-space analysis (configs/url_analysis/encoder_analysis.yaml)
# of the URL-encoded run at the furthest checkpoint all compared runs share: encodes the domain
# library (also needed by slurm/steering/*), runs every analysis, then draws
# the figures/tables into outputs/results/<size>/url_analysis/<ckpt label>/.
# Run with: sbatch slurm/url_analysis/encoder_analysis.slurm
# Another scale: sbatch --export=ALL,SCALE=3b slurm/url_analysis/encoder_analysis.slurm

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
python scripts/url_analysis/run_encoder_analysis.py --comparison-config "$COMPARISON"
python scripts/url_analysis/plot_encoder_analysis.py --comparison-config "$COMPARISON"
