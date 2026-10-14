#!/bin/bash
#SBATCH --job-name=fw_download
#SBATCH --partition=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
mkdir -p logs

CONDA_ENV=ctrl_tokens
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd /work/wac20/ctrl_tokens/text_conditioned

# Secrets (HF_TOKEN, etc.) live in .env, which is gitignored -- see
# .env.example for the template. Never hardcode a real token in this file.
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# FineWeb itself is ungated; HF_TOKEN just raises your rate limit (safe to leave unset).
export HF_HOME=/work/wac20/.cache/huggingface
export HF_TOKEN=${HF_TOKEN:-}

# Resumable: snapshot_download skips files it already has, so a re-run after a
# timeout/preemption just picks up where it left off.
#   sbatch slurm/download_fineweb.slurm                                   # FineWeb -> data/fineweb_raw
#   sbatch --export=ALL,DATASET=fineweb-edu slurm/download_fineweb.slurm  # FineWeb-Edu -> data/fineweb_edu_raw
DATASET=${DATASET:-fineweb}
python -m src.data.download_fineweb --dataset "$DATASET" --sample "${SAMPLE:-100BT}" \
  --out-dir "./data/${DATASET//-/_}_raw"
