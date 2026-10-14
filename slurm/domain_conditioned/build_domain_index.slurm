#!/bin/bash
#SBATCH --job-name=tc_build_domain_index
#SBATCH --partition=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

# Builds the training-split URL-domain index (data/domain_index/100BT) that
# both domain-conditioned experiments read their in-domain partner documents
# from -- see scripts/build_domain_index.py / src/data/domain_index.py.
# CPU only; run once before either train_*.slurm in this folder, e.g.:
#   jid=$(sbatch --parsable slurm/domain_conditioned/build_domain_index.slurm)
#   sbatch --dependency=afterok:$jid slurm/domain_conditioned/train_text_conditioned_domain_mean.slurm
#   sbatch --dependency=afterok:$jid slurm/domain_conditioned/train_text_conditioned_domain_random.slurm

set -euo pipefail

cd /work/wac20/ctrl_tokens/text_conditioned
mkdir -p logs

CONDA_ENV=ctrl_tokens
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python scripts/build_domain_index.py --num-workers "${SLURM_CPUS_PER_TASK:-32}"
