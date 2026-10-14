#!/bin/bash
#SBATCH --job-name=fw_tokenize_meta
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out

# Pretokenizes every shard's URL metadata (domain + full URL) into
# <TOK_DIR>/metadata/ (src/data/metadata_tokens.py), so packed MeCo /
# URL-encoded training never runs the tokenizer. Resumable: finished shards
# are skipped.
#   sbatch slurm/tokenize_metadata.slurm
#   sbatch --export=ALL,TOK_DIR=data/fineweb_tokenized/other slurm/tokenize_metadata.slurm

set -euo pipefail
cd /work/wac20/ctrl_tokens/text_conditioned
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctrl_tokens
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
export HF_HOME=/work/wac20/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

python -m src.data.metadata_tokens --tok-dir "${TOK_DIR:-data/fineweb_tokenized/100BT}" \
  --workers "${SLURM_CPUS_PER_TASK:-32}"
