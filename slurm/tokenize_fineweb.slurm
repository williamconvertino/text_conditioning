#!/bin/bash
#SBATCH --job-name=fw_tokenize
#SBATCH --partition=scavenger
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
mkdir -p logs

CONDA_ENV=ctrl_tokens
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd /work/wac20/ctrl_tokens/text_conditioned

# meta-llama/Llama-3.2-3B is GATED: HF_TOKEN must belong to an account that has
# accepted its license, or tokenizer loading will fail with a 401.
# Secrets live in .env, which is gitignored -- see .env.example for the
# template. NEVER hardcode a token in this file (a real token was previously
# committed here; it has been removed and should be treated as
# compromised/rotated).
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export HF_HOME=/work/wac20/.cache/huggingface
export HF_TOKEN=${HF_TOKEN:?"set HF_TOKEN in .env (needs accepted Llama-3.2-3B license)"}
export TOKENIZERS_PARALLELISM=false

# CPU-only, array-parallel (no GPU needed): #SBATCH --array=0-N-1 above must
# match --num-workers below. Each task tokenizes a strided subset of shards and
# is independently resumable (a completed shard's .done marker is skipped on
# re-run), so a failed/preempted array task can just be resubmitted on its own.
#
# NOTE: an earlier run under the previous codebase already wrote a manifest.json
# (with 0 shards done) at
#   /work/wac20/ctrl_tokens/experiments/data/fineweb_tokenized/100BT
# Point --output-dir there instead of ./data/fineweb_tokenized/100BT if you'd
# rather resume that location than start a fresh cache -- the .bin/_index.parquet
# format written here is byte-for-byte identical either way.
#
# DATASET=fineweb-edu tokenizes data/fineweb_edu_raw into data/fineweb_edu_tokenized
# (same format; slurm/data/prepare_fineweb_edu.sh chains download -> this -> metadata).
DATASET=${DATASET:-fineweb}
SAMPLE=${SAMPLE:-100BT}
python -m src.data.tokenize_fineweb \
  --input-dir "./data/${DATASET//-/_}_raw/sample/${SAMPLE}" \
  --output-dir "./data/${DATASET//-/_}_tokenized/${SAMPLE}" \
  --tokenizer meta-llama/Llama-3.2-3B \
  --max-doc-tokens 4096 \
  --num-workers "${SLURM_ARRAY_TASK_COUNT:-16}" \
  --worker-id "${SLURM_ARRAY_TASK_ID}"
