#!/bin/bash
#SBATCH --job-name=tc_bench_text_conditioned_url
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-5

# lm-eval-harness benchmarking for the "text_conditioned_url" run
# (outputs/checkpoints/text_conditioned_url_medium): its unconditioned (null-vector) mode, plus
# steered in vector space toward every named source in
# configs/benchmarks/sources.yaml: z = E(the source's meco_domain), the same
# string MeCo prepends (README "URL-Encoded Conditioning"). One array task per
# (mode, source) combination. For ad hoc steering strings, run
# scripts/benchmarks/run_benchmarks.py --mode text --domain <string> directly.
# Run with: sbatch slurm/url_conditioned/bench_text_conditioned_url.slurm
#
# Set LIMIT (docs/task) via --export for a fast smoke test, e.g.:
#   sbatch --export=ALL,LIMIT=200 slurm/url_conditioned/bench_text_conditioned_url.slurm

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

RUN_DIR=${RUN_DIR:-outputs/checkpoints/text_conditioned_url_medium}

# SLURM_ARRAY_TASK_ID -> (--mode, --source): unconditioned, then every named
# source in configs/benchmarks/sources.yaml, URL-vector-conditioned.
COMBOS=(
  "none"
  "text wikipedia"
  "text reddit"
  "text stackexchange"
  "text news"
  "text github"
)
read -r MODE SOURCE <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

ARGS=(--run-dir "$RUN_DIR" --mode "$MODE")
[[ -n "$SOURCE" ]] && ARGS+=(--source "$SOURCE")
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "${LIMIT}")

python scripts/benchmarks/run_benchmarks.py "${ARGS[@]}"
