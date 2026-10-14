#!/bin/bash
#SBATCH --job-name=tc_bench_meco
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

# lm-eval-harness benchmarking for the "meco" run (outputs/checkpoints/meco_p0.9_medium): its
# unconditioned mode, plus MeCo-conditioned on every named source in
# configs/benchmarks/sources.yaml (wikipedia/reddit/stackexchange/news/github).
# One array task per (mode, source) combination -- each runs as its own SLURM
# job on its own GPU rather than looping sequentially in one job.
# Run with: sbatch slurm/benchmarks/bench_meco.slurm
#
# Set RUN_DIR via --export to benchmark another run instead, e.g. the original
# p_condition=1.0 MeCo run:
#   sbatch --export=ALL,RUN_DIR=outputs/checkpoints/meco_medium slurm/benchmarks/bench_meco.slurm
#
# Set LIMIT (docs/task) via --export for a fast smoke test before committing
# to a full run, e.g.:
#   sbatch --export=LIMIT=200 slurm/benchmarks/bench_meco.slurm

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

# SLURM_ARRAY_TASK_ID -> (--mode, --source): unconditioned, then every named
# source in configs/benchmarks/sources.yaml, MeCo-conditioned.
COMBOS=(
  "none"
  "meco wikipedia"
  "meco reddit"
  "meco stackexchange"
  "meco news"
  "meco github"
)
read -r MODE SOURCE <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

ARGS=(--run-dir "$RUN_DIR" --mode "$MODE")
[[ -n "$SOURCE" ]] && ARGS+=(--source "$SOURCE")
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "${LIMIT}")

python scripts/benchmarks/run_benchmarks.py "${ARGS[@]}"
