#!/bin/bash
#SBATCH --job-name=tc_bench
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# lm-eval-harness benchmarking for one (run, conditioning) combination
# (scripts/benchmarks/run_benchmarks.py). Pass args via sbatch --export, e.g.:
#
#   sbatch --export=RUN_DIR=outputs/checkpoints/baseline_medium,MODE=none \
#       slurm/benchmarks/run_benchmarks.slurm
#
#   sbatch --export=RUN_DIR=outputs/checkpoints/meco_medium,MODE=meco,SOURCE=wikipedia \
#       slurm/benchmarks/run_benchmarks.slurm
#
#   sbatch --export=RUN_DIR=outputs/checkpoints/text_conditioned_medium,MODE=text,SOURCE=reddit \
#       slurm/benchmarks/run_benchmarks.slurm
#
# DOMAIN / URL / TAG / LIMIT are optional and map to the matching --flag.

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

export HF_HOME=/work/wac20/.cache/huggingface
export HF_TOKEN=${HF_TOKEN:?"set HF_TOKEN in .env (needs accepted Llama-3.2-3B license)"}

RUN_DIR=${RUN_DIR:?"set RUN_DIR (e.g. sbatch --export=RUN_DIR=outputs/checkpoints/baseline_medium,MODE=none slurm/benchmarks/run_benchmarks.slurm)"}
MODE=${MODE:?"set MODE (none | meco | text)"}

ARGS=(--run-dir "${RUN_DIR}" --mode "${MODE}")
[[ -n "${SOURCE:-}" ]] && ARGS+=(--source "${SOURCE}")
[[ -n "${DOMAIN:-}" ]] && ARGS+=(--domain "${DOMAIN}")
[[ -n "${URL:-}" ]] && ARGS+=(--url "${URL}")
[[ -n "${TAG:-}" ]] && ARGS+=(--tag "${TAG}")
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "${LIMIT}")

python scripts/benchmarks/run_benchmarks.py "${ARGS[@]}"
