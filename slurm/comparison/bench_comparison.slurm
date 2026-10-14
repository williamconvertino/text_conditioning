#!/bin/bash
#SBATCH --job-name=tc_bench_comparison
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-12

# The standard lm-eval suite for the three compared runs (configs/comparison.yaml)
# at the furthest checkpoint they all share (scripts/comparison/show_checkpoint.py):
# Baseline unconditioned; MeCo and URL-encoded (full URL) unconditioned plus
# conditioned on every named source in configs/benchmarks/sources.yaml. One
# array task per (run, mode, source). Results land in
# outputs/experiments/<size>/benchmarks/<run>/<ckpt label>/<tag>/results.json; tabulate with
# scripts/comparison/make_comparison.py.
#
# Run with: sbatch slurm/comparison/bench_comparison.slurm
# Smoke test:               sbatch --export=ALL,LIMIT=200 slurm/comparison/bench_comparison.slurm

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

BASELINE=outputs/checkpoints/baseline_medium
MECO=outputs/checkpoints/meco_p0.9_medium
URL=outputs/checkpoints/text_conditioned_url_full_medium

COMBOS=(
  "$BASELINE none"
  "$MECO none"
  "$MECO meco wikipedia"
  "$MECO meco reddit"
  "$MECO meco stackexchange"
  "$MECO meco news"
  "$MECO meco github"
  "$URL none"
  "$URL text wikipedia"
  "$URL text reddit"
  "$URL text stackexchange"
  "$URL text news"
  "$URL text github"
)
read -r RUN_DIR MODE SOURCE <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

ARGS=(--run-dir "$RUN_DIR" --checkpoint shared --mode "$MODE")
[[ -n "${SOURCE:-}" ]] && ARGS+=(--source "$SOURCE")
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "${LIMIT}")

python scripts/benchmarks/run_benchmarks.py "${ARGS[@]}"
