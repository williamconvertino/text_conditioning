#!/bin/bash
#SBATCH --job-name=tc_scale_bench
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --array=0-12

# slurm/comparison/bench_comparison.slurm for one scale: lm-eval benchmarks
# for the scale's Baseline / MeCo / URL runs (configs/comparison_<SCALE>.yaml)
# at their shared checkpoint, unconditioned and steered toward each source.
#   sbatch --export=ALL,SCALE=1.5b slurm/scale/bench.slurm
# Runs missing from the comparison config are skipped; the array then only
# needs one task per remaining combination, e.g. URL run only (none + 5 sources):
#   sbatch --array=0-5 --export=ALL,SCALE=3b slurm/scale/bench.slurm

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

SCALE=${SCALE:?"set SCALE (1.5b | 3b | 8b)"}
COMPARISON=configs/comparison_${SCALE}.yaml
run_dir() { python -c "import yaml, sys; print(yaml.safe_load(open('$COMPARISON'))['runs'].get(sys.argv[1], ''))" "$1"; }
BASELINE=$(run_dir baseline)
MECO=$(run_dir meco)
URL=$(run_dir url)

COMBOS=()
[[ -n "$BASELINE" ]] && COMBOS+=("$BASELINE none")
[[ -n "$MECO" ]] && COMBOS+=("$MECO none" "$MECO meco wikipedia" "$MECO meco reddit"
                             "$MECO meco stackexchange" "$MECO meco news" "$MECO meco github")
[[ -n "$URL" ]] && COMBOS+=("$URL none" "$URL text wikipedia" "$URL text reddit"
                            "$URL text stackexchange" "$URL text news" "$URL text github")
if (( SLURM_ARRAY_TASK_ID >= ${#COMBOS[@]} )); then
  echo "array task $SLURM_ARRAY_TASK_ID: only ${#COMBOS[@]} combinations for $COMPARISON, nothing to do"
  exit 0
fi
read -r RUN_DIR MODE SOURCE <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"

ARGS=(--run-dir "$RUN_DIR" --checkpoint shared --comparison-config "$COMPARISON" --mode "$MODE")
[[ -n "${SOURCE:-}" ]] && ARGS+=(--source "$SOURCE")
[[ -n "${LIMIT:-}" ]] && ARGS+=(--limit "${LIMIT}")

python scripts/benchmarks/run_benchmarks.py "${ARGS[@]}"
