#!/bin/bash
#SBATCH --job-name=tc_bench_trajectory
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%A_%a.out

# Benchmarks across pretraining (scripts/comparison/make_trajectory.py): every
# token-milestone snapshot of the compared runs, unconditioned and steered
# toward each source -- one array task per line of
# scripts/comparison/trajectory_jobs.py. Use slurm/comparison/submit_trajectory.sh,
# which sizes --array and queues the figures; by hand:
#   n=$(python scripts/comparison/trajectory_jobs.py --config configs/comparison_3b.yaml --count)
#   sbatch --array=0-$((n - 1)) --export=ALL,SCALE=3b slurm/comparison/bench_trajectory.slurm
# Env (optional): SCALE (default: configs/comparison.yaml), SOURCES (e.g. "wikipedia reddit").
# Finished (run, milestone, conditioning) results are skipped, so resubmitting
# after a failure only redoes what is missing.

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
JOB_ARGS=(--config "$COMPARISON")
[[ -n "${SOURCES:-}" ]] && JOB_ARGS+=(--sources ${SOURCES})
mapfile -t JOBS < <(python scripts/comparison/trajectory_jobs.py "${JOB_ARGS[@]}")
if (( SLURM_ARRAY_TASK_ID >= ${#JOBS[@]} )); then
  echo "array task $SLURM_ARRAY_TASK_ID: only ${#JOBS[@]} jobs for $COMPARISON, nothing to do"
  exit 0
fi
read -r RUN_DIR PIN MODE SOURCE <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
echo "run=$RUN_DIR checkpoint=$PIN mode=$MODE source=${SOURCE:-} comparison=$COMPARISON"

ARGS=(--run-dir "$RUN_DIR" --checkpoint "$PIN" --mode "$MODE" --skip-existing)
[[ -n "${SOURCE:-}" ]] && ARGS+=(--source "$SOURCE")
python scripts/benchmarks/run_benchmarks.py "${ARGS[@]}"
