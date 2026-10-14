#!/bin/bash
# Queue the pretraining-trajectory benchmarks for one scale, then the figures
# once they finish (scripts/comparison/make_trajectory.py):
#   bash slurm/comparison/submit_trajectory.sh 3b
#   NICE=100 DEPENDENCY=afterany:123 bash slurm/comparison/submit_trajectory.sh 3b
# The figures also need the main comparison's benchmarks at the shared
# checkpoint (slurm/scale/bench.slurm); pass their job id via DEPENDENCY.
set -euo pipefail
cd /work/wac20/ctrl_tokens/text_conditioned

SCALE=${1:-}
COMPARISON=configs/comparison.yaml
[[ -n "$SCALE" ]] && COMPARISON=configs/comparison_${SCALE}.yaml
JOB_ARGS=(--config "$COMPARISON")
[[ -n "${SOURCES:-}" ]] && JOB_ARGS+=(--sources ${SOURCES})

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctrl_tokens
n=$(python scripts/comparison/trajectory_jobs.py "${JOB_ARGS[@]}" --count)

DEP=()
[[ -n "${DEPENDENCY:-}" ]] && DEP=(--dependency="$DEPENDENCY")
BENCH=$(sbatch --parsable --nice="${NICE:-0}" --array=0-$((n - 1)) \
  --export=ALL,SCALE="$SCALE",SOURCES="${SOURCES:-}" slurm/comparison/bench_trajectory.slurm)
FIG_DEP=afterany:$BENCH
[[ -n "${DEPENDENCY:-}" ]] && FIG_DEP="$FIG_DEP:${DEPENDENCY#*:}"
FIGS=$(sbatch --parsable --dependency="$FIG_DEP" --export=ALL,SCALE="$SCALE" slurm/comparison/report.slurm)
echo "trajectory benchmarks: $BENCH ($n jobs)"
echo "report:                $FIGS"
