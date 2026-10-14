#!/bin/bash
# Optional convenience: queue the whole encoder-analysis + steering pipeline
# with dependencies (README "Encoder Analysis & Steering"). Every script also
# works on its own via sbatch -- each resolves the shared checkpoint itself.
#
#   bench (13) ───────────────────────────────────────────────────┐
#   encoder_analysis ─┬─ mixtures (6) ────────────────────────────┤
#                     └─ task_vectors (30) ─ task_transfer (2) ───┴─ report
#
# Run once every run in configs/comparison.yaml has finished training:
#   bash slurm/steering/submit_all.sh
# Another scale (configs/comparison_<scale>.yaml): array sizes follow the runs
# that config lists (e.g. only the URL run at 3B so far):
#   bash slurm/steering/submit_all.sh 3b

set -euo pipefail
cd /work/wac20/ctrl_tokens/text_conditioned
mkdir -p logs

SCALE=${1:-}
COMPARISON=configs/comparison.yaml
[[ -n "$SCALE" ]] && COMPARISON=configs/comparison_${SCALE}.yaml

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctrl_tokens
python scripts/comparison/show_checkpoint.py --config "$COMPARISON"

read -r -a RUN_LIST <<< "$(python -c "import yaml; print(' '.join(yaml.safe_load(open('$COMPARISON'))['runs']))")"
has() { [[ " ${RUN_LIST[*]} " == *" $1 "* ]]; }
TV_RUNS=() TRANSFER_RUNS=() N_BENCH=0
for r in baseline meco url; do has "$r" && TV_RUNS+=("$r"); done
for r in url meco; do has "$r" && TRANSFER_RUNS+=("$r"); done
has baseline && N_BENCH=$((N_BENCH + 1))
has meco && N_BENCH=$((N_BENCH + 6))
has url && N_BENCH=$((N_BENCH + 6))

sub() { sbatch --parsable "$@"; }
EXPORT=ALL,SCALE="$SCALE"
if [[ -n "$SCALE" ]]; then
  BENCH=$(sub --array=0-$((N_BENCH - 1)) --export="$EXPORT" slurm/scale/bench.slurm)
else
  BENCH=$(sub slurm/comparison/bench_comparison.slurm)
fi
ANALYSIS=$(sub --export="$EXPORT" slurm/url_analysis/encoder_analysis.slurm)
MIX=$(sub --dependency=afterok:"$ANALYSIS" --export="$EXPORT" slurm/steering/mixtures.slurm)
TV=$(sub --dependency=afterok:"$ANALYSIS" --array=0-$((10 * ${#TV_RUNS[@]} - 1)) \
  --export="$EXPORT",RUNS="${TV_RUNS[*]}" slurm/steering/task_vectors.slurm)
TRANSFER=$(sub --dependency=afterany:"$TV" --array=0-$((${#TRANSFER_RUNS[@]} - 1)) \
  --export="$EXPORT",RUNS="${TRANSFER_RUNS[*]}" slurm/steering/task_transfer.slurm)
REPORT=$(sub --dependency=afterany:"$BENCH:$MIX:$TRANSFER" --export="$EXPORT" slurm/comparison/report.slurm)

echo "bench:            $BENCH"
echo "encoder_analysis: $ANALYSIS"
echo "mixtures:         $MIX"
echo "task_vectors:     $TV"
echo "task_transfer:    $TRANSFER"
echo "report:           $REPORT"
