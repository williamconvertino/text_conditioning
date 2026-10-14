#!/bin/bash
# Queue every fine-tuning job for one scale, then the report once they finish
# (README "Fine-Tuning"):
#   bash slurm/finetune/submit_all.sh            # the medium runs (configs/comparison.yaml)
#   bash slurm/finetune/submit_all.sh 1.5b       # configs/comparison_1.5b.yaml
#   DEPENDENCY=afterok:123 bash slurm/finetune/submit_all.sh 3b   # wait for other jobs first
#   SEEDS="1 2" bash slurm/finetune/submit_all.sh                  # more training seeds (default 1234)
#
# 3 runs x 3 methods (standard, conditioned, peft) x 9 tasks = 81 jobs, 1 GPU
# each. At 8B, full fine-tuning (standard, conditioned) needs ~128GB of
# params + grads + AdamW state, so those jobs get 4 GPUs with FSDP; peft
# trains one vector and stays on 1 GPU. METHODS / TASKS / RUNS / EXTRA_ARGS
# are passed through (see slurm/finetune/finetune.slurm); TIME sets the
# per-job limit (default 12:00:00; ~2h is plenty at the medium scale).
set -euo pipefail
cd /work/wac20/ctrl_tokens/text_conditioned

SCALE=${1:-}
read -r -a RUN_LIST <<< "${RUNS:-baseline meco url}"
read -r -a METHOD_LIST <<< "${METHODS:-standard conditioned peft}"
read -r -a TASK_LIST <<< "${TASKS:-hellaswag piqa arc_easy arc_challenge winogrande boolq openbookqa sciq_closed mmlu_cont}"
read -r -a SEED_LIST <<< "${SEEDS:-1234}"
DEP=()
[[ -n "${DEPENDENCY:-}" ]] && DEP=(--dependency="$DEPENDENCY")

submit() {  # submit <methods> <gpus>
  local methods=$1 gpus=$2
  local n=$(( ${#RUN_LIST[@]} * $(wc -w <<< "$methods") * ${#TASK_LIST[@]} * ${#SEED_LIST[@]} ))
  sbatch --parsable "${DEP[@]}" --array=0-$((n - 1)) --gres=gpu:"$gpus" --cpus-per-task=$((8 * gpus)) \
    --time="${TIME:-12:00:00}" \
    --mem=$((128 * gpus))G \
    --export=ALL,SCALE="$SCALE",RUNS="${RUN_LIST[*]}",METHODS="$methods",TASKS="${TASK_LIST[*]}",SEEDS="${SEED_LIST[*]}",NPROC="$gpus" \
    slurm/finetune/finetune.slurm
}

JOBS=()
if [[ "$SCALE" == "8b" ]]; then
  full=() peft=()
  for m in "${METHOD_LIST[@]}"; do
    if [[ "$m" == peft* ]]; then peft+=("$m"); else full+=("$m"); fi
  done
  (( ${#full[@]} )) && JOBS+=("$(submit "${full[*]}" 4)")
  (( ${#peft[@]} )) && JOBS+=("$(submit "${peft[*]}" 1)")
else
  JOBS+=("$(submit "${METHOD_LIST[*]}" 1)")
fi
echo "fine-tuning arrays: ${JOBS[*]}"

dep=$(IFS=:; echo "${JOBS[*]}")
rep=$(sbatch --parsable --dependency=afterany:"$dep" --export=ALL,SCALE="$SCALE" slurm/finetune/report.slurm)
echo "report: $rep"
