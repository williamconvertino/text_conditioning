#!/bin/bash
#SBATCH --job-name=tc_finetune
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out

# Fine-tuning experiments (configs/finetune/default.yaml, README
# "Fine-Tuning"): one array task per (run, method, task), each recording
# accuracy vs step. Use slurm/finetune/submit_all.sh, which sizes --array and
# the GPUs; by hand, with the defaults below (3 runs x 3 methods x 9 tasks):
#   sbatch --array=0-80 slurm/finetune/finetune.slurm
#   sbatch --array=0-80 --export=ALL,SCALE=1.5b slurm/finetune/finetune.slurm
#
# Env (all optional):
#   SCALE       1.5b | 3b | 8b -> configs/comparison_<SCALE>.yaml (default: configs/comparison.yaml)
#   RUNS        default "baseline meco url"
#   METHODS     default "standard conditioned peft"
#   TASKS       default: the 9 benchmark tasks (add "all" for the joint jobs)
#   SEEDS       training seeds, default "1234" (one array task per seed too)
#   NPROC       GPUs per job; > 1 runs under torchrun with FSDP (full fine-tuning at 8B)
#   EXTRA_ARGS  passed through, e.g. "--set objective=lm"

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
read -r -a RUN_LIST <<< "${RUNS:-baseline meco url}"
read -r -a METHOD_LIST <<< "${METHODS:-standard conditioned peft}"
read -r -a TASK_LIST <<< "${TASKS:-hellaswag piqa arc_easy arc_challenge winogrande boolq openbookqa sciq_closed mmlu_cont}"
read -r -a SEED_LIST <<< "${SEEDS:-1234}"

# index = ((run * methods + method) * tasks + task) * seeds + seed
n_s=${#SEED_LIST[@]}; n_t=${#TASK_LIST[@]}; n_m=${#METHOD_LIST[@]}
i=$SLURM_ARRAY_TASK_ID
SEED=${SEED_LIST[$(( i % n_s ))]}; i=$(( i / n_s ))
TASK=${TASK_LIST[$(( i % n_t ))]}; i=$(( i / n_t ))
METHOD=${METHOD_LIST[$(( i % n_m ))]}; i=$(( i / n_m ))
RUN=${RUN_LIST[$i]}
echo "run=$RUN method=$METHOD task=$TASK seed=$SEED comparison=$COMPARISON"

ARGS=(--comparison-config "$COMPARISON" --model "$RUN" --method "$METHOD" --task "$TASK" --seed "$SEED")
NPROC=${NPROC:-1}
# EXTRA_ARGS is intentionally unquoted so it splits into separate flags.
if (( NPROC > 1 )); then
  torchrun --standalone --nproc_per_node="$NPROC" scripts/finetune/run_finetune.py "${ARGS[@]}" \
    --set fsdp=true ${EXTRA_ARGS:-}
else
  python scripts/finetune/run_finetune.py "${ARGS[@]}" ${EXTRA_ARGS:-}
fi
