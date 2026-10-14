#!/bin/bash
#SBATCH --job-name=tc_scale_train
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=768G
#SBATCH --time=60:00:00
#SBATCH --output=logs/%x-%j.out

# Pretrains one experiment at a Llama 3 scale (configs/scale/<SCALE>.yaml,
# README "Scaling Up") -> outputs/checkpoints/<experiment>[_p<P>]_<SCALE>. Resume is
# automatic, so a run can continue over any number of these jobs. Each job
# stops itself STOP_MARGIN_MIN minutes (default 45) before its time limit:
# it validates, writes ckpt_last.pt and exits (TRAIN_STOP_AT in
# src/training/trainer.py), so hitting the limit loses no work.
#
#   sbatch --export=ALL,SCALE=1.5b,EXPERIMENT=baseline slurm/scale/train.slurm
#   sbatch --export=ALL,SCALE=3b,EXPERIMENT=meco,P_CONDITION=0.9 slurm/scale/train.slurm
#   sbatch --export=ALL,SCALE=1.5b,EXPERIMENT=baseline,DATA=fineweb_edu slurm/scale/train.slurm
#   sbatch --nodes=2 --export=ALL,SCALE=8b,EXPERIMENT=text_conditioned_url_full slurm/scale/train.slurm
#
# Multi-node (--nodes=N): one torchrun per node, rendezvous on the first node.
# The scale configs fix global_batch_docs, so grad_accum shrinks as GPUs are
# added and every GPU count trains the same optimizer steps. Keep the GPU
# count fixed for the life of a run: the data cursor is per rank.

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
EXPERIMENT=${EXPERIMENT:?"set EXPERIMENT (e.g. baseline | meco | text_conditioned_url_full)"}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
ARGS=(--experiment "configs/experiment/${EXPERIMENT}.yaml" --scale "$SCALE" --out-dir outputs/checkpoints)
[[ -n "${P_CONDITION:-}" ]] && ARGS+=(--p-condition "$P_CONDITION")
[[ -n "${DATA:-}" ]] && ARGS+=(--data "$DATA")        # configs/data/<DATA>.yaml, e.g. fineweb_edu

# stop cleanly before the time limit (d-hh:mm:ss / hh:mm:ss / mm:ss)
LIMIT=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
export TRAIN_STOP_AT=$(python -c "
import sys, time
d, _, hms = sys.argv[1].rpartition('-')
parts = [int(x) for x in hms.split(':')]
while len(parts) < 3: parts.insert(0, 0)
secs = int(d or 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]
print(int(time.time() + secs - 60 * int(sys.argv[2])))" "$LIMIT" "${STOP_MARGIN_MIN:-45}")
echo "time limit $LIMIT -> TRAIN_STOP_AT $(date -d @"$TRAIN_STOP_AT")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
TORCHRUN=$(command -v torchrun)
launch() {
  # EXTRA_ARGS is intentionally unquoted so it splits into separate flags.
  if (( SLURM_NNODES == 1 )); then
    "$TORCHRUN" --standalone --nproc_per_node="$GPUS_PER_NODE" \
      scripts/train.py "${ARGS[@]}" ${EXTRA_ARGS:-} "${@:2}"
  else
    # --export=ALL: srun otherwise launches with sbatch's submit-time
    # environment (SLURM_EXPORT_ENV), which lacks the conda env
    srun --export=ALL --ntasks="$SLURM_NNODES" --ntasks-per-node=1 "$TORCHRUN" \
      --nnodes="$SLURM_NNODES" --nproc_per_node="$GPUS_PER_NODE" \
      --rdzv_id="$SLURM_JOB_ID-$1" --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:29500" \
      scripts/train.py "${ARGS[@]}" ${EXTRA_ARGS:-} "${@:2}"
  fi
}

# If the run dies (e.g. an OOM on a rare worst-case batch), resume from
# ckpt_last.pt with the memory-safe settings for the rest of the job instead
# of leaving the allocation idle. FALLBACK_ARGS="" disables this. Changing
# local_batch keeps the data stream: documents per step are fixed by
# global_batch_docs.
FALLBACK_ARGS=${FALLBACK_ARGS---set gradient_checkpointing=true --set local_batch=8}
if ! launch main; then
  if [[ -n "$FALLBACK_ARGS" ]] && (( $(date +%s) < TRAIN_STOP_AT - 1800 )); then
    echo "training exited with an error; resuming with: $FALLBACK_ARGS"
    read -r -a FB <<< "$FALLBACK_ARGS"
    launch fallback "${FB[@]}"
  else
    exit 1
  fi
fi
