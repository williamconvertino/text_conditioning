#!/bin/bash
#SBATCH --job-name=tc_1.5b_throughput
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x-%j.out

# Single-GPU throughput / peak-memory check of the packed 1.5B config
# (configs/scale/1.5b.yaml) at several micro-batch sizes; writes nothing but
# a scratch run dir. Compare the "s/it" and "mem" columns.
set -euo pipefail
cd /work/wac20/ctrl_tokens/text_conditioned
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctrl_tokens
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export HF_HOME=/work/wac20/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCRATCH=${SCRATCH_DIR:-/tmp/tc_bench_$SLURM_JOB_ID}
nvidia-smi --query-gpu=name,memory.total --format=csv

for EXP in ${EXPERIMENTS:-text_conditioned_url_full baseline}; do
  for LB in ${LOCAL_BATCHES:-4 8 16}; do
    echo "=== $EXP local_batch=$LB"
    python scripts/train.py --experiment configs/experiment/$EXP.yaml --scale 1.5b \
      --out-dir "$SCRATCH" --run-name "${EXP}_lb$LB" --no-resume --max-steps 12 \
      --set local_batch=$LB --set global_batch_seqs=$((LB * 2)) --set log_interval=2 \
      --set val_interval=100000 --set val_docs=64 --set save_checkpoints=false \
      --set ckpt_save_interval_tokens=1000000000000 2>&1 | grep -vE "^\s*$" | tail -12 || echo "FAILED"
  done
done
rm -rf "$SCRATCH"
