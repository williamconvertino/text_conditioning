#!/bin/bash
#SBATCH --job-name=tc_check_3b
#SBATCH --account=henaolab_h200
#SBATCH --partition=h200-hp
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1:30:00
#SBATCH --output=logs/%x-%j.out

# Throughput / peak-memory check for the 3B URL-encoded run (README "Scaling
# Up"): a few steps each of DDP vs FSDP, with and without activation
# checkpointing, with and without length bucketing, on real data. Per-GPU
# memory under DDP does not depend on the GPU count, so 2 GPUs answer "does
# 3B fit under DDP on an H200". Nothing is saved (save_checkpoints=false).
#   sbatch slurm/scale/check_3b_parallelism.slurm

set -uo pipefail
cd /work/wac20/ctrl_tokens/text_conditioned
mkdir -p logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ctrl_tokens
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export HF_HOME=/work/wac20/.cache/huggingface

COMMON=(--experiment configs/experiment/text_conditioned_url_full.yaml --scale 3b --no-resume
        --max-steps 8 --set log_interval=1 --set val_docs=32 --set val_interval=100000
        --set save_checkpoints=false --set ckpt_save_interval_tokens=1000000000000
        --set global_batch_docs=256 --set warmup_steps=100)
CONFIGS=(
  "ddp_ckpt_lb8        fsdp=false gradient_checkpointing=true  local_batch=8 length_bucketing=true"
  "ddp_nockpt_lb4      fsdp=false gradient_checkpointing=false local_batch=4 length_bucketing=true"
  "ddp_nockpt_lb8      fsdp=false gradient_checkpointing=false local_batch=8 length_bucketing=true"
  "fsdp_ckpt_lb8       fsdp=true  gradient_checkpointing=true  local_batch=8 length_bucketing=true"
  "ddp_ckpt_lb8_padded fsdp=false gradient_checkpointing=true  local_batch=8 length_bucketing=false"
)
for c in "${CONFIGS[@]}"; do
  read -r name rest <<< "$c"
  sets=()
  for kv in $rest; do sets+=(--set "$kv"); done
  echo "===== $name"
  torchrun --standalone --nproc_per_node=2 scripts/train.py "${COMMON[@]}" "${sets[@]}" \
    --out-dir outputs/_check_3b --run-name "$name" 2>&1 | grep -E "params|^step|Error|error|OutOfMemory" | tail -12
  python - "$name" <<'PY'
import json, sys
recs = [json.loads(l) for l in open(f"outputs/_check_3b/{sys.argv[1]}/metrics.jsonl") if '"train"' in l]
tail = [r for r in recs if r["step"] >= 3]
if tail:
    sps = sum(r["sec_per_step"] for r in tail) / len(tail)
    dtok = (tail[-1]["tokens"] - recs[1]["tokens"]) / (tail[-1]["step"] - recs[1]["step"])
    print(f"RESULT {sys.argv[1]}: {sps:.2f} s/step  {dtok / sps / 2:.0f} real tok/s/GPU  "
          f"peak mem {max(r.get('max_mem_gb', 0) for r in recs):.1f} GiB")
PY
done
rm -rf outputs/_check_3b
