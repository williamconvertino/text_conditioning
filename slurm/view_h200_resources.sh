#!/bin/bash

PARTITION="h200-hp"

printf "%-30s %-12s %6s %6s %6s %8s %8s %10s\n" \
    "NODE" "STATE" "GPUS" "FREE" "HP" "NON-HP" "OTHER" "CPU FREE"

printf "%-30s %-12s %6s %6s %6s %8s %8s %10s\n" \
    "------------------------------" "------------" "------" "------" \
    "------" "--------" "--------" "----------"

total_gpus=0
total_free=0
total_hp=0
total_nonhp=0
total_cpus=0
total_cpu_free=0

# Get every node belonging to the partition.
nodes=$(sinfo -p "$PARTITION" -N -h -o "%N" | sort -u)

for node in $nodes; do

    # Get node information from Slurm.
    info=$(scontrol show node -o "$node")

    state=$(echo "$info" | grep -o 'State=[^ ]*' | cut -d= -f2 | cut -d+ -f1)

    cpu_total=$(echo "$info" | grep -o 'CPUTot=[0-9]*' | cut -d= -f2)
    cpu_alloc=$(echo "$info" | grep -o 'CPUAlloc=[0-9]*' | cut -d= -f2)

    cpu_total=${cpu_total:-0}
    cpu_alloc=${cpu_alloc:-0}
    cpu_free=$((cpu_total - cpu_alloc))

    # Get total GPU count from CfgTRES.
    cfg_tres=$(echo "$info" | grep -o 'CfgTRES=[^ ]*' | cut -d= -f2-)

    gpu_total=$(echo "$cfg_tres" \
        | tr ',' '\n' \
        | grep '^gres/gpu=' \
        | head -1 \
        | cut -d= -f2)

    # Fall back to typed GPU field if gres/gpu isn't present.
    if [[ -z "$gpu_total" ]]; then
        gpu_total=$(echo "$cfg_tres" \
            | tr ',' '\n' \
            | grep '^gres/gpu:h200=' \
            | head -1 \
            | cut -d= -f2)
    fi

    gpu_total=${gpu_total:-0}

    # Get Slurm's authoritative number of allocated GPUs.
    alloc_tres=$(echo "$info" | grep -o 'AllocTRES=[^ ]*' | cut -d= -f2-)

    gpu_alloc=$(echo "$alloc_tres" \
        | tr ',' '\n' \
        | grep '^gres/gpu=' \
        | head -1 \
        | cut -d= -f2)

    if [[ -z "$gpu_alloc" ]]; then
        gpu_alloc=$(echo "$alloc_tres" \
            | tr ',' '\n' \
            | grep '^gres/gpu:h200=' \
            | head -1 \
            | cut -d= -f2)
    fi

    gpu_alloc=${gpu_alloc:-0}
    gpu_free=$((gpu_total - gpu_alloc))

    #
    # Classify currently-running GPU jobs on the node by partition.
    #
    # %P = partition
    # %b = GRES requested by the job
    #
    hp_gpus=0
    nonhp_gpus=0

    while IFS='|' read -r job_partition gres; do

        [[ -z "$job_partition" ]] && continue

        # Examples:
        #   gpu:4
        #   gpu:h200:4
        #   gpu:h200:4(S:0-1)
        #
        count=$(echo "$gres" | awk -F',' '
        {
            total = 0

            for (i = 1; i <= NF; i++) {
                n = split($i, a, ":")

                if (a[1] == "gpu") {
                    if (n >= 3)
                        total += a[3]
                    else if (n >= 2)
                        total += a[2]
                }
            }

            print total
        }')

        count=${count:-0}

        if [[ "$job_partition" == "$PARTITION" ]]; then
            hp_gpus=$((hp_gpus + count))
        else
            nonhp_gpus=$((nonhp_gpus + count))
        fi

    done < <(
        squeue \
            -w "$node" \
            -t RUNNING \
            -h \
            -o "%P|%b"
    )

    # Sometimes Slurm reports allocated resources that aren't represented by
    # RUNNING jobs (e.g. COMPLETING jobs or transitional states).
    classified=$((hp_gpus + nonhp_gpus))
    other=$((gpu_alloc - classified))

    if (( other < 0 )); then
        other=0
    fi

    printf "%-30s %-12s %6d %6d %6d %8d %8d %4d/%-5d\n" \
        "$node" \
        "$state" \
        "$gpu_total" \
        "$gpu_free" \
        "$hp_gpus" \
        "$nonhp_gpus" \
        "$other" \
        "$cpu_free" \
        "$cpu_total"

    total_gpus=$((total_gpus + gpu_total))
    total_free=$((total_free + gpu_free))
    total_hp=$((total_hp + hp_gpus))
    total_nonhp=$((total_nonhp + nonhp_gpus))
    total_cpus=$((total_cpus + cpu_total))
    total_cpu_free=$((total_cpu_free + cpu_free))

done

echo
echo "GPU summary: $total_free/$total_gpus free | $total_hp HP used | $total_nonhp non-HP used"
echo "CPU summary: $total_cpu_free/$total_cpus free"