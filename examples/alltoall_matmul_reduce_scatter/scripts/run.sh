#!/bin/bash
# This script runs the AlltoallMatmulReduceScatter example.
# Usage: bash run.sh <device_ids>
#   e.g., bash run.sh 0,1      # Runs on devices 0 and 1 (rank size = 2)
#   e.g., bash run.sh 0,1,2,3  # Runs on devices 0,1,2,3 (rank size = 4)

set -e

# 1. Environment Setup
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
EXAMPLE_DIR=$(dirname "$SCRIPT_DIR")
PROJECT_ROOT=$(dirname "$EXAMPLE_DIR")
EXEC_BIN_NAME="alltoall_matmul_reduce_scatter"
EXEC_BIN_PATH="${PROJECT_ROOT}/build/bin/${EXEC_BIN_NAME}"

# Check for executable
if [ ! -f "${EXEC_BIN_PATH}" ]; then
    echo "Executable ${EXEC_BIN_PATH} not found. Please build the project first."
    exit 1
fi

# 2. Parse Arguments
DEVICE_IDS="$1"
if [ -z "$DEVICE_IDS" ]; then
    echo "Please provide a comma-separated list of device IDs."
    echo "Usage: bash run.sh <device_ids>"
    exit 1
fi

IFS=',' read -ra DEVICE_ID_LIST <<< "$DEVICE_IDS"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
echo "Running with RANK_SIZE=${RANK_SIZE} on devices: ${DEVICE_IDS}"

# 3. Process Test Shapes
CSV_FILE="${SCRIPT_DIR}/test_shapes.csv"
if [ ! -f "$CSV_FILE" ]; then
    echo "Test shapes file not found: ${CSV_FILE}"
    exit 1
fi

cd ${EXAMPLE_DIR}
tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N; do
    echo "--------------------------------------------------"
    echo "Processing test case: M=${M}, K=${K}, N=${N}"
    echo "--------------------------------------------------"

    # 4. Generate Data
    echo "Generating data..."
    rm -rf ./output
    python3 gen_data.py ${M} ${N} ${K} ${RANK_SIZE}
    echo "Data generation complete."

    # 5. Launch Processes
    # The C++ executable will read its rank from the environment.
    export RANK_SIZE=${RANK_SIZE}
    
    # Use a simple TCP port for process coordination if needed by SHMEM
    export MASTER_ADDR="127.0.0.1"
    export MASTER_PORT="29500"

    PIDS=()
    for (( i=0; i<${RANK_SIZE}; i++ )); do
        DEVICE_ID=${DEVICE_ID_LIST[$i]}
        export RANK_ID=${i}
        export ASCEND_DEVICE_ID=${DEVICE_ID}
        
        echo "Launching rank ${i} on device ${DEVICE_ID}..."
        # The executable should get M, N, K from args
        ${EXEC_BIN_PATH} ${M} ${N} ${K} > ./output/rank_${i}/output.log 2>&1 &
        PIDS+=($!)
    done

    # 6. Wait for all processes to complete
    echo "Waiting for all ranks to finish..."
    for pid in ${PIDS[*]}; do
        wait $pid
    done
    echo "All ranks finished."

    # 7. Verification Info
    echo "Verification can be done by comparing output.bin with golden.bin in each rank's output directory."
    echo "Example: diff output/rank_0/output.bin output/rank_0/golden.bin"
    echo "Note: A dedicated verification script (e.g., verify_result.py) should be used for float comparisons."

done

cd ${CURRENT_DIR}