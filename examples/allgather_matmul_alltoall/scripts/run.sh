#!/bin/bash
# Description: One-click script for allgather_matmul_alltoall operator
#
# Usage:
# bash run.sh 0,1      # Run on devices 0 and 1 (rank size = 2)
# bash run.sh 0,1,2,3  # Run on devices 0,1,2,3 (rank size = 4)

set -e

# 1. Environment Setup
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))
OPERATOR_DIR=$(dirname "$SCRIPT_DIR")
EXEC_BIN_NAME="allgather_matmul_alltoall"
EXEC_BIN="${PROJECT_ROOT}/build/bin/${EXEC_BIN_NAME}"

# Check if executable exists
if [ ! -f "$EXEC_BIN" ]; then
    echo "Executable ${EXEC_BIN} not found. Please build the project first."
    exit 1
fi

# Data and output directory
DATA_DIR="${OPERATOR_DIR}/data"
mkdir -p "${DATA_DIR}"
rm -rf "${DATA_DIR}"/*

# 2. Parse Arguments
DEVICE_ID_STRING="$1"
if [ -z "$DEVICE_ID_STRING" ]; then
    echo "Usage: bash run.sh <device_id_list>"
    echo "Example: bash run.sh 0,1,2,3"
    exit 1
fi

IFS=',' read -ra DEVICE_ID_LIST <<< "$DEVICE_ID_STRING"
RANK_SIZE=${#DEVICE_ID_LIST[@]}

# 3. Test Case Configuration
# For simplicity, test shapes are defined here. A CSV file can be used for multiple shapes.
M=128
N=256
K=64

echo "--- Test Case ---"
echo "RANK_SIZE: ${RANK_SIZE}"
echo "DEVICE_IDS: ${DEVICE_ID_STRING}"
echo "M: ${M}, N: ${N}, K: ${K}"
echo "-----------------"

# 4. Generate Data and Golden Reference
echo "[STEP 1] Generating input data and golden reference..."
python3 "${OPERATOR_DIR}/gen_data.py" "${RANK_SIZE}" "${M}" "${N}" "${K}" "${DATA_DIR}"
echo "Data generation complete."

# 5. Run the Operator
echo -e "\n[STEP 2] Running the ${EXEC_BIN_NAME} operator for ${RANK_SIZE} ranks..."
IPPORT="tcp://127.0.0.1:28888" # Use a unique port

pids=()
for (( i=0; i<${RANK_SIZE}; i++ )); do
    RANK_ID=$i
    echo "  Starting rank ${RANK_ID} on device ${DEVICE_ID_LIST[$i]}..."
    # The main executable takes: rank_size, rank_id, ipport, m, n, k, data_path, device_id_list
    ${EXEC_BIN} "${RANK_SIZE}" "${RANK_ID}" "${IPPORT}" "${M}" "${N}" "${K}" "${DATA_DIR}" "${DEVICE_ID_STRING}" &
    pids+=($!)
done

# Wait for all background processes to finish
wait "${pids[@]}"
echo "All ranks have finished execution."

# 6. Verify Results
echo -e "\n[STEP 3] Verifying results..."
python3 "${OPERATOR_DIR}/verify_result.py" "${RANK_SIZE}" "${M}" "${N}" "${K}" "${DATA_DIR}"

if [ $? -eq 0 ]; then
    echo -e "\nVerification PASSED"
else
    echo -e "\nVerification FAILED"
    exit 1
fi

cd "${CURRENT_DIR}"
exit 0
