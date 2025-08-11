#!/bin/bash
# eg. bash run_quant.sh 0,1      # 在 0/1 卡上运行，rank size = 2
# eg. bash run_quant.sh 1,3,5,7  # 在 1/3/5/6 卡上运行，rank size = 4

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))
UTILS_PATH=${PROJECT_ROOT}/examples/utils

CSV_FILE="${SCRIPT_DIR}/test_quant_shapes.csv"

IFS=',' read -ra DEVICE_ID_LIST <<< "$1"
RANK_SIZE=${#DEVICE_ID_LIST[@]}
if [ $RANK_SIZE -gt 8 ]; then
    echo "Rank size is illegal"
    exit 1
fi

cd ${PROJECT_ROOT}/examples/02_matmul_reduce_scatter/
EXEC_BIN=${PROJECT_ROOT}/build/bin/02_quant_matmul_reduce_scatter

test_idx=0
mkdir -p output
tail -n +2 "$CSV_FILE" | while IFS=',' read -r M K N; do
    # pkill -9  02_quant_matmul
    # sleep 0.1
    let test_idx+=1
    echo "Processing test case: M=${M}, K=${K}, N=${N}"

    # Generate golden data
    rm -rf output/*.bin
    python3 ${UTILS_PATH}/gen_quant_data.py 2 1 ${RANK_SIZE} ${M} ${N} ${K} 0 0

    # Set necessary parameters
    IPPORT="tcp://127.0.0.1:47530"

    # Start Process
    # for (( rank_idx = 0; rank_idx < ${RANK_SIZE}; rank_idx = rank_idx + 1 )); do
    ${EXEC_BIN} "$RANK_SIZE" "1" "$IPPORT" "$M" "$N" "$K" "$1"
    # done

    # Wait until all process exit
    wait

    # Verify output
    python3 ${UTILS_PATH}/verify_quant_result.py ./output/output.bin ./output/golden.bin 1 ${M} ${N} ${K}
done

cd ${CURRENT_DIR}