#!/bin/bash
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))

RANKSIZE=$1
M=$2
K=$3
N=$4

cd ${PROJECT_ROOT}/examples/matmul_allreduce/

rm -rf out/*.bin
python3 utils/gen_data.py 1 ${RANKSIZE} ${M} ${N} ${K} 0 0

export LD_LIBRARY_PATH=${PROJECT_ROOT}/install/shmem/lib:${ASCEND_HOME_PATH}/lib64:${PROJECT_ROOT}/install/memfabric_hybrid/lib:$LD_LIBRARY_PATH

RANK_SIZE="${RANKSIZE}"
IPPORT="tcp://127.0.0.1:8766"

echo "Test Case, M: ${M}, K: ${K}, N: ${N}"

for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
    ./out/matmul_allreduce "$RANK_SIZE" "$idx" "$IPPORT" "$M" "$K" "$N" &
done

wait

python3 utils/verify_result.py ./out/output.bin ./out/golden.bin 1 ${M} ${N} ${K}

cd ${CURRENT_DIR}