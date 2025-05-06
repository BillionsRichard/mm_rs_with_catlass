#!/bin/bash
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $(dirname "$SCRIPT_DIR"))

cd ${SCRIPT_DIR}

export LD_LIBRARY_PATH=${PROJECT_ROOT}/install/shmem/lib:${ASCEND_HOME_PATH}/lib64:${PROJECT_ROOT}/install/memfabric_hybrid/lib:$LD_LIBRARY_PATH

RANK_SIZE="2"
IPPORT="tcp://127.0.0.1:8766"

for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
    ./out/matmul_allreduce "$RANK_SIZE" "$idx" "$IPPORT" &
done

cd ${CURRENT_DIR}