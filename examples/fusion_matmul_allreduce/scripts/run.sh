#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))

# Default Args
RANK_SIZE="2"
IPPORT="tcp://127.0.0.1:8766"
FIRST_NPU="0"

# Args Parse
while [[ $# -gt 0 ]]; do
    case "$1" in
        -ranks)
            if [ -n "$2" ]; then
                RANK_SIZE="$2"
                shift 2
            else
                echo "Error: -ranks requires a value."
                exit 1
            fi
            ;;
        -fnpu)
            if [ -n "$2" ]; then
                FIRST_NPU="$2"
                shift 2
            else
                echo "Error: -fnpu requires a value."
                exit 1
            fi
            ;;
        -ipport)
            if [ -n "$2" ]; then
                IPPORT="$2"
                shift 2
            else
                echo "Error: -ipport requires a value."
                exit 1
            fi
            ;;
        -M)
            if [ -n "$2" ]; then
                M="$2"
                shift 2
            else
                echo "Error: -M requires a value."
                exit 1
            fi
            ;;
        -K)
            if [ -n "$2" ]; then
                K="$2"
                shift 2
            else
                echo "Error: -K requires a value."
                exit 1
            fi
            ;;
        -N)
            if [ -n "$2" ]; then
                N="$2"
                shift 2
            else
                echo "Error: -N requires a value."
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1."
            exit 1
            ;;
    esac
done

cd ${PROJECT_ROOT}/examples/fusion_matmul_allreduce/

DATA_DIR=`realpath ./out`
echo "DATA_DIR: $DATA_DIR"

# Generate golden data
rm -rf out/*.bin
python3 utils/gen_data.py \
    --data_dir ./out \
    --out_data_type 1 \
    --rank_size ${RANK_SIZE}\
    --m ${M} \
    --n ${N} \
    --k ${K} \
    --transA 0 \
    --transB 0

# Start Process
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "Test Case, M: ${M}, K: ${K}, N: ${N}"
export LD_LIBRARY_PATH=${PROJECT_ROOT}/build/lib:${PROJECT_ROOT}/3rdparty/memfabric_hybrid/output/smem/lib64:${PROJECT_ROOT}/3rdparty/memfabric_hybrid/output/hybm/lib:${ASCEND_HOME_PATH}/lib64:$LD_LIBRARY_PATH
for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
    ${PROJECT_ROOT}/build/bin/fusion_matmul_allreduce "$RANK_SIZE" "$idx" "$IPPORT" "$FIRST_NPU" "$M" "$K" "$N" $DATA_DIR &
done

# Wait until all process exit
wait

# Verify output
python3 utils/verify_result.py ${DATA_DIR}/shmem_output.bin ${DATA_DIR}/golden.bin 1 ${M} ${N} ${K}

cd ${CURRENT_DIR}