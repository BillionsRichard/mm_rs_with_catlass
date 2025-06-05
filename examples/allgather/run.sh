#!/bin/bash
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $(dirname "$SCRIPT_DIR"))

cd ${SCRIPT_DIR}

export LD_LIBRARY_PATH=${SCRIPT_DIR}/build/lib:${PROJECT_ROOT}/install/shmem/lib:${ASCEND_HOME_PATH}/lib64:${PROJECT_ROOT}/install/memfabric_hybrid/lib:$LD_LIBRARY_PATH

RANK_SIZE="2"
IPPORT="tcp://127.0.0.1:8766"
GNPU_NUM="8"
FIRST_NPU="0"
FIRST_RANK="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -ranks)
            if [ -n "$2" ]; then
                RANK_SIZE="$2"
                if [ "$GNPU_NUM" -gt "$RANK_SIZE" ]; then
                    GNPU_NUM="$RANK_SIZE"
                    echo "Because GNPU_NUM is greater than RANK_SIZE, GNPU_NUM is assigned the value of RANK_SIZE=${RANK_SIZE}."
                fi
                shift 2
            else
                echo "Error: -ranks requires a value."
                exit 1
            fi
            ;;
        -frank)
            if [ -n "$2" ]; then
                FIRST_RANK="$2"
                shift 2
            else
                echo "Error: -frank requires a value."
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
        -gnpus)
            if [ -n "$2" ]; then
                GNPU_NUM="$2"
                shift 2
            else
                echo "Error: -gnpus requires a value."
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
        *)
            echo "Error: Unknown option $1."
            exit 1
            ;;
    esac
done

for (( idx =0; idx < ${GNPU_NUM}; idx = idx + 1 )); do
    ./build/bin/allgather "$RANK_SIZE" "$idx" "$IPPORT" "$GNPU_NUM" "$FIRST_RANK" "$FIRST_NPU" &
done

cd ${CURRENT_DIR}