#!/bin/bash
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

cd ${PROJECT_ROOT}/examples/matmul_allreduce/

# Generate golden data
rm -rf out/*.bin
python3 utils/gen_data.py 1 ${RANK_SIZE} ${M} ${N} ${K} 0 0

# Start Process
echo "Test Case, M: ${M}, K: ${K}, N: ${N}"
export LD_LIBRARY_PATH=${PROJECT_ROOT}/install/shmem/lib:${ASCEND_HOME_PATH}/lib64:${PROJECT_ROOT}/install/memfabric_hybrid/lib:$LD_LIBRARY_PATH
for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
    ./out/matmul_allreduce "$RANK_SIZE" "$idx" "$IPPORT" "$FIRST_NPU" "$M" "$K" "$N" &
done

# Wait until all process exit
wait

# Verify output
python3 utils/verify_result.py ./out/output.bin ./out/golden.bin 1 ${M} ${N} ${K}

cd ${CURRENT_DIR}