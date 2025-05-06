#!/bin/bash
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd ${PROJECT_ROOT}

set -e
RANK_SIZE="8"
IPPORT="tcp://127.0.0.1:8666"
GNPU_NUM="8"
if [ -z "${GTEST_FILTER}" ]; then
    TEST_FILTER="*.*"
else
    TEST_FILTER="${GTEST_FILTER}"
fi

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
        -test_filter)
            if [ -n "$2" ]; then
                TEST_FILTER="*$2*"
                shift 2
            else
                echo "Error: -test_filter requires a value."
                exit 1
            fi
            ;;
        *)
            echo "Error: Unknown option $1."
            exit 1
            ;;
    esac
done

export LD_LIBRARY_PATH=$(pwd)/install/shmem/lib:${ASCEND_HOME_PATH}/lib64:$(pwd)/install/memfabric_hybrid/lib:$LD_LIBRARY_PATH
GTEST_FILTER=${TEST_FILTER} ./build/bin/shmem_unittest "$RANK_SIZE" "$IPPORT" "$GNPU_NUM" --gtest_output=xml:test_detail.xml

cd ${CURRENT_DIR}