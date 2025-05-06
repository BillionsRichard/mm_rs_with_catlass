#!/bin/bash
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $(dirname "$SCRIPT_DIR"))

echo ${PROJECT_ROOT}

MEMFABRIC_INCLUDE_PATH=$PROJECT_ROOT/3rdparty/memfabric_hybrid/include/smem/
MEMFABRIC_LIB_PATH=$PROJECT_ROOT/3rdparty/memfabric_hybrid/lib/

SHMEM_INCLUDE_PATH=$PROJECT_ROOT/install/shmem/include/
SHMEM_LIB_PATH=$PROJECT_ROOT/install/shmem/lib/

cd ${SCRIPT_DIR}

export LD_LIBRARY_PATH=${PROJECT_ROOT}/install/shmem/lib:${ASCEND_HOME_PATH}/lib64:${PROJECT_ROOT}/install/memfabric_hybrid/lib:$LD_LIBRARY_PATH

mkdir -p out
ccec -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220                \
    -mllvm -cce-aicore-stack-size=0x8000                            \
    -mllvm -cce-aicore-function-stack-size=0x8000                   \
    -mllvm -cce-aicore-record-overflow=true                         \
    -mllvm -cce-aicore-addr-transform                               \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false                 \
    -DL2_CACHE_HINT                                                 \
    -I${ASCEND_HOME_PATH}/compiler/tikcpp                           \
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw                    \
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl               \
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface          \
    -I${ASCEND_HOME_PATH}/include                                   \
    -I${ASCEND_HOME_PATH}/include/experiment/runtime                \
    -I${ASCEND_HOME_PATH}/include/experiment/msprof                 \
    -I$PROJECT_ROOT/3rdparty/ascendc-templates/include/             \
    -I$PROJECT_ROOT/3rdparty/ascendc-templates/examples/common/     \
    -I$PROJECT_ROOT/include/                                        \
    -I${PROJECT_ROOT}/src/host/                                     \
    -I${PROJECT_ROOT}/src/host/common                               \
    -I${PROJECT_ROOT}/src/host/init                                 \
    -I${PROJECT_ROOT}/src/host/mem                                  \
    -I${PROJECT_ROOT}/src/host/team                                 \
    -I$MEMFABRIC_INCLUDE_PATH/host/                                 \
    -I$MEMFABRIC_INCLUDE_PATH/device/                               \
    -I$SHMEM_INCLUDE_PATH/host/                                     \
    -I$SHMEM_INCLUDE_PATH/host_device/                              \
    -I$SHMEM_INCLUDE_PATH/device/                                   \
    -I./include/                                                    \
    -L${ASCEND_HOME_PATH}/lib64                                     \
    -L$MEMFABRIC_LIB_PATH/ -lmf_smem -lmf_hybm_core                 \
    -L$SHMEM_LIB_PATH/ -lshmem_host -lshmem_device                  \
    -Wno-macro-redefined -Wno-ignored-attributes                    \
    -lruntime -lstdc++ -lascendcl -lm -ltiling_api                  \
    -lplatform -lc_sec -ldl -lnnopbase                              \
    main.cpp -o out/matmul_allreduce

# Generate Matmul_AllReduce Golden
python3 utils/gen_data.py 1 2 1024 1024 16 0 0

RANK_SIZE="2"
IPPORT="tcp://127.0.0.1:8766"

for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
    ./out/matmul_allreduce "$RANK_SIZE" "$idx" "$IPPORT" &
done

# Verify Matmul_AllReduce Output
python3 utils/verify_result.py ./out/output.bin ./out/golden.bin 1 1024 1024 16

cd ${CURRENT_DIR}