#!/bin/bash
CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))

echo ${PROJECT_ROOT}

MEMFABRIC_INCLUDE_PATH=$PROJECT_ROOT/install/output/memfabric_hybrid/include/smem/
MEMFABRIC_LIB_PATH=$PROJECT_ROOT/install/output/memfabric_hybrid/lib/

SHMEM_INCLUDE_PATH=$PROJECT_ROOT/install/output/shmem/include/
SHMEM_LIB_PATH=$PROJECT_ROOT/install/output/shmem/lib/

cd ${PROJECT_ROOT}/examples/fusion_matmul_allreduce/

export LD_LIBRARY_PATH=${PROJECT_ROOT}/install/output/shmem/lib:${ASCEND_HOME_PATH}/lib64:${PROJECT_ROOT}/install/output/memfabric_hybrid/lib:$LD_LIBRARY_PATH

mkdir -p out
bisheng -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220             \
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
    -I$PROJECT_ROOT/3rdparty/catlass/include/                       \
    -I$PROJECT_ROOT/3rdparty/catlass/examples/common/               \
    -I$PROJECT_ROOT/include/                                        \
    -I$MEMFABRIC_INCLUDE_PATH/host/                                 \
    -I$MEMFABRIC_INCLUDE_PATH/device/                               \
    -I./                                                            \
    -L${ASCEND_HOME_PATH}/lib64                                     \
    -L$MEMFABRIC_LIB_PATH/ -lmf_smem -lmf_hybm_core                 \
    -L$SHMEM_LIB_PATH/ -lshmem                 \
    -Wno-macro-redefined -Wno-ignored-attributes                    \
    -lruntime -lstdc++ -lascendcl -lm -ltiling_api                  \
    -lplatform -lc_sec -ldl -lnnopbase                              \
    -lascendc_runtime -lerror_manager -lprofapi -lmmpa -lascendalog \
    ${PROJECT_ROOT}/examples/fusion_matmul_allreduce/main.cpp -o out/fusion_matmul_allreduce

cd ${CURRENT_DIR}