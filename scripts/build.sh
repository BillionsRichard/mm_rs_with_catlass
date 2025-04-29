#!/bin/bash
if [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
fi

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}

source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

CURRENT_DIR=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
THIRD_PARTY_DIR=$PROJECT_ROOT/3rdparty
cd ${PROJECT_ROOT}

function fn_build_googletest()
{
    if [ -d "$THIRD_PARTY_DIR/googletest/lib" ]; then
        return 0
    fi
    cd $THIRD_PARTY_DIR
    [[ ! -d "googletest" ]] && git clone --branch v1.14.0 --depth 1 https://github.com/google/googletest.git
    cd googletest
    rm -rf build && mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/googletest -DCMAKE_SKIP_RPATH=TRUE -DCMAKE_CXX_FLAGS="-fPIC"
    cmake --build . --parallel $(nproc)
    cmake --install . > /dev/null
    [[ -d "$THIRD_PARTY_DIR/googletest/lib64" ]] && cp -rf $THIRD_PARTY_DIR/googletest/lib64 $THIRD_PARTY_DIR/googletest/lib
    echo "Googletest is successfully installed to $THIRD_PARTY_DIR/googletest"
    cd ${PROJECT_ROOT}
}
set -e
fn_build_googletest
rm -rf build
mkdir -p build

cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make install -j8
cd -

MEMFABRIC_INCLUDE_PATH=$PROJECT_ROOT/3rdparty/memfabric_hybrid/include/smem/
MEMFABRIC_LIB_PATH=$PROJECT_ROOT/3rdparty/memfabric_hybrid/lib/

SHMEM_INCLUDE_PATH=$PROJECT_ROOT/include/
SHMEM_LIB_PATH=$PROJECT_ROOT/install/lib/

cd examples/matmul_allreduce
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
    -I$PROJECT_ROOT/3rdparty/ascendc-templates/examples/common      \
    -I$PROJECT_ROOT/examples/include/                               \
    -I$PROJECT_ROOT/include/                                        \
    -I$MEMFABRIC_INCLUDE_PATH/host/                                 \
    -I$MEMFABRIC_INCLUDE_PATH/device/                               \
    -I$SHMEM_INCLUDE_PATH/host/                                     \
    -I$SHMEM_INCLUDE_PATH/host_device/                              \
    -I$SHMEM_INCLUDE_PATH/device/                                   \
    -I$PROJECT_ROOT/3rdparty/ascendc-templates/include/             \
    -I..                                                            \
    -I.                                                             \
    -L${ASCEND_HOME_PATH}/lib64                                     \
    -L$MEMFABRIC_LIB_PATH/ -lmf_smem -lmf_hybm_core                 \
    -L$SHMEM_LIB_PATH/ -lshmem_host -lshmem_device                  \
    -Wno-macro-redefined -Wno-ignored-attributes                    \
    -lruntime -lstdc++ -lascendcl -lm -ltiling_api                  \
    -lplatform -lc_sec -ldl -lnnopbase                              \
    main.cpp -o out/matmul_allreduce

cd ${CURRENT_DIR}