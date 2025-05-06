#!/bin/bash
BULID_DIR="build"
if [ ! -d "$BULID_DIR" ]; then
    mkdir "$BULID_DIR"
fi
cd $BULID_DIR

cmake ..
make
cd ..

RANK_SIZE="8"
IPPORT="tcp://127.0.0.1:8666"

export LD_LIBRARY_PATH=../../install/shmem/lib:${ASCEND_HOME_PATH}/lib64:../../install/memfabric_hybrid/lib:$LD_LIBRARY_PATH
for (( idx =0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
    ./build/bin/demo "$RANK_SIZE" "$idx" "$IPPORT" &
done