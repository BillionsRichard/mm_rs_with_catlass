#!/bin/bash
Current_path=$(pwd)
cd ..

rm -rf team_example
cp ./build/bin/team_example ./

export LD_LIBRARY_PATH=$(pwd)/install/lib:${ASCEND_HOME_PATH}/lib64:$(pwd)/3rdparty/output:$LD_LIBRARY_PATH
# mpirun --allow-run-as-root -n 8 ./team_example

RANK_SIZE="8"
IP_PORT="tcp://127.0.0.1:8666"
echo " ====== example run, ranksize: ${RANK_SIZE} ip: ${IP_PORT} ======"

rm -f scalar_putget_kernels
cp ./build/bin/scalar_putget_kernels ./

for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
    ./scalar_putget_kernels ${RANK_SIZE} ${idx} ${IP_PORT} &
done

cd ${Current_path}