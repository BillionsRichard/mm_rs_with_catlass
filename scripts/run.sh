#!/bin/bash
Current_path=$(pwd)
cd ..

rm -rf test_scalar_p
cp ./build/bin/test_scalar_p ./

rm -rf team_example
cp ./build/bin/team_example ./

RANK_SIZE="8"
IP_PORT="tcp://127.0.0.1:8666"
export LD_LIBRARY_PATH=$(pwd)/install/lib:${ASCEND_HOME_PATH}/lib64:$(pwd)/3rdparty/output:$LD_LIBRARY_PATH

for (( idx = 0; idx < ${RANK_SIZE}; idx = idx + 1 )); do
    ./test_scalar_p ${RANK_SIZE} ${idx} ${IP_PORT} &
    # ./team_example ${RANK_SIZE} ${idx} ${IP_PORT} &
done

cd ${Current_path}