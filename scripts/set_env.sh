#!/bin/bash
set_env_path="${BASH_SOURCE[0]}"
if [[ -f "$set_env_path" ]] && [[ "$set_env_path" =~ "set_env.sh" ]]; then
    shmem_path=$(cd $(dirname $set_env_path); pwd)
    export SHMEM_HOME_PATH="$shmem_path"
    export LD_LIBRARY_PATH=$SHMEM_HOME_PATH/shmem/lib:LD_LIBRARY_PATH=$SHMEM_HOME_PATH/memfabric_hybrid/lib:$LD_LIBRARY_PATH
    export PATH=$SHMEM_HOME_PATH/bin:$PATH
fi