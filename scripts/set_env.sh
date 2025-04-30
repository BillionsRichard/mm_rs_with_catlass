#!/bin/bash
set_env_path="${BASH_SOURCE[0]}"
if [[ -f "$set_env_path" ]] && [[ "$set_env_path" =~ "set_env.sh" ]]; then
    shmem_path=$(cd $(dirname $set_env_path); pwd)
    export SHMEM_HOME_PATH="$shmem_path"
fi