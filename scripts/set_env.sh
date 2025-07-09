#!/bin/bash
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
set_env_path="${BASH_SOURCE[0]}"
if [[ -f "$set_env_path" ]] && [[ "$set_env_path" =~ "set_env.sh" ]]; then
    shmem_path=$(cd $(dirname $set_env_path); pwd)
    export SHMEM_HOME_PATH="$shmem_path"
    export LD_LIBRARY_PATH=$SHMEM_HOME_PATH/shmem/lib:$SHMEM_HOME_PATH/memfabric_hybrid/lib:$LD_LIBRARY_PATH
    export PATH=$SHMEM_HOME_PATH/bin:$PATH
fi