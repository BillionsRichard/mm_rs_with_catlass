/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acl/acl.h"
#include "kernel_operator.h"

#include "shmem_api.h"

// kernels
SHMEM_GLOBAL void k_shmem_barrier(int32_t tid) {

    shmemi_barrier<false>(tid);
} 

// interfaces
int32_t shmemi_barrier_on_stream(shmem_team_t tid, aclrtStream stream) {
    // TODO: clear all internal working streams

    // call barrier kernel
    k_shmem_barrier<<<1, nullptr, stream>>>((int32_t) tid);
    return aclrtSynchronizeStream(stream);
}