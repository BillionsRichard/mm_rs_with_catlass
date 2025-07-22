/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*
    WARNING： 
    
    Our barrier implementation ensures that:
        On systems with only HCCS: All operations of all ranks of a team ON EXECUTING/INTERNAL STREAMs before the barrier are visiable to all ranks of the team after the barrier.
        
    Refer to shmem_device_sync.h for using restrictions.
*/

#ifndef SHMEM_HOST_SYNC_H
#define SHMEM_HOST_SYNC_H

#include "acl/acl.h"
#include "shmem_host_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn SHMEM_HOST_API uint64_t shmemx_get_ffts_config()
 * @brief Get runtime ffts config. This config should be passed to MIX Kernel and set by MIX Kernel using shmemx_set_ffts. Refer to shmemx_set_ffts for more details.
 *
 * @return ffts config
 *
 */
SHMEM_HOST_API uint64_t shmemx_get_ffts_config();

#ifdef __cplusplus
}
#endif

#endif