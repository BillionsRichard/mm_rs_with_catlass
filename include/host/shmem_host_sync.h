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

SHMEM_HOST_API uint64_t shmemx_get_ffts_config();

/**
 * @brief barrier of a team on specific stream
 *
 * @param tid              [in] team to do barrier
 * @param stream           [in] stream the barrier will be executed on
 */
SHMEM_HOST_API void shmem_barrier_on_stream(shmem_team_t tid, aclrtStream stream);

/**
 * @brief barrier of all PEs on specific stream
 *
 * @param stream           [in] stream the barrier will be executed on
 */
SHMEM_HOST_API void shmem_barrier_all_on_stream(aclrtStream stream);

/**
 * @fn SHMEM_HOST_API void shmem_barrier(shmem_team_t tid)
 * @brief Both the host and device have a function named <b>shmem_barrier()</b> and has different meanings, which is distinguished by prefix macros SHMEM_HOST_API and SHMEM_DEVICE.
 *        <br>
 *        On the host side, this method is a barrier of team on default stream
 *
 * @param tid              [in] team to do barrier
 */
SHMEM_HOST_API void shmem_barrier(shmem_team_t tid);

/**
 * @fn SHMEM_HOST_API void shmem_barrier_all()
 * @brief Both the host and device have a function named <b>shmem_barrier_all()</b> and has different meanings, which is distinguished by prefix macros SHMEM_HOST_API and SHMEM_DEVICE.
 *        <br>
 *        On the host side, this method is a barrier of all PEs on default stream
 */
SHMEM_HOST_API void shmem_barrier_all();

#ifdef __cplusplus
}
#endif

#endif