/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_SHMEMI_HOST_COMMON_H
#define SHMEM_SHMEMI_HOST_COMMON_H

#include "shmem_api.h"

#include "common/shmemi_logger.h"
#include "common/shmemi_functions.h"
#include "init/shmemi_init.h"
#include "team/shmemi_team.h"
#include "mem/shmemi_mm.h"
#include "sync/shmemi_sync.h"

// smem api
#include <smem_bm_def.h>
#include <smem_bm.h>
#include <smem.h>
#include <smem_security.h>
#include <smem_shm_def.h>
#include <smem_shm.h>
#include <smem_trans_def.h>
#include <smem_trans.h>

#endif // SHMEM_SHMEMI_HOST_COMMON_H
