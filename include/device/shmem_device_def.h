/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_DEF_H
#define SHMEM_DEVICE_DEF_H

#include "kernel_operator.h"
#include "host_device/shmem_types.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 * @addtogroup group_macros
 * @{
*/

// Non-Contiguous Datacopy Param
struct non_contiguous_copy_param
{
    uint32_t repeat;
    uint32_t length;
    uint32_t src_ld;     // src data leading dimension. Interval between the head of the repeat and the head of the following repeat
    uint32_t dst_ld;     // dst data leading dimension
};

/**@} */ // end of group_macros

#ifdef __cplusplus
}
#endif

#endif