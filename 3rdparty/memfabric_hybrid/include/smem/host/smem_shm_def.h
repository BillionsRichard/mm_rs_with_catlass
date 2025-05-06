/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */
#ifndef __MEMFABRIC_SMEM_DEF_H_
#define __MEMFABRIC_SMEM_DEF_H_

#include "stdint.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *smem_shm_t;
typedef void *smem_shm_team_t;

typedef enum {
    SMEMS_DATA_OP_MTE = 1U << 0,
    SMEMS_DATA_OP_SDMA = 1U << 1,
    SMEMS_DATA_OP_ROCE = 1U << 2,
} smem_shm_data_op_type;

typedef struct {
    uint32_t shmInitTimeout;
    uint32_t shmCreateTimeout;
    uint32_t controlOperationTimeout;
    bool startConfigStore;
    uint32_t flags;
} smem_shm_config_t;

#ifdef __cplusplus
}
#endif
#endif // __MEMFABRIC_SMEM_DEF_H_