/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_HOST_INIT_H
#define SHMEM_HOST_INIT_H

#include "shmem_host_def.h"
#include "host_device/shmem_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Query the current initialization status.
 *
 * @return Returns initialization status. Returning SHMEM_STATUS_IS_INITIALIZED indicates that initialization is complete.
 */
SHMEM_HOST_API int shmem_init_status();

/**
 * @brief Set the default attributes to be used in <b>shmem_init_attr()</b>.
 *
 * @param my_rank            [in] Current rank
 * @param n_ranks            [in] Total number of ranks
 * @param local_mem_size      [in] The size of shared memory currently occupied by current rank
 * @param ip_port            [in] The ip and port number of the sever, e.g. tcp://ip:port
 * @param attributes        [out] Pointer to the default attributes used for initialization
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int shmem_set_attr(int my_rank, int n_ranks, uint64_t local_mem_size, const char* ip_port, shmem_init_attr_t **attributes);

/**
 * @brief Modify the data operation engine type in the attributes that will be used for initialization.
 *        If this method is not used, the default data_op_engine_type value is SHMEM_DATA_OP_MTE
 *        if method <b>shmem_set_attr()</b> is used after this method, the data_op_engine_type param will be overwritten by the default value.
 *
 * @param attributes        [in/out] Pointer to the attributes to modify the data operation engine type
 * @param value             [in] Value of data operation engine type
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int shmem_set_data_op_engine_type(shmem_init_attr_t *attributes, data_op_engine_type_t value);

/**
 * @brief Modify the timeout in the attributes that will be used for initialization.
 *        If this method is not used, the default timeout value is 120
 *        if method <b>shmem_set_attr()</b> is used after this method, the timeout param will be overwritten by the default value.
 *
 * @param attributes        [in/out] Pointer to the attributes to modify the data operation engine type
 * @param value             [in] Value of timeout
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int shmem_set_timeout(shmem_init_attr_t *attributes, uint32_t value);

/**
 * @brief Initialization based on attributes and build the shmem library.
 *        Attributes can be created by users or obtained by calling <b>shmem_set_attr()</b>.
 *        if the self-created attr structure is incorrect, the initialization will fail.
 *        It is recommended to build the attributes by <b>shmem_set_attr()</b>. 
 *
 * @param attributes        [in] Pointer to the user-defined attributes.
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int shmem_init_attr(shmem_init_attr_t *attributes);

/**
 * @brief Ends the program previously initialized by <b>shmem_init_attr()</b>.
 *        release all resources used by the SHMEM library.
 *
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int shmem_finalize();

#ifdef __cplusplus
}
#endif

#endif