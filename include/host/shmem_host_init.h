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
 * @return Returns initialization status. Returning SHMEM_STATUS_IS_INITIALIZED indicates that initialization is complete. All return types can be found in <b>\ref shmem_init_status_t</b>.
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
 * @brief Initialize the resources required for SHMEM task based on attributes.
 *        Attributes can be created by users or obtained by calling <b>shmem_set_attr()</b>.
 *        if the self-created attr structure is incorrect, the initialization will fail.
 *        It is recommended to build the attributes by <b>shmem_set_attr()</b>. 
 *
 * @param attributes        [in] Pointer to the user-defined attributes.
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int shmem_init_attr(shmem_init_attr_t *attributes);

/**
 * @brief Register a decrypt key password handler.
 *
 * @param decrypt_handler decrypt function pointer
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int32_t shmem_register_decrypt_handler(const shmem_decrypt_handler decrypt_handler);

/**
 * @brief Set the log print function for the SHMEM library.
 *
 * @param func the logging function, takes level and msg as parameter
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int32_t shmem_set_extern_logger(void (*func)(int level, const char *msg));

/**
 * @brief Set the logging level.
 *
 * @param level the logging level. 0-debug, 1-info, 2-warn, 3-error
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int32_t shmem_set_log_level(int level);

/**
 * @brief Initialize the config store tls info.
 *
 * @param enable whether to enable tls
 * @param tls_info the format describle in memfabric SECURITYNOTE.md, if disabled tls_info won't be use
 * @param tls_info_len length of tls_info, if disabled tls_info_len won't be use
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int32_t shmem_set_conf_store_tls(bool enable, const char *tls_info, const uint32_t tls_info_len);

/**
 * @brief Release all resources used by the SHMEM library.
 *
 * @return Returns 0 on success or an error code on failure
 */
SHMEM_HOST_API int shmem_finalize();

#ifdef __cplusplus
}
#endif

#endif