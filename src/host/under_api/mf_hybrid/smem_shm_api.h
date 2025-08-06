/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_MF_HYBRID_API_H
#define SHMEM_MF_HYBRID_API_H

#include <string>
#include <mutex>
#include <dlfcn.h>
#include "smem_shm.h"
#include "common/shmemi_logger.h"

namespace shm {
/* smem functions */
using smem_init_func = int32_t (*)(uint32_t);
using smem_set_extern_logger_func = int32_t (*)(void (*func)(int32_t level, const char *));
using smem_set_log_level_func = int32_t (*)(int32_t);
using smem_un_init_func = void (*)();
using smem_get_last_err_msg_func = const char *(*)();
using smem_get_and_clear_last_err_msg_func = const char *(*)();

/* smem shm functions */
using smem_shm_config_init_func = int32_t (*)(smem_shm_config_t *config);
using smem_shm_init_func = int32_t (*)(const char *, uint32_t, uint32_t, uint16_t, smem_shm_config_t *);
using smem_shm_un_init_func = void (*)(uint32_t flags);
using smem_shm_query_support_data_op_func = uint32_t (*)(void);
using smem_shm_create_func = smem_shm_t (*)(uint32_t, uint32_t, uint32_t, uint64_t, smem_shm_data_op_type, uint32_t,
                                         void **);
using smem_shm_destroy_func = int32_t (*)(smem_shm_t, uint32_t);
using smem_shm_set_extra_context_func = int32_t (*)(smem_shm_t, const void *, uint32_t);
using smem_shm_control_barrier_func = int32_t (*)(smem_shm_t);
using smem_shm_control_all_gather_func = int32_t (*)(smem_shm_t, const char *, uint32_t, char *, uint32_t);
using smem_shm_topo_can_reach_func = int32_t (*)(smem_shm_t, uint32_t, uint32_t *);

class smem_api {
public:
    static int32_t load_library(const std::string &lib_dir_path);

public:
    /* smem api */
    static inline int32_t smem_init(uint32_t flags)
    {
        if (g_smem_init == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_init is NULL");
            return -1;
        }

        return g_smem_init(flags);
    }

    static inline int32_t smem_set_extern_logger(void (*func)(int32_t level, const char *msg))
    {
        if (g_smem_set_extern_logger == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_set_extern_logger is NULL");
            return -1;
        }

        return g_smem_set_extern_logger(func);
    }

    static inline int32_t smem_set_log_level(int32_t level)
    {
        if (g_smem_set_log_level == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_set_log_level is NULL");
            return -1;
        }
        
        return g_smem_set_log_level(level);
    }

    static inline void smem_un_init()
    {
        if (g_smem_un_init == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_un_init is NULL");
            return;
        }

        g_smem_un_init();

        if (g_smem_handle) {
            dlclose(g_smem_handle);
            g_smem_handle = nullptr;
            g_loaded = false;
            SHM_LOG_INFO("dl close library in smem_un_init");
        }
        return; 
    }

    static inline const char *smem_get_last_err_msg()
    {
        if (g_smem_get_last_err_msg == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_get_last_err_msg is NULL");
            return "";
        }
        
        return g_smem_get_last_err_msg();
    }

    static inline const char *smem_get_and_clear_last_err_msg()
    {
        if (g_smem_get_and_clear_last_err_msg == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_get_and_clear_last_err_msg is NULL");
            return "";
        }
        
        return g_smem_get_and_clear_last_err_msg();
    }

    /* smem shm api */
    static inline int32_t smem_shm_config_init(smem_shm_config_t *config)
    {
        if (g_smem_shm_config_init == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_shm_config_init is NULL");
            return -1;
        }
        
        return g_smem_shm_config_init(config);
    }

    static inline int32_t smem_shm_init(const char *config_store_ipport, uint32_t world_size, uint32_t rank_id,
                                      uint16_t device_id, smem_shm_config_t *config)
    {
        if (g_smem_shm_init == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_shm_init is NULL");
            return -1;
        }
        
        return g_smem_shm_init(config_store_ipport, world_size, rank_id, device_id, config);
    }

    static inline void smem_shm_un_init(uint32_t flags)
    {
        if (g_smem_shm_un_init == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_shm_un_init is NULL");
            return;
        }
        
        return g_smem_shm_un_init(flags);
    }

    static inline uint32_t smem_shm_query_support_data_op()
    {
        if (g_smem_shm_query_support_data_op == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_shm_query_support_data_op is NULL");
            return 0;
        }
        
        return g_smem_shm_query_support_data_op();
    }

    static inline smem_shm_t smem_shm_create(uint32_t id, uint32_t rank_size, uint32_t rank_id, uint64_t symmetric_size,
                                           smem_shm_data_op_type data_op_type, uint32_t flags, void **gva)
    {
        if (g_smem_shm_create == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_shm_create is NULL");
            return 0;
        }
        
        return g_smem_shm_create(id, rank_size, rank_id, symmetric_size, data_op_type, flags, gva);
    }

    static inline int32_t smem_shm_destroy(smem_shm_t handle, uint32_t flags)
    {
        if (g_smem_shm_destroy == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_shm_destroy is NULL");
            return -1;
        }
        
        return g_smem_shm_destroy(handle, flags);
    }

    static inline int32_t smem_shm_set_extra_context(smem_shm_t handle, const void *context, uint32_t size)
    {
        if (g_smem_shm_set_extra_context == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_shm_set_extra_context is NULL");
            return -1;
        }
        
        return g_smem_shm_set_extra_context(handle, context, size);
    }

    static inline int32_t smem_shm_control_barrier(smem_shm_t team)
    {
        if (g_smem_shm_control_barrier == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_shm_control_barrier is NULL");
            return -1;
        }
        
        return g_smem_shm_control_barrier(team);
    }

    static inline int32_t smem_shm_control_all_gather(smem_shm_t team, const char *send_buf, uint32_t send_size,
                                                  char *recv_buf, uint32_t recv_size)
    {
        if (g_smem_shm_control_all_gather == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_shm_control_all_gather is NULL");
            return -1;
        }
        
        return g_smem_shm_control_all_gather(team, send_buf, send_size, recv_buf, recv_size);
    }

    static inline int32_t smem_shm_topo_can_reach(smem_shm_t handle, uint32_t remote_rank, uint32_t *reach_info)
    {
        if (g_smem_shm_topo_can_reach == nullptr) {
            SHM_LOG_ERROR("function pointer g_smem_shm_topo_can_reach is NULL");
            return -1;
        }
        
        return g_smem_shm_topo_can_reach(handle, remote_rank, reach_info);
    }

    ~smem_api();

private:
    static bool g_loaded;
    static std::mutex g_mutex;

    static void *g_smem_handle;
    static const char *g_smem_file_name;

    static smem_init_func g_smem_init;
    static smem_set_extern_logger_func g_smem_set_extern_logger;
    static smem_set_log_level_func g_smem_set_log_level;
    static smem_un_init_func g_smem_un_init;
    static smem_get_last_err_msg_func g_smem_get_last_err_msg;
    static smem_get_and_clear_last_err_msg_func g_smem_get_and_clear_last_err_msg;

    static smem_shm_config_init_func g_smem_shm_config_init;
    static smem_shm_init_func g_smem_shm_init;
    static smem_shm_un_init_func g_smem_shm_un_init;
    static smem_shm_query_support_data_op_func g_smem_shm_query_support_data_op;
    static smem_shm_create_func g_smem_shm_create;
    static smem_shm_destroy_func g_smem_shm_destroy;
    static smem_shm_set_extra_context_func g_smem_shm_set_extra_context;
    static smem_shm_control_barrier_func g_smem_shm_control_barrier;
    static smem_shm_control_all_gather_func g_smem_shm_control_all_gather;
    static smem_shm_topo_can_reach_func g_smem_shm_topo_can_reach;
};
}  // namespace shm

#endif  //SHMEM_MF_HYBRID_API_H
