/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */
#include <dlfcn.h>
#include "smem_shm_api.h"
#include "shmemi_host_common.h"

namespace shm {
bool smem_api::g_loaded = false;
std::mutex smem_api::g_mutex;

void *smem_api::g_smem_handle = nullptr;
const char *smem_api::g_smem_file_name = "libmf_smem.so";

/* smem api define */
smem_init_func smem_api::g_smem_init = nullptr;
smem_set_extern_logger_func smem_api::g_smem_set_extern_logger = nullptr;
smem_set_log_level_func smem_api::g_smem_set_log_level = nullptr;
smem_un_init_func smem_api::g_smem_un_init = nullptr;
smem_get_last_err_msg_func smem_api::g_smem_get_last_err_msg = nullptr;
smem_get_and_clear_last_err_msg_func smem_api::g_smem_get_and_clear_last_err_msg = nullptr;

/* smem shm api define */
smem_shm_config_init_func smem_api::g_smem_shm_config_init = nullptr;
smem_shm_init_func smem_api::g_smem_shm_init = nullptr;
smem_shm_un_init_func smem_api::g_smem_shm_un_init = nullptr;
smem_shm_query_support_data_op_func smem_api::g_smem_shm_query_support_data_op = nullptr;
smem_shm_create_func smem_api::g_smem_shm_create = nullptr;
smem_shm_destroy_func smem_api::g_smem_shm_destroy = nullptr;
smem_shm_set_extra_context_func smem_api::g_smem_shm_set_extra_context = nullptr;
smem_shm_get_global_team_func smem_api::g_smem_shm_get_global_team = nullptr;
smem_shm_team_get_rank_func smem_api::g_smem_shm_team_get_rank = nullptr;
smem_shm_team_get_size_func smem_api::g_smem_shm_team_get_size = nullptr;
smem_shm_control_barrier_func smem_api::g_smem_shm_control_barrier = nullptr;
smem_shm_control_all_gather_func smem_api::g_smem_shm_control_all_gather = nullptr;
smem_shm_topo_can_reach_func smem_api::g_smem_shm_topo_can_reach = nullptr;

int32_t smem_api::load_library(const std::string &lib_dir_path)
{
    SHM_LOG_DEBUG("try to load library: " << g_smem_file_name << ", dir: " << lib_dir_path.c_str());
    std::lock_guard<std::mutex> guard(g_mutex);
    if (g_loaded) {
        return SHMEM_SUCCESS;
    }

    std::string real_path;
    if (!lib_dir_path.empty()) {
        if (!funci::get_library_real_path(lib_dir_path, std::string(g_smem_file_name), real_path)) {
            SHM_LOG_ERROR("get lib path failed, library path: " << lib_dir_path);
            return SHMEM_INNER_ERROR;
        }
    } else {
        real_path = std::string(g_smem_file_name);
    }

    /* dlopen library */
    g_smem_handle = dlopen(real_path.c_str(), RTLD_NOW);
    if (g_smem_handle == nullptr) {
        SHM_LOG_ERROR("Failed to open library: " << real_path << ", error: " << dlerror());
        return -1L;
    }

    /* load sym of smem */
    DL_LOAD_SYM(g_smem_init, smem_init_func, g_smem_handle, "smem_init");
    DL_LOAD_SYM(g_smem_un_init, smem_un_init_func, g_smem_handle, "smem_uninit");
    DL_LOAD_SYM(g_smem_set_extern_logger, smem_set_extern_logger_func, g_smem_handle, "smem_set_extern_logger");
    DL_LOAD_SYM(g_smem_set_log_level, smem_set_log_level_func, g_smem_handle, "smem_set_log_level");
    DL_LOAD_SYM(g_smem_get_last_err_msg, smem_get_last_err_msg_func, g_smem_handle, "smem_get_last_err_msg");
    DL_LOAD_SYM(g_smem_get_and_clear_last_err_msg, smem_get_and_clear_last_err_msg_func, g_smem_handle,
                "smem_get_and_clear_last_err_msg");

    /* load sym of smem_shm */
    DL_LOAD_SYM(g_smem_shm_config_init, smem_shm_config_init_func, g_smem_handle, "smem_shm_config_init");
    DL_LOAD_SYM(g_smem_shm_init, smem_shm_init_func, g_smem_handle, "smem_shm_init");
    DL_LOAD_SYM(g_smem_shm_un_init, smem_shm_un_init_func, g_smem_handle, "smem_shm_uninit");
    DL_LOAD_SYM(g_smem_shm_query_support_data_op, smem_shm_query_support_data_op_func, g_smem_handle,
                "smem_shm_query_support_data_operation");
    DL_LOAD_SYM(g_smem_shm_create, smem_shm_create_func, g_smem_handle, "smem_shm_create");
    DL_LOAD_SYM(g_smem_shm_destroy, smem_shm_destroy_func, g_smem_handle, "smem_shm_destroy");
    DL_LOAD_SYM(g_smem_shm_set_extra_context, smem_shm_set_extra_context_func, g_smem_handle, "smem_shm_set_extra_context");
    DL_LOAD_SYM(g_smem_shm_get_global_team, smem_shm_get_global_team_func, g_smem_handle, "smem_shm_get_global_team");
    DL_LOAD_SYM(g_smem_shm_team_get_rank, smem_shm_team_get_rank_func, g_smem_handle, "smem_shm_team_get_rank");
    DL_LOAD_SYM(g_smem_shm_team_get_size, smem_shm_team_get_size_func, g_smem_handle, "smem_shm_team_get_size");
    DL_LOAD_SYM(g_smem_shm_control_barrier, smem_shm_control_barrier_func, g_smem_handle, "smem_shm_control_barrier");
    DL_LOAD_SYM(g_smem_shm_control_all_gather, smem_shm_control_all_gather_func, g_smem_handle, "smem_shm_control_allgather");
    DL_LOAD_SYM(g_smem_shm_topo_can_reach, smem_shm_topo_can_reach_func, g_smem_handle, "smem_shm_topology_can_reach");

    g_loaded = true;
    SHM_LOG_INFO("loaded library: " << g_smem_file_name << " under dir: " << lib_dir_path.c_str());
    return SHMEM_SUCCESS;
}
}  // namespace shm