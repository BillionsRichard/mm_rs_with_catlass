/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */
#include <dlfcn.h>
#include "smem_shm_api.h"
#include "shmemi_host_common.h"

namespace shm {
bool SmemApi::gLoaded = false;
std::mutex SmemApi::gMutex;

void *SmemApi::gSmemHandle = nullptr;
const char *SmemApi::gSmemFileName = "libmf_smem.so";

/* smem api define */
SmemInitFunc SmemApi::gSmemInit = nullptr;
SmemSetExternLoggerFunc SmemApi::gSmemSetExternLogger = nullptr;
SmemSetLogLevelFunc SmemApi::gSmemSetLogLevel = nullptr;
SmemUnInitFunc SmemApi::gSmemUnInit = nullptr;
SmemGetLastErrMsgFunc SmemApi::gSmemGetLastErrMsg = nullptr;
SmemGetAndClearLastErrMsgFunc SmemApi::gSmemGetAndClearLastErrMsg = nullptr;

/* smem shm api define */
SmemShmConfigInitFunc SmemApi::gSmemShmConfigInit = nullptr;
SmemShmInitFunc SmemApi::gSmemShmInit = nullptr;
SmemShmUnInitFunc SmemApi::gSmemShmUnInit = nullptr;
SmemShmQuerySupportDataOpFunc SmemApi::gSmemShmQuerySupportDataOp = nullptr;
SmemShmCreateFunc SmemApi::gSmemShmCreate = nullptr;
SmemShmDestroyFunc SmemApi::gSmemShmDestroy = nullptr;
SmemShmSetExtraContextFunc SmemApi::gSmemShmSetExtraContext = nullptr;
SmemShmGetGlobalTeamFunc SmemApi::gSmemShmGetGlobalTeam = nullptr;
SmemShmTeamGetRankFunc SmemApi::gSmemShmTeamGetRank = nullptr;
SmemShmTeamGetSizeFunc SmemApi::gSmemShmTeamGetSize = nullptr;
SmemShmControlBarrierFunc SmemApi::gSmemShmControlBarrier = nullptr;
SmemShmControlAllGatherFunc SmemApi::gSmemShmControlAllGather = nullptr;
SmemShmTopoCanReachFunc SmemApi::gSmemShmTopoCanReach = nullptr;

int32_t SmemApi::LoadLibrary(const std::string &libDirPath)
{
    SHM_LOG_DEBUG("try to load library: " << gSmemFileName << ", dir: " << libDirPath.c_str());
    std::lock_guard<std::mutex> guard(gMutex);
    if (gLoaded) {
        return SHMEM_SUCCESS;
    }

    std::string realPath;
    if (!libDirPath.empty()) {
        if (!Func::LibraryRealPath(libDirPath, std::string(gSmemFileName), realPath)) {
            SHM_LOG_ERROR("get lib path failed, library path: " << libDirPath);
            return SHMEM_INNER_ERROR;
        }
    } else {
        realPath = std::string(gSmemFileName);
    }

    /* dlopen library */
    gSmemHandle = dlopen(realPath.c_str(), RTLD_NOW);
    if (gSmemHandle == nullptr) {
        SHM_LOG_ERROR("Failed to open library: " << realPath << ", error: " << dlerror());
        return -1L;
    }

    /* load sym of smem */
    DL_LOAD_SYM(gSmemInit, SmemInitFunc, gSmemHandle, "smem_init");
    DL_LOAD_SYM(gSmemUnInit, SmemUnInitFunc, gSmemHandle, "smem_uninit");
    DL_LOAD_SYM(gSmemSetExternLogger, SmemSetExternLoggerFunc, gSmemHandle, "smem_set_extern_logger");
    DL_LOAD_SYM(gSmemSetLogLevel, SmemSetLogLevelFunc, gSmemHandle, "smem_set_log_level");
    DL_LOAD_SYM(gSmemGetLastErrMsg, SmemGetLastErrMsgFunc, gSmemHandle, "smem_get_last_err_msg");
    DL_LOAD_SYM(gSmemGetAndClearLastErrMsg, SmemGetAndClearLastErrMsgFunc, gSmemHandle,
                "smem_get_and_clear_last_err_msg");

    /* load sym of smem_shm */
    DL_LOAD_SYM(gSmemShmConfigInit, SmemShmConfigInitFunc, gSmemHandle, "smem_shm_config_init");
    DL_LOAD_SYM(gSmemShmInit, SmemShmInitFunc, gSmemHandle, "smem_shm_init");
    DL_LOAD_SYM(gSmemShmUnInit, SmemShmUnInitFunc, gSmemHandle, "smem_shm_uninit");
    DL_LOAD_SYM(gSmemShmQuerySupportDataOp, SmemShmQuerySupportDataOpFunc, gSmemHandle,
                "smem_shm_query_support_data_operation");
    DL_LOAD_SYM(gSmemShmCreate, SmemShmCreateFunc, gSmemHandle, "smem_shm_create");
    DL_LOAD_SYM(gSmemShmDestroy, SmemShmDestroyFunc, gSmemHandle, "smem_shm_destroy");
    DL_LOAD_SYM(gSmemShmSetExtraContext, SmemShmSetExtraContextFunc, gSmemHandle, "smem_shm_set_extra_context");
    DL_LOAD_SYM(gSmemShmGetGlobalTeam, SmemShmGetGlobalTeamFunc, gSmemHandle, "smem_shm_get_global_team");
    DL_LOAD_SYM(gSmemShmTeamGetRank, SmemShmTeamGetRankFunc, gSmemHandle, "smem_shm_team_get_rank");
    DL_LOAD_SYM(gSmemShmTeamGetSize, SmemShmTeamGetSizeFunc, gSmemHandle, "smem_shm_team_get_size");
    DL_LOAD_SYM(gSmemShmControlBarrier, SmemShmControlBarrierFunc, gSmemHandle, "smem_shm_control_barrier");
    DL_LOAD_SYM(gSmemShmControlAllGather, SmemShmControlAllGatherFunc, gSmemHandle, "smem_shm_control_allgather");
    DL_LOAD_SYM(gSmemShmTopoCanReach, SmemShmTopoCanReachFunc, gSmemHandle, "smem_shm_topology_can_reach");

    gLoaded = true;
    SHM_LOG_INFO("loaded library: " << gSmemFileName << " under dir: " << libDirPath.c_str());
    return SHMEM_SUCCESS;
}
}  // namespace shm