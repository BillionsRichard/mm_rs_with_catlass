#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "acl/acl.h"
#include "shmemi_host_common.h"

using namespace std;

namespace shm {

#define DEFAULT_MY_PE -1
#define DEFAULT_N_PES -1
#define DEFAULT_FLAG 0
#define DEFAULT_ID 0
#define DEFAULT_TIMEOUT 120

// initializer
#define SHMEM_DEVICE_HOST_STATE_INITALIZER                                            \
    {                                                                                 \
        (1 << 16) + sizeof(ShmemiDeviceHostState),  /* version */                     \
            DEFAULT_MY_PE,                           /* mype */                       \
            DEFAULT_N_PES,                           /* npes */                       \
            NULL,                                    /* heapBase */                   \
            {NULL},                                  /* p2pHeapBase */                \
            {NULL},                                  /* sdmaHeapBase */               \
            {NULL},                                  /* roceHeapBase */               \
            SIZE_MAX,                                /* heapSize */                   \
            {NULL},                                   /* teamPools */                  \
            NULL,                                    /* psyncPool */                  \
            NULL,                                    /* syncCounter */                \
            false,                                   /* shmem_is_shmem_initialized */ \
            false,                                   /* shmem_is_shmem_created */     \
            {0, 16 * 1024, 0},                       /* shmem_mte_config */           \
    }

ShmemiDeviceHostState gState = SHMEM_DEVICE_HOST_STATE_INITALIZER;
shmem_init_attr_t gAttr;
static smem_shm_t gSmemHandle = nullptr;

int32_t VersionCompatible()
{
    int32_t status = SHMEM_SUCCESS;
    return status;
}

int32_t ShmemOptionsInit()
{
    int32_t status = SHMEM_SUCCESS;
    return status;
}

int32_t ShmemStateInitAttr(shmem_init_attr_t *attributes)
{
    int32_t status = SHMEM_SUCCESS;
    gState.mype = attributes->myRank;
    gState.npes = attributes->nRanks;
    gState.heapSize = attributes->localMemSize + SHMEM_EXTRA_SIZE;
    return status;
}

int32_t SmemHeapInit(shmem_init_attr_t *attributes)
{
    void *gva = nullptr;
    int32_t status = SHMEM_SUCCESS;
    uint64_t smemGlobalSize = gState.heapSize * gState.npes;
    int32_t deviceId;
    SHMEM_CHECK_RET(aclrtGetDevice(&deviceId));

    status = SmemApi::SmemInit(DEFAULT_FLAG);
    if (status != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("smem_init Failed");
        return SHMEM_SMEM_ERROR;
    }
    smem_shm_config_t config;
    (void) SmemApi::SmemShmConfigInit(&config);
    status = SmemApi::SmemShmInit(attributes->ipPort, attributes->nRanks, attributes->myRank, deviceId, smemGlobalSize,
             &config);
    if (status != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("smem_init Failed");
        return SHMEM_SMEM_ERROR;
    }

    config.shmInitTimeout = attributes->optionAttr.shmInitTimeout;
    config.shmCreateTimeout = attributes->optionAttr.shmCreateTimeout;
    config.controlOperationTimeout = attributes->optionAttr.controlOperationTimeout;

    gSmemHandle = SmemApi::SmemShmCreate(DEFAULT_ID, attributes->nRanks, attributes->myRank, gState.heapSize,
                  static_cast<smem_shm_data_op_type>(attributes->optionAttr.dataOpEngineType),DEFAULT_FLAG, &gva);

    if (gSmemHandle == nullptr || gva == nullptr) {
        SHM_LOG_ERROR("smem_shm_create Failed");
        return SHMEM_SMEM_ERROR;
    }
    gState.heapBase = (void *) ((uintptr_t) gva + gState.heapSize * attributes->myRank);
    uint32_t reachInfo = 0;
    for (int32_t i = 0; i < gState.npes; i++) {
        status = SmemApi::SmemShmTopoCanReach(gSmemHandle, i, &reachInfo);
        if (reachInfo & SMEMS_DATA_OP_MTE) {
            gState.p2pHeapBase[i] = (void *) ((uintptr_t) gva + gState.heapSize * i);
        } else {
            gState.p2pHeapBase[i] = NULL;
        }
        if (reachInfo & SMEMS_DATA_OP_SDMA) {
            gState.sdmaHeapBase[i] = (void *) ((uintptr_t) gva + gState.heapSize * i);
        } else {
            gState.sdmaHeapBase[i] = NULL;
        }
        if (reachInfo & SMEMS_DATA_OP_ROCE) {
            gState.roceHeapBase[i] = (void *) ((uintptr_t) gva + gState.heapSize * i);
        } else {
            gState.roceHeapBase[i] = NULL;
        }
    }
    gState.isShmemCreated = true;
    return status;
}

int32_t ShmemiControlBarrierAll()
{
    SHM_ASSERT_RETURN(gSmemHandle != nullptr, SHMEM_INVALID_PARAM);
    smem_shm_team_t obj = SmemApi::SmemShmGetGlobalTeam(gSmemHandle);
    SHM_ASSERT_RETURN(obj != nullptr, SHMEM_INVALID_PARAM);
    return SmemApi::SmemShmControlBarrier(obj);
}

int32_t UpdateDeviceState()
{
    if (!gState.isShmemCreated) {
        return SHMEM_NOT_INITED;
    }
    return SmemApi::SmemShmSetExtraContext(gSmemHandle, (void *) &gState, sizeof(ShmemiDeviceHostState));
}

int32_t CheckAttr(shmem_init_attr_t *attributes)
{
    if ((attributes->myRank < 0) || (attributes->nRanks <= 0)) {
        SHM_LOG_ERROR("myRank:" << attributes->myRank << " and nRanks: " << attributes->nRanks <<
            " cannot be less 0 , nRanks still cannot be equal 0");
        return SHMEM_INVALID_VALUE;
    } else if (attributes->myRank >= attributes->nRanks) {
        SHM_LOG_ERROR("nRanks:" << attributes->nRanks << " cannot be less than myRank:" << attributes->myRank);
        return SHMEM_INVALID_PARAM;
    } else if (attributes->localMemSize <= 0) {
        SHM_LOG_ERROR("localMemSize:" << attributes->localMemSize << " cannot be less or equal 0");
        return SHMEM_INVALID_VALUE;
    }
    return SHMEM_SUCCESS;
}

int32_t ShmemiLoadLib()
{
    auto ret = shm::SmemApi::LoadLibrary("");
    if (ret != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("load smem library failed, please set LD_LIBRARY_PATH, ret: " << ret);
        return ret;
    }

    return SHMEM_SUCCESS;
}

} // namespace shm

int32_t shmem_set_data_op_engine_type(shmem_init_attr_t *attributes, data_op_engine_type_t value)
{
    attributes->optionAttr.dataOpEngineType = value;
    return SHMEM_SUCCESS;
}

int32_t shmem_set_timeout(shmem_init_attr_t *attributes, uint32_t value)
{
    attributes->optionAttr.shmInitTimeout = value;
    attributes->optionAttr.shmCreateTimeout = value;
    attributes->optionAttr.controlOperationTimeout = value;
    return SHMEM_SUCCESS;
}

int32_t shmem_set_attr(int32_t myRank, int32_t nRanks, uint64_t localMemSize, const char *ipPort,
                       shmem_init_attr_t **attributes)
{
    *attributes = &shm::gAttr;
    shm::gAttr.version = (1 << 16) + sizeof(shmem_init_attr_t);
    shm::gAttr.myRank = myRank;
    shm::gAttr.nRanks = nRanks;
    shm::gAttr.ipPort = ipPort;
    shm::gAttr.localMemSize = localMemSize;
    shm::gAttr.optionAttr = {SHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    return SHMEM_SUCCESS;
}

int32_t shmem_init_attributes()
{
    if (!shm::gState.isShmemCreated) return SHMEM_STATUS_NOT_INITALIZED;
    else if (!shm::gState.isShmemInitialized) return SHMEM_STATUS_SHM_CREATED;
    else if (shm::gState.isShmemInitialized) return SHMEM_STATUS_IS_INITALIZED;
    else return SHMEM_STATUS_INVALID;
}

int32_t shmem_init_attr(shmem_init_attr_t *attributes)
{
    int32_t ret;
    SHM_ASSERT_RETURN(attributes != nullptr, SHMEM_INVALID_PARAM);
    SHMEM_CHECK_RET(shm::CheckAttr(attributes));
    SHMEM_CHECK_RET(shm::VersionCompatible());
    SHMEM_CHECK_RET(shm::ShmemOptionsInit());

    SHMEM_CHECK_RET(shm::ShmemStateInitAttr(attributes));
    SHMEM_CHECK_RET(shm::ShmemiLoadLib());
    SHMEM_CHECK_RET(shm::SmemHeapInit(attributes));
    SHMEM_CHECK_RET(shm::UpdateDeviceState());

    SHMEM_CHECK_RET(shm::MemoryManagerInitialize(shm::gState.heapBase, shm::gState.heapSize));
    SHMEM_CHECK_RET(shm::ShmemiTeamInit(attributes->myRank, attributes->nRanks));
    SHMEM_CHECK_RET(shm::UpdateDeviceState());
    shm::gState.isShmemInitialized = true;
    SHMEM_CHECK_RET(shm::ShmemiControlBarrierAll());
    return SHMEM_SUCCESS;
}

int32_t shmem_init()
{
    SHMEM_CHECK_RET(shmem_init_attr(&shm::gAttr));
    return SHMEM_SUCCESS;
}

int32_t shmem_finalize()
{
    SHMEM_CHECK_RET(shm::ShmemiTeamFinalize());
    if (shm::gSmemHandle != nullptr) {
        (void)shm::SmemApi::SmemShmDestroy(shm::gSmemHandle, 0);
        shm::gSmemHandle = nullptr;
    }
    shm::SmemApi::SmemUnInit();
    return SHMEM_SUCCESS;
}