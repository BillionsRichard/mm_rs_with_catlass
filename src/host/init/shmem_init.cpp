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
        (1 << 16) + sizeof(shmemi_device_host_state_t),  /* version */                     \
            DEFAULT_MY_PE,                           /* mype */                       \
            DEFAULT_N_PES,                           /* npes */                       \
            NULL,                                    /* heap_base */                   \
            {NULL},                                  /* p2p_heap_base */                \
            {NULL},                                  /* sdmaHeapBase */               \
            {NULL},                                  /* roceHeapBase */               \
            SIZE_MAX,                                /* heap_size */                   \
            {NULL},                                   /* team_pools */                  \
            NULL,                                    /* psyncPool */                  \
            NULL,                                    /* sync_counter */                \
            false,                                   /* shmem_is_shmem_initialized */ \
            false,                                   /* shmem_is_shmem_created */     \
            {0, 16 * 1024, 0},                       /* shmem_mte_config */           \
    }

shmemi_device_host_state_t gState = SHMEM_DEVICE_HOST_STATE_INITALIZER;
shmem_init_attr_t gAttr;
static smem_shm_t gSmemHandle = nullptr;
static bool gAttrInit = false;
static char* gIpPort = nullptr;

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
    gState.mype = attributes->my_rank;
    gState.npes = attributes->n_ranks;
    gState.heap_size = attributes->local_mem_size + SHMEM_EXTRA_SIZE;
    return status;
}

int32_t SmemHeapInit(shmem_init_attr_t *attributes)
{
    void *gva = nullptr;
    int32_t status = SHMEM_SUCCESS;
    uint64_t smemGlobalSize = gState.heap_size * gState.npes;
    int32_t deviceId;
    SHMEM_CHECK_RET(aclrtGetDevice(&deviceId));

    status = SmemApi::SmemInit(DEFAULT_FLAG);
    if (status != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("smem_init Failed");
        return SHMEM_SMEM_ERROR;
    }
    smem_shm_config_t config;
    (void) SmemApi::SmemShmConfigInit(&config);
    status = SmemApi::SmemShmInit(attributes->ip_port, attributes->n_ranks, attributes->my_rank, deviceId, smemGlobalSize,
             &config);
    if (status != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("smem_init Failed");
        return SHMEM_SMEM_ERROR;
    }

    config.shmInitTimeout = attributes->option_attr.shmInit_timeout;
    config.shmCreateTimeout = attributes->option_attr.shm_create_timeout;
    config.controlOperationTimeout = attributes->option_attr.control_operation_timeout;

    gSmemHandle = SmemApi::SmemShmCreate(DEFAULT_ID, attributes->n_ranks, attributes->my_rank, gState.heap_size,
                  static_cast<smem_shm_data_op_type>(attributes->option_attr.data_op_engine_type),DEFAULT_FLAG, &gva);

    if (gSmemHandle == nullptr || gva == nullptr) {
        SHM_LOG_ERROR("smem_shm_create Failed");
        return SHMEM_SMEM_ERROR;
    }
    gState.heap_base = (void *) ((uintptr_t) gva + gState.heap_size * attributes->my_rank);
    uint32_t reachInfo = 0;
    for (int32_t i = 0; i < gState.npes; i++) {
        status = SmemApi::SmemShmTopoCanReach(gSmemHandle, i, &reachInfo);
        if (reachInfo & SMEMS_DATA_OP_MTE) {
            gState.p2p_heap_base[i] = (void *) ((uintptr_t) gva + gState.heap_size * i);
        } else {
            gState.p2p_heap_base[i] = NULL;
        }
        if (reachInfo & SMEMS_DATA_OP_SDMA) {
            gState.sdmaHeapBase[i] = (void *) ((uintptr_t) gva + gState.heap_size * i);
        } else {
            gState.sdmaHeapBase[i] = NULL;
        }
        if (reachInfo & SMEMS_DATA_OP_ROCE) {
            gState.roceHeapBase[i] = (void *) ((uintptr_t) gva + gState.heap_size * i);
        } else {
            gState.roceHeapBase[i] = NULL;
        }
    }
    if (shm::gIpPort != nullptr) {
        delete[] shm::gIpPort;
        attributes->ip_port = nullptr;
    } else {
         SHM_LOG_WARN("my_rank:" << attributes->my_rank << " shm::gIpPort is released in advance!");
         attributes->ip_port = nullptr;
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
    return SmemApi::SmemShmSetExtraContext(gSmemHandle, (void *) &gState, sizeof(shmemi_device_host_state_t));
}

int32_t CheckAttr(shmem_init_attr_t *attributes)
{
    if ((attributes->my_rank < 0) || (attributes->n_ranks <= 0)) {
        SHM_LOG_ERROR("my_rank:" << attributes->my_rank << " and n_ranks: " << attributes->n_ranks <<
            " cannot be less 0 , n_ranks still cannot be equal 0");
        return SHMEM_INVALID_VALUE;
    } else if (attributes->my_rank >= attributes->n_ranks) {
        SHM_LOG_ERROR("n_ranks:" << attributes->n_ranks << " cannot be less than my_rank:" << attributes->my_rank);
        return SHMEM_INVALID_PARAM;
    } else if (attributes->local_mem_size <= 0) {
        SHM_LOG_ERROR("local_mem_size:" << attributes->local_mem_size << " cannot be less or equal 0");
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
    attributes->option_attr.data_op_engine_type = value;
    return SHMEM_SUCCESS;
}

int32_t shmem_set_timeout(shmem_init_attr_t *attributes, uint32_t value)
{
    attributes->option_attr.shmInit_timeout = value;
    attributes->option_attr.shm_create_timeout = value;
    attributes->option_attr.control_operation_timeout = value;
    return SHMEM_SUCCESS;
}

int32_t shmem_set_attr(int32_t my_rank, int32_t n_ranks, uint64_t local_mem_size, const char *ip_port,
                       shmem_init_attr_t **attributes)
{
    *attributes = &shm::gAttr;
    size_t ipLen = strlen(ip_port);
    shm::gIpPort = new char[ipLen + 1];
    strcpy(shm::gIpPort, ip_port);
    if (shm::gIpPort == nullptr) {
        SHM_LOG_ERROR("my_rank:" << my_rank << " shm::gIpPort is nullptr!");
        return SHMEM_INVALID_VALUE;
    }
    int attrVersion = (1 << 16) + sizeof(shmem_init_attr_t);
    shm::gAttr.my_rank = my_rank;
    shm::gAttr.n_ranks = n_ranks;
    shm::gAttr.ip_port = shm::gIpPort;
    shm::gAttr.local_mem_size = local_mem_size;
    shm::gAttr.option_attr = {attrVersion, SHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    shm::gAttrInit = true;
    return SHMEM_SUCCESS;
}

int32_t shmem_init_status()
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

    SHMEM_CHECK_RET(shm::MemoryManagerInitialize(shm::gState.heap_base, shm::gState.heap_size));
    SHMEM_CHECK_RET(shm::ShmemiTeamInit(shm::gState.mype, shm::gState.npes));
    SHMEM_CHECK_RET(shm::UpdateDeviceState());
    shm::gState.isShmemInitialized = true;
    SHMEM_CHECK_RET(shm::ShmemiControlBarrierAll());
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