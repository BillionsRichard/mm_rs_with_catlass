#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace std;

#include "acl/acl.h"
#include "shmemi_host_intf.h"

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
            {NULL},                                  /* teamPools */                  \
            NULL,                                    /* psyncPool */                  \
            NULL,                                    /* syncCounter */                \
            false,                                   /* shmem_is_shmem_initialized */ \
            false,                                   /* shmem_is_shmem_created */     \
            {0, 16 * 1024, 0},                       /* shmem_mte_config */           \
    }

ShmemiDeviceHostState gState;
ShmemInitAttrT gAttr;

static smem_shm_t handle = nullptr;

int SetDataOpEngineType(ShmemInitAttrT *attributes, DataOpEngineType value) {
    attributes->optionAttr.dataOpEngineType = value;
    return SHMEM_SUCCESS;
}
int SetTimeout(ShmemInitAttrT *attributes, uint32_t value) {
    attributes->optionAttr.shmInitTimeout = value;
    attributes->optionAttr.shmCreateTimeout = value;
    attributes->optionAttr.controlOperationTimeout = value;
    return SHMEM_SUCCESS;
}


int ShmemSetAttr(int myRank, int nRanks, uint64_t localMemSize, const char* ipPort, ShmemInitAttrT **attributes) {
    *attributes = &gAttr;
    gAttr.version = (1 << 16) + sizeof(ShmemInitAttrT);
    gAttr.myRank = myRank;
    gAttr.nRanks = nRanks;
    gAttr.ipPort = ipPort;
    gAttr.localMemSize = localMemSize;
    DataOpEngineType supportDataOp = static_cast<DataOpEngineType>(smem_shm_query_support_data_operation());
    gAttr.optionAttr = {supportDataOp, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    return SHMEM_SUCCESS;
}

int VersionCompatible(){
    int status = SHMEM_SUCCESS;
    return status;
}

int ShmemOptionsInit(){
    int status = SHMEM_SUCCESS;
    return status;
}

void ShmemStateInit() { gState = SHMEM_DEVICE_HOST_STATE_INITALIZER; }

int ShmemInitStatus(){
    if (!gState.shemeIsShmemCreated) return SHMEM_STATUS_NOT_INITALIZED;
    else if (!gState.shemeIsShmemInitialized) return SHMEM_STATUS_SHM_CREATED;
    else if (gState.shemeIsShmemInitialized) return SHMEM_STATUS_IS_INITALIZED;
    else return SHMEM_STATUS_INVALID;
}

int ShmemStateInitAttr(ShmemInitAttrT *attributes){
    int status = SHMEM_SUCCESS;
    gState.mype = attributes->myRank;
    gState.npes = attributes->nRanks;
    gState.heapSize = attributes->localMemSize + SHMEM_EXTRA_SIZE;
    return status;
}

int SmemHeapInit(ShmemInitAttrT *attributes){
    void *gva = nullptr;
    int status = SHMEM_SUCCESS;
    uint64_t smemGlobalSize = gState.heapSize * gState.npes;
    int32_t deviceId;
    CHECK_ACL(aclrtGetDevice(&deviceId));

    status = smem_init(DEFAULT_FLAG);
    if (status != SHMEM_SUCCESS) {
        ERROR_LOG("smem_init Failed");
        return ERROR_SMEM_ERROR;
    }
    smem_shm_config_t config;
    (void) smem_shm_config_init(&config);
    status = smem_shm_init(attributes->ipPort, attributes->nRanks, attributes->myRank, deviceId, smemGlobalSize, &config);
    if (status != SHMEM_SUCCESS) {
        ERROR_LOG("smem_init Failed");
        return ERROR_SMEM_ERROR;
    }

    config.shmInitTimeout=attributes->optionAttr.shmInitTimeout;
    config.shmCreateTimeout=attributes->optionAttr.shmCreateTimeout;
    config.controlOperationTimeout=attributes->optionAttr.controlOperationTimeout;

    handle = smem_shm_create(DEFAULT_ID, attributes->nRanks, attributes->myRank, gState.heapSize, 
                             static_cast<smem_shm_data_op_type>(attributes->optionAttr.dataOpEngineType), 
                             DEFAULT_FLAG, &gva);

    if (handle == nullptr || gva == nullptr) {
        ERROR_LOG("smem_shm_create Failed");
        return ERROR_SMEM_ERROR;
    }
    gState.heapBase = (void *)((uintptr_t)gva + (attributes->localMemSize + SHMEM_EXTRA_SIZE) * attributes->myRank);
    uint32_t reachInfo = 0;
    for ( int i = 0;  i < gState.npes; i++){
        status = smem_shm_topology_can_reach(handle, i, &reachInfo);
        if (reachInfo & SMEMS_DATA_OP_MTE) {
            gState.p2pHeapBase[i] = (void *)((uintptr_t)gva + (attributes->localMemSize + SHMEM_EXTRA_SIZE) * i);
        } else {
            gState.p2pHeapBase[i] = NULL;
        }
        if (reachInfo & SMEMS_DATA_OP_SDMA) {
            gState.sdmaHeapBase[i] = (void *)((uintptr_t)gva + (attributes->localMemSize + SHMEM_EXTRA_SIZE) * i);
        } else {
            gState.sdmaHeapBase[i] = NULL;
        }
        if (reachInfo & SMEMS_DATA_OP_ROCE) {
            gState.roceHeapBase[i] = (void *)((uintptr_t)gva + (attributes->localMemSize + SHMEM_EXTRA_SIZE) * i);
        } else {
            gState.roceHeapBase[i] = NULL;
        }
    }
    gState.shemeIsShmemCreated = true;
    return status;
}

int UpdateDeviceState(){
    int status = SHMEM_SUCCESS;
    status = smem_shm_set_extra_context(handle, (void *)&gState, sizeof(ShmemiDeviceHostState));
    return status;
}

int CheckAttr(ShmemInitAttrT *attributes) {
    if ((attributes->myRank < 0) || (attributes->nRanks <= 0)) {
        ERROR_LOG("myRank:%d and nRanks%d cannot be less 0 , nRanks still cannot be equal 0",
                attributes->myRank, attributes->nRanks);
        return ERROR_INVALID_VALUE;
    } else if (attributes->myRank >= attributes->nRanks) {
        ERROR_LOG("nRanks:%d cannot be less than myRank%d", attributes->nRanks, attributes->myRank);
        return ERROR_INVALID_PARAM;
    } else if (attributes->localMemSize <= 0) {
        ERROR_LOG("localMemSize:%llu cannot be less or equal 0", attributes->localMemSize);
        return ERROR_INVALID_VALUE;
    }
    return SHMEM_SUCCESS;
}

int ShmemInitAttr(ShmemInitAttrT *attributes){
    int status = SHMEM_SUCCESS;
    CHECK_SHMEM(CheckAttr(attributes), status);
    CHECK_SHMEM(VersionCompatible(), status);
    CHECK_SHMEM(ShmemOptionsInit(), status);

    ShmemStateInit();
    if (attributes != NULL){
        CHECK_SHMEM(ShmemStateInitAttr(attributes), status);
    }
    CHECK_SHMEM_STATUS(SmemHeapInit(attributes), status, "Failed to initialize the share memory heap");

    CHECK_SHMEM(UpdateDeviceState(), status);
    CHECK_SHMEM_STATUS(ShmemiTeamInit(attributes->myRank, attributes->nRanks), status, "Failed to initialize the team");
    CHECK_SHMEM(UpdateDeviceState(), status);
    gState.shemeIsShmemInitialized = true;
    return status;
}

int ShmemInit(){
    int status = SHMEM_SUCCESS;
    CHECK_SHMEM(ShmemInitAttr(&gAttr), status);
    return status;
}

int ShmemFinalize() {
    int status = SHMEM_SUCCESS;
    CHECK_SHMEM(ShmemiTeamFinalize(), status);
    ShmemStateInit();
    CHECK_SHMEM(smem_shm_destroy(handle, DEFAULT_FLAG), status);
    smem_uninit();
    return status;
};

int ShmemSetConfig() {
    int status = SHMEM_SUCCESS;
    return status;
}