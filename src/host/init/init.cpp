#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace std;

#include "constants.h"
#include "team.h"
#include "init_internal.h"
#include "shmem_api.h"
#include "data_utils.h"
#include "types_internal.h"

#include "smem.h"
#include "smem_shm.h"

smem_shm_t handle = nullptr;
ShmemDeviceHostStateT shmemDeviceHostState;
ShmemInitAttrT shmemInitAttr;

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
    *attributes = &shmemInitAttr;
    shmemInitAttr.version = (1 << 16) + sizeof(ShmemInitAttrT);
    shmemInitAttr.myRank = myRank;
    shmemInitAttr.nRanks = nRanks;
    shmemInitAttr.ipPort = ipPort;
    shmemInitAttr.localMemSize = localMemSize;
    DataOpEngineType supportDataOp = static_cast<DataOpEngineType>(smem_shm_query_support_data_operation());
    shmemInitAttr.optionAttr = {supportDataOp, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
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

void ShmemStateInit() { shmemDeviceHostState = SHMEM_DEVICE_HOST_STATE_INITALIZER; }

int ShmemInitStatus(){
    if (!shmemDeviceHostState.shemeIsShmemCreated) return SHMEM_STATUS_NOT_INITALIZED;
    else if (!shmemDeviceHostState.shemeIsShmemInitialized) return SHMEM_STATUS_SHM_CREATED;
    else if (shmemDeviceHostState.shemeIsShmemInitialized) return SHMEM_STATUS_IS_INITALIZED;
    else return SHMEM_STATUS_INVALID;
}

int ShmemStateInitAttr(ShmemInitAttrT *attributes){
    int status = SHMEM_SUCCESS;
    shmemDeviceHostState.mype = attributes->myRank;
    shmemDeviceHostState.npes = attributes->nRanks;
    shmemDeviceHostState.heapSize = attributes->localMemSize + DEFAULT_EXTRA_SIZE;
    return status;
}

int SmemHeapInit(ShmemInitAttrT *attributes){
    void *gva = nullptr;
    int status = SHMEM_SUCCESS;
    uint64_t smemGlobalSize = shmemDeviceHostState.heapSize * shmemDeviceHostState.npes;
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

    handle = smem_shm_create(DEFAULT_ID, attributes->nRanks, attributes->myRank, shmemDeviceHostState.heapSize, 
                             static_cast<smem_shm_data_op_type>(attributes->optionAttr.dataOpEngineType), 
                             DEFAULT_FLAG, &gva);

    if (handle == nullptr || gva == nullptr) {
        ERROR_LOG("smem_shm_create Failed");
        return ERROR_SMEM_ERROR;
    }
    shmemDeviceHostState.heapBase = (void *)((uintptr_t)gva + (attributes->localMemSize + DEFAULT_EXTRA_SIZE) * attributes->myRank);
    uint32_t reachInfo = 0;
    for ( int i = 0;  i < shmemDeviceHostState.npes; i++){
        status = smem_shm_topology_can_reach(handle, i, &reachInfo);
        if (reachInfo & SMEMS_DATA_OP_MTE) {
            shmemDeviceHostState.p2pHeapBase[i] = (void *)((uintptr_t)gva + (attributes->localMemSize + DEFAULT_EXTRA_SIZE) * i);
        } else {
            shmemDeviceHostState.p2pHeapBase[i] = NULL;
        }
        if (reachInfo & SMEMS_DATA_OP_SDMA) {
            shmemDeviceHostState.sdmaHeapBase[i] = (void *)((uintptr_t)gva + (attributes->localMemSize + DEFAULT_EXTRA_SIZE) * i);
        } else {
            shmemDeviceHostState.sdmaHeapBase[i] = NULL;
        }
        if (reachInfo & SMEMS_DATA_OP_ROCE) {
            shmemDeviceHostState.roceHeapBase[i] = (void *)((uintptr_t)gva + (attributes->localMemSize + DEFAULT_EXTRA_SIZE) * i);
        } else {
            shmemDeviceHostState.roceHeapBase[i] = NULL;
        }
    }
    shmemDeviceHostState.shemeIsShmemCreated = true;
    return status;
}

int UpdateDeviceState(){
    int status = SHMEM_SUCCESS;
    status = smem_shm_set_extra_context(handle, (void *)&shmemDeviceHostState, sizeof(ShmemDeviceHostStateT));
    return status;
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
    CHECK_SHMEM_STATUS(ShmemTeamInit(attributes->myRank, attributes->nRanks), status, "Failed to initialize the team");
    CHECK_SHMEM(UpdateDeviceState(), status);
    shmemDeviceHostState.shemeIsShmemInitialized = true;
    return status;
}

int ShmemInit(){
    int status = SHMEM_SUCCESS;
    CHECK_SHMEM(ShmemInitAttr(&shmemInitAttr), status);
    return status;
}

int ShmemFinalize() {
    int status = SHMEM_SUCCESS;
    CHECK_SHMEM(ShmemTeamFinalize(), status);
    ShmemStateInit();
    CHECK_SHMEM(smem_shm_destroy(handle, DEFAULT_FLAG), status);
    smem_uninit();
    return status;
};

int ShmemSetConfig() {
    int status = SHMEM_SUCCESS;
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