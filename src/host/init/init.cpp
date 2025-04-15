#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace std;

#include "constants.h"
#include "init.h"
#include "shmem_api.h"
#include "data_utils.h"

smem_shm_t handle = nullptr;
ShmemDeviceHostStateT shmemDeviceHostState;

ShmemInitAttr CreateAttributes(int id, const char* ipPort, int myRank, int nRanks, int deviceId,
                                uint64_t localMemSize, uint64_t extraSize, smem_shm_data_op_type dataOpType,
                                int timeout){
    ShmemInitAttrT shmemInitAttr;
    shmemInitAttr.version = (1 << 16) + sizeof(ShmemInitAttrT);
    shmemInitAttr.id = id;
    shmemInitAttr.ipPort = ipPort;
    shmemInitAttr.myRank = myRank;
    shmemInitAttr.nRanks = nRanks;
    shmemInitAttr.deviceId = deviceId;
    shmemInitAttr.localMemSize = localMemSize;
    shmemInitAttr.dataOpType = dataOpType;
    shmemInitAttr.timeout = timeout;
    shmemInitAttr.globalSize = localMemSize * nRanks;
    shmemInitAttr.extraSize = extraSize;
    return shmemInitAttr;
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
    shmemDeviceHostState.heapSize = attributes->localMemSize;
    return status;
}

int SmemHeapInit(uint32_t flag, ShmemInitAttrT *attributes){
    void *gva;
    int status = SHMEM_SUCCESS;
    status = smem_init(attributes->globalSize + (attributes->extraSize * attributes->nRanks), flag);
    handle = smem_shm_create(attributes->id, attributes->ipPort, attributes->nRanks, attributes->myRank,
                                attributes->deviceId, attributes->localMemSize + attributes->extraSize, attributes->dataOpType, 
                                attributes->timeout, flag, &gva);
    if (handle == nullptr || gva == nullptr) {
        return ERROR_SMEM_ERROR;
    }
    shmemDeviceHostState.heapBase = gva;
    for ( int i = 0;  i < shmemDeviceHostState.npes; i++){
        shmemDeviceHostState.p2pHeapBase[i] = (void *)((uintptr_t)gva + attributes->localMemSize * i);
    }
    shmemDeviceHostState.shemeIsShmemCreated = true;
    return status;
}

int UpdateDeviceState(){
    int status = SHMEM_SUCCESS;
    status = smem_shm_set_extra_context(handle, (void *)&shmemDeviceHostState, sizeof(ShmemDeviceHostStateT));
    return status;
}

int ShmemHostInitAttr(uint32_t flag, ShmemInitAttrT *attributes){
    int status = SHMEM_SUCCESS;
    status = VersionCompatible();
    status = ShmemOptionsInit();

    ShmemStateInit();
    if (attributes != NULL){
        status = ShmemStateInitAttr(attributes);
    }
    status = SmemHeapInit(flag, attributes);
    if (!shmemDeviceHostState.shemeIsShmemCreated){
        ERROR_LOG("Failed to initialize the share memory heap");
        return status;
    }
    status = UpdateDeviceState();
    status = ShmemTeamInit(attributes->myRank, attributes->nRanks);
    if (status != SHMEM_SUCCESS) {
        ERROR_LOG("Failed to initialize the team");
        return status;
    }
    status = UpdateDeviceState();
    shmemDeviceHostState.shemeIsShmemInitialized = true;
    return status;
}

int ShmemInit(int rank, int size) {
    int status = SHMEM_SUCCESS;
    return status;
}

int ShmemInit(uint32_t flag, ShmemInitAttrT *attributes){
    int status = SHMEM_SUCCESS;
    if (attributes == NULL) {
        ERROR_LOG("Empty attr is not currently supported");
        return ERROR_INVALID_PARAM;
    }
    status = ShmemHostInitAttr(flag, attributes);
    return status;
}

int ShmemFinalize(uint32_t flag){
    int status = SHMEM_SUCCESS;
    ShmemTeamFinalize();
    status = smem_shm_destroy(handle, flag);
    smem_uninit();
    return status;
};

int ShmemSetConfig() {
    int status = SHMEM_SUCCESS;
    return status;
}