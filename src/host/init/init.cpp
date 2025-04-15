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
ShmemCommAttrT shmemCommAttr;

ShmemInitAttr CreateAttributes(int myRank, int nRanks, uint64_t localMemSize){
    ShmemInitAttrT shmemInitAttr;
    shmemInitAttr.myRank = myRank;
    shmemInitAttr.nRanks = nRanks;
    shmemInitAttr.localMemSize = localMemSize;
    return shmemInitAttr;
}

int CommAttrInit(ShmemInitAttr *shmemInitAttr){
    int status = SHMEM_SUCCESS;
    shmemCommAttr = SHMEM_COMM_ATTR;
    int32_t deviceId;
    status = aclrtGetDevice(&deviceId);
    shmemCommAttr.deviceId = deviceId;
    shmemCommAttr.globalSize = (shmemInitAttr->localMemSize + shmemCommAttr.extraSize) * shmemInitAttr->nRanks;
    return status;
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

int SmemHeapInit(ShmemInitAttrT *attributes){
    void *gva;
    int status = SHMEM_SUCCESS;
    status = smem_init(attributes->globalSize + (attributes->extraSize * attributes->nRanks), shmemCommAttr.flag);
    if (status != SHMEM_SUCCESS) {
        ERROR_LOG("smem_init Failed");
        return ERROR_SMEM_ERROR;
    }
    handle = smem_shm_create(attributes->id, attributes->ipPort, attributes->nRanks, attributes->myRank,
                                attributes->deviceId, attributes->localMemSize + attributes->extraSize, attributes->dataOpType, 
                                attributes->timeout, shmemCommAttr.flag, &gva);
    if (handle == nullptr || gva == nullptr) {
        ERROR_LOG("smem_shm_create Failed");
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

int ShmemHostInitAttr(ShmemInitAttrT *attributes){
    int status = SHMEM_SUCCESS;
    status = VersionCompatible();
    status = ShmemOptionsInit();

    ShmemStateInit();
    if (attributes != NULL){
        status = ShmemStateInitAttr(attributes);
        status = CommAttrInit(attributes);
    }
    status = SmemHeapInit(attributes);
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

int ShmemInit(int myRank, int nRanks, uint64_t localMemSize){
    int status = SHMEM_SUCCESS;
    ShmemInitAttrT attributes = CreateAttributes(myRank, nRanks, localMemSize);
    status = ShmemHostInitAttr(&attributes);
    return status;
}

int ShmemFinalize(){
    int status = SHMEM_SUCCESS;
    ShmemTeamFinalize();
    status = smem_shm_destroy(handle, shmemCommAttr.flag);
    smem_uninit();
    return status;
};

int ShmemSetConfig() {
    int status = SHMEM_SUCCESS;
    return status;
}