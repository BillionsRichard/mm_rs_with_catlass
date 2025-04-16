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
    CHECK_ACL(aclrtGetDevice(&deviceId));
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
    void *gva = nullptr;
    int status = SHMEM_SUCCESS;
    status = smem_init(shmemCommAttr.globalSize, shmemCommAttr.flag);
    if (status != SHMEM_SUCCESS) {
        ERROR_LOG("smem_init Failed");
        return ERROR_SMEM_ERROR;
    }
    handle = smem_shm_create(shmemCommAttr.id, shmemCommAttr.ipPort, attributes->nRanks, attributes->myRank,
                                shmemCommAttr.deviceId, attributes->localMemSize + shmemCommAttr.extraSize, shmemCommAttr.dataOpType, 
                                shmemCommAttr.timeout, shmemCommAttr.flag, &gva);
    if (handle == nullptr || gva == nullptr) {
        ERROR_LOG("smem_shm_create Failed");
        return ERROR_SMEM_ERROR;
    }
    shmemDeviceHostState.heapBase = gva;
    uint32_t reachInfo = 0;
    for ( int i = 0;  i < shmemDeviceHostState.npes; i++){
        status = smem_shm_topology_can_reach(handle, i, shmemCommAttr.dataOpType, &reachInfo);
        if  (reachInfo <= SMEM_TRANSPORT_CAP_MAP) {
            shmemDeviceHostState.p2pHeapBase[i] = (void *)((uintptr_t)gva + attributes->localMemSize * i);
        } else {
            shmemDeviceHostState.p2pHeapBase[i] = NULL;
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

int ShmemHostInitAttr(ShmemInitAttrT *attributes){
    int status = SHMEM_SUCCESS;
    CHECK_SHMEM(VersionCompatible(), status);
    CHECK_SHMEM(ShmemOptionsInit(), status);

    ShmemStateInit();
    if (attributes != NULL){
        CHECK_SHMEM(ShmemStateInitAttr(attributes), status);
        CHECK_SHMEM(CommAttrInit(attributes), status);
    }
    CHECK_SHMEM_STATUS(SmemHeapInit(attributes), status, "Failed to initialize the share memory heap");

    CHECK_SHMEM(UpdateDeviceState(), status);
    CHECK_SHMEM_STATUS(ShmemTeamInit(attributes->myRank, attributes->nRanks), status, "Failed to initialize the team");
    CHECK_SHMEM(UpdateDeviceState(), status);
    shmemDeviceHostState.shemeIsShmemInitialized = true;
    return status;
}

int ShmemInit(int myRank, int nRanks, uint64_t localMemSize){
    int status = SHMEM_SUCCESS;
    CHECK_SHMEM(CheckAttr(myRank, nRanks, localMemSize), status);
    ShmemInitAttrT attributes = CreateAttributes(myRank, nRanks, localMemSize);
    CHECK_SHMEM(ShmemHostInitAttr(&attributes), status);
    return status;
}

int ShmemFinalize(){
    int status = SHMEM_SUCCESS;
    CHECK_SHMEM(ShmemTeamFinalize(), status);
    ShmemStateInit();
    CHECK_SHMEM(smem_shm_destroy(handle, shmemCommAttr.flag), status);
    smem_uninit();
    return status;
};

int ShmemSetConfig() {
    int status = SHMEM_SUCCESS;
    return status;
}

int CheckAttr(int myRank, int nRanks, uint64_t localMemSize) {
    if (myRank < 0 || nRanks <= 0) {
        ERROR_LOG("myRank:%d and nRanks%d cannot be less 0 , nRanks still cannot be equal 0",
                myRank, nRanks);
        return ERROR_INVALID_VALUE;
    } else if (myRank >= nRanks) {
        ERROR_LOG("nRanks:%d cannot be less than myRank%d", nRanks, myRank);
        return ERROR_INVALID_PARAM;
    } else if (localMemSize <= 0) {
        ERROR_LOG("localMemSize:%llu cannot be less or equal 0", localMemSize);
        return ERROR_INVALID_VALUE;
    }
    return SHMEM_SUCCESS;
}