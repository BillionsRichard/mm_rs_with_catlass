#include "kernel_operator.h"

#include "shmem_device_api.h"

SHMEM_DEVICE void CubeGuard() {
    using namespace AscendC;

#ifdef __DAV_C220_CUBE__
    LocalTensor<float> result;
    result.address_.logicPos = (uint8_t)TPosition::CO1;
    result.InitBuffer(0, 256);
    
    LocalTensor<half> left;
    left.address_.logicPos = (uint8_t)TPosition::A2;
    left.InitBuffer(0, 256);

    LocalTensor<half> right;
    right.address_.logicPos = (uint8_t)TPosition::B2;
    right.InitBuffer(0, 256);

    MmadParams param;
    param.m = 16;
    param.n = 16;
    param.k = 16;

    Mmad<float, half, half>(result, left, right, param);
#endif
}

SHMEM_DEVICE void VecGuard() {
    using namespace AscendC;

#ifdef __DAV_C220_VEC__
    LocalTensor<half> local;
    local.address_.logicPos = (uint8_t)TPosition::VECIN;
    local.InitBuffer(0, 32);
    
    GlobalTensor<half> global;
    ShmemiTeam *team = ShmemiGetState()->teamPools[0];
    auto addr = (__gm__ half *) ((uint64_t) ShmemiGetTeamSyncCounter(team) + SHMEMI_SYNCBIT_SIZE);
    global.SetGlobalBuffer(addr, 32);

    DataCopy<half>(local, global, 32);

    local.address_.logicPos = (uint8_t)TPosition::VECOUT;
    DataCopy<half>(global, local, 32);
#endif
}

SHMEM_DEVICE void CVGuard() {
    // insert Mmad and DataCopy calls to make sure barrier works.
    CubeGuard();
    VecGuard();
}

extern "C" SHMEM_GLOBAL void fetchAddr(GM_ADDR syncArray, GM_ADDR syncCounter) {
    ShmemiTeam *team = ShmemiGetState()->teamPools[0];
    *((__gm__ uint64_t*) syncArray) = (uint64_t) ShmemiGetTeamSyncArray(team);
    *((__gm__ uint64_t*) syncCounter) = (uint64_t) ShmemiGetTeamSyncCounter(team);
}

extern "C" SHMEM_GLOBAL void barrier(GM_ADDR stub) {
    CubeGuard();
    ShmemBarrierAll();
    VecGuard();
}

extern "C" SHMEM_GLOBAL void increase(GM_ADDR addr, int rankId, int rankSize) {
    CVGuard();
    
#ifdef __DAV_C220_CUBE__
    // scalar unit of cube core is not affected by barrier
    ShmemBarrierAll();
    ShmemBarrierAll();
#endif

#ifdef __DAV_C220_VEC__
    uint64_t val = ShmemiLoad<uint64_t>(addr);

    ShmemBarrierAll();
    GM_ADDR remote = ShmemiPtr(addr, (rankId + 1) % rankSize);
    ShmemiStore<uint64_t>(remote, val + 1);
    ShmemBarrierAll();
#endif
}

void fetchAddrDo(uint8_t* syncArray, uint8_t* syncCounter) {
    fetchAddr<<<1, nullptr, nullptr>>>(syncArray, syncCounter);
}

void barrierDo(uint8_t *stub) {
    barrier<<<16, nullptr, nullptr>>>(stub);
}

void increaseDo(uint8_t *addr, int rankId, int rankSize) {
    increase<<<16, nullptr, nullptr>>>(addr, rankId, rankSize);
}
