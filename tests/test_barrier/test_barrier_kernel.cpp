#include "kernel_operator.h"

#include "shmem_device_api.h"

extern "C" __global__ __aicore__ void fetchAddr(GM_ADDR syncArray, GM_ADDR syncCounter) {
    ShmemiTeam *team = ShmemiGetState()->teamPools[0];
    *((__gm__ uint64_t*) syncArray) = (uint64_t) ShmemiGetTeamSyncArray(team);
    *((__gm__ uint64_t*) syncCounter) = (uint64_t) ShmemiGetTeamSyncCounter(team);
}

extern "C" __global__ __aicore__ void barrier(GM_ADDR stub) {
    CubeGuard();
    ShmemBarrierAll();
    VecGuard();
}

extern "C" __global__ __aicore__ void increase(GM_ADDR addr, int rankId, int rankSize) {
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
