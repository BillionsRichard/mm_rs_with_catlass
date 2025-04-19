#include "kernel_operator.h"

#include "shmem_sync.h"

extern "C" __global__ __aicore__ void fetchAddr(GM_ADDR syncArray, GM_ADDR syncCounter) {
    ShmemTeam *team = getState()->teamPools[0];
    *((__gm__ uint64_t*) syncArray) = (uint64_t) ShmemiGetTeamSyncArray(team);
    *((__gm__ uint64_t*) syncCounter) = (uint64_t) ShmemiGetTeamSyncCounter(team);
}

extern "C" __global__ __aicore__ void barrier(GM_ADDR stub) {
    ShmemBarrierAll();
}

extern "C" __global__ __aicore__ void increase(GM_ADDR addr, int rankId, int rankSize) {
    uint64_t val = load<uint64_t>(addr);

    ShmemBarrierAll();
    GM_ADDR remote = ShmemiPtr(addr, (rankId + 1) % rankSize);
    store<uint64_t>(remote, val + 1);
    ShmemBarrierAll();
}

void fetchAddrDo(uint8_t* syncArray, uint8_t* syncCounter) {
    fetchAddr<<<1, nullptr, nullptr>>>(syncArray, syncCounter);
}

void barrierDo(uint8_t *stub) {
    barrier<<<1, nullptr, nullptr>>>(stub);
}

void increaseDo(uint8_t *addr, int rankId, int rankSize) {
    increase<<<1, nullptr, nullptr>>>(addr, rankId, rankSize);
}
