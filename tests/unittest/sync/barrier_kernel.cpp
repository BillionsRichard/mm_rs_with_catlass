#include "kernel_operator.h"
#include "device/shmem_device_def.h"
#include "shmem_api.h"

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
    shmemi_team_t *team = shmemi_get_state()->team_pools[0];
    auto addr = (__gm__ half *) ((uint64_t) shmemi_get_team_sync_counter(team->team_idx) + SHMEMI_SYNCBIT_SIZE);
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

extern "C" SHMEM_GLOBAL void fetchAddr(GM_ADDR sync_array, GM_ADDR sync_counter) {
    shmemi_team_t *team = shmemi_get_state()->team_pools[0];
    *((__gm__ uint64_t*) sync_array) = (uint64_t) shmemi_get_team_sync_array(team->team_idx);
    *((__gm__ uint64_t*) sync_counter) = (uint64_t) shmemi_get_team_sync_counter(team->team_idx);
}

extern "C" SHMEM_GLOBAL void barrier(GM_ADDR stub) {
    CubeGuard();
    shmem_barrier_all();
    VecGuard();
}

extern "C" SHMEM_GLOBAL void increase(GM_ADDR addr, int rankId, int rankSize) {
    CVGuard();
    
#ifdef __DAV_C220_CUBE__
    // scalar unit of cube core is not affected by barrier
    shmem_barrier_all();
    shmem_barrier_all();
#endif

#ifdef __DAV_C220_VEC__
    uint64_t val = shmemi_load<uint64_t>(addr);

    shmem_barrier_all();
    GM_ADDR remote = shmemi_ptr(addr, (rankId + 1) % rankSize);
    shmemi_store<uint64_t>(remote, val + 1);
    shmem_barrier_all();
#endif
}

void fetchAddrDo(void* stream, uint8_t* sync_array, uint8_t* sync_counter) {
    fetchAddr<<<1, nullptr, stream>>>(sync_array, sync_counter);
}

void barrierDo(void* stream, uint8_t *stub) {
    barrier<<<16, nullptr, stream>>>(stub);
}

void increaseDo(void* stream, uint8_t *addr, int rankId, int rankSize) {
    increase<<<16, nullptr, stream>>>(addr, rankId, rankSize);
}
