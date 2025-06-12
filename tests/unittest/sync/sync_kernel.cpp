#include "kernel_operator.h"
#include "shmem_api.h"

SHMEM_DEVICE void cube_guard() {
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

SHMEM_DEVICE void vec_guard() {
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
    cube_guard();
    vec_guard();
}

extern "C" SHMEM_GLOBAL void increase(GM_ADDR addr, int rank_id, int rank_size) {
    CVGuard();
    
#ifdef __DAV_C220_CUBE__
    // scalar unit of cube core is not affected by barrier
    shmem_barrier_all();
    shmem_barrier_all();
#endif

#ifdef __DAV_C220_VEC__
    uint64_t val = shmemi_load<uint64_t>(addr);

    shmem_barrier_all();
    GM_ADDR remote = shmemi_ptr(addr, (rank_id + 1) % rank_size);
    shmemi_store<uint64_t>(remote, val + 1);
    shmem_barrier_all();
#endif
}

extern "C" SHMEM_GLOBAL void p2p_chain(GM_ADDR addr, int rank_id, int rank_size) {
    auto sig_addr = (__gm__ int32_t *)addr;
    int32_t val = *sig_addr;
    int next = (rank_id + 1) % rank_size;

    CVGuard();
    shmem_barrier_all();

#ifdef __DAV_C220_VEC__
    if (rank_id == 0) {
        shmemx_signal_op(sig_addr, 1, SHMEM_SIGNAL_SET, next);
        shmem_signal_wait_until(sig_addr, SHMEM_CMP_EQ, 1);
    } else {
        shmem_signal_wait_until(sig_addr, SHMEM_CMP_EQ, 1);
        shmemx_signal_op(sig_addr, 1, SHMEM_SIGNAL_SET, next);
    }
#endif

    shmem_barrier_all();

#ifdef __DAV_C220_VEC__
    if (rank_id == 0) {
        shmemx_signal_op(sig_addr, 1, SHMEM_SIGNAL_ADD, next);
        shmem_signal_wait_until(sig_addr, SHMEM_CMP_EQ, 3);
    } else {
        shmem_signal_wait_until(sig_addr, SHMEM_CMP_EQ, 3);
        shmemx_signal_op(sig_addr, 1, SHMEM_SIGNAL_ADD, next);
    }
#endif

    shmem_barrier_all();
}

void increase_do(void* stream, uint8_t *addr, int rank_id, int rank_size) {
    increase<<<16, nullptr, stream>>>(addr, rank_id, rank_size);
}

void p2p_chain_do(void *stream, uint8_t *addr, int rank_id, int rank_size) {
    p2p_chain<<<1, nullptr, stream>>>(addr, rank_id, rank_size);
}