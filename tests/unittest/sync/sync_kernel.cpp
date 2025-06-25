#include "kernel_operator.h"
#include "shmem_api.h"

extern "C" SHMEM_GLOBAL void increase(uint64_t config, GM_ADDR addr, int rank_id, int rank_size) {
    shmemx_set_ffts_config(config);
    
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

extern "C" SHMEM_GLOBAL void p2p_chain(uint64_t config, GM_ADDR addr, int rank_id, int rank_size) {
    shmemx_set_ffts_config(config);
    auto sig_addr = (__gm__ int32_t *)addr;
    int32_t val = *sig_addr;
    int next = (rank_id + 1) % rank_size;

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

void increase_do(void* stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size) {
    increase<<<16, nullptr, stream>>>(config, addr, rank_id, rank_size);
}

void p2p_chain_do(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size) {
    p2p_chain<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size);
}