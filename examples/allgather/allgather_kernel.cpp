#ifndef _KERNEL_ALLGATHER_HPP
#define _KERNEL_ALLGATHER_HPP

#include "kernel_operator.h"
#include "shmem_api.h"

extern "C" __global__ __aicore__ void device_all_gather_test(GM_ADDR gva)
{
    int64_t my_rank = shmem_my_pe();
    int64_t pe_size = shmem_n_pes();
    __gm__ int32_t* gva_gm = (__gm__ int32_t *)gva;
    AscendC::PipeBarrier<PIPE_ALL>();
    // All Gather
    for (int i = 0; i < pe_size - 1; i++) {
        shmem_put_int32_mem_nbi(gva_gm + 16 * my_rank, gva_gm + 16 * my_rank, 16, i);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    shmemi_barrier_all();
}

void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva)
{
    device_all_gather_test<<<block_dim, nullptr, stream>>>(gva);
}



#endif  // _KERNEL_ALLGATHER_HPP