#include "kernel_operator.h"
#include "shmem_api.h"

class kernel_p {
public:
    __aicore__ inline kernel_p() {}
    __aicore__ inline void Init(GM_ADDR gva, float val)
    {
        gva_gm = (__gm__ float *)gva;
        value = val;

        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();
    }
    __aicore__ inline void Process()
    {
        shmem_float_p(gva_gm, value, (rank + 1) % rank_size);
    }
private:
    __gm__ float *gva_gm;
    float value;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void p_num_test(GM_ADDR gva, float val)
{
    kernel_p op;
    op.Init(gva, val);
    op.Process();
}

void put_one_num_do(uint32_t block_dim, void* stream, uint8_t* gva, float val)
{
    p_num_test<<<block_dim, nullptr, stream>>>(gva, val);
}