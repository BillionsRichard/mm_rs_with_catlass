#include "kernel_operator.h"
#include "shmem_api.h"

class KernelP {
public:
    __aicore__ inline KernelP() {}
    __aicore__ inline void Init(GM_ADDR gva, float val)
    {
        gvaGm = (__gm__ float *)gva;
        value = val;

        rank = smem_shm_get_global_rank();
        rankSize = smem_shm_get_global_rank_size();
    }
    __aicore__ inline void Process()
    {
        shmem_float_p(gvaGm, value, (rank + 1) % rankSize);
    }
private:
    __gm__ float *gvaGm;
    float value;

    int64_t rank;
    int64_t rankSize;
};

extern "C" __global__ __aicore__ void PNumTest(GM_ADDR gva, float val)
{
    KernelP op;
    op.Init(gva, val);
    op.Process();
}

void PutOneNumDo(uint32_t blockDim, void* stream, uint8_t* gva, float val)
{
    PNumTest<<<blockDim, nullptr, stream>>>(gva, val);
}