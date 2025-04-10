#include "kernel_operator.h"
#include "smem_shm_aicore.h"
#include "smem_shm_aicore_common.h"

#include "scalar_p.hpp"

class KernelPutNum {
public:
    __aicore__ inline KernelPutNum() {}
    __aicore__ inline void Init(GM_ADDR gva, float val)
    {
        gvaGm = (__gm__ float *)gva;
        value = val;

        rank = SMEM_GET_LOCAL_RANK();
        rankSize = SMEM_GET_RANK_SIZE();
    }
    __aicore__ inline void Process()
    {
        ShmemP(gvaGm, value, (rank + 1) % rankSize);
    }
private:
    __gm__ float *gvaGm;
    float value;

    int64_t rank;
    int64_t rankSize;
};

extern "C" __global__ __aicore__ void PutNumTest(GM_ADDR gva, float val)
{
    KernelPutNum op;
    op.Init(gva, val);
    op.Process();
}

void PutOneNumDo(uint32_t blockDim, void* stream, uint8_t* gva, float val)
{
    PutNumTest<<<blockDim, nullptr, stream>>>(gva, val);
}