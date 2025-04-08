#include "kernel_operator.h"
#include "smem_shm_aicore.h"
#include "smem_shm_aicore_common.h"

constexpr uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

template <typename T>
class ScalarPut {
public:
    __aicore__ inline ScalarPut() {}
    __aicore__ inline void Init(__gm__ T* addr, T value)
    {
        addrGm = addr;
        inValue = value;
    }

    __aicore__ inline void Process()
    {
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }

        *addrGm = inValue;

        __asm__ __volatile__("");
        dcci(static_cast<__gm__ void *>(addrGm), static_cast<uint64_t>(SINGLE_CACHE_LINE), static_cast<uint64_t>(CACHELINE_OUT));
        __asm__ __volatile__("");
    }
private:
    __gm__ T* addrGm = nullptr;
    T inValue;

    // TEventID maybe better?
    int event_id = 0;
};

template <typename T>
SMEM_INLINE_AICORE void shmem_p(__gm__ T* dst, const T value, int pe)
{
    ScalarPut<T> scalarPutKernel;
    // address translate
    uint64_t offset = SMEM_GET_SYMMETRIC_SIZE();
    uint64_t dst64 = reinterpret_cast<uint64_t>(dst) + offset * pe;

    scalarPutKernel.Init(reinterpret_cast<__gm__ T*>(dst64), value);
    scalarPutKernel.Process();
}

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
        shmem_p(gvaGm, value, (rank + 1) % rankSize);
    }
private:
    __gm__ float *gvaGm;
    float value;

    int64_t rank;
    int64_t rankSize;
};

extern "C" __global__ __aicore__ void put_num_test(GM_ADDR gva, float val)
{
    KernelPutNum op;
    op.Init(gva, val);
    op.Process();
}

void put_one_num_do(uint32_t blockDim, void* stream, uint8_t* gva, float val)
{
    put_num_test<<<blockDim, nullptr, stream>>>(gva, val);
}