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