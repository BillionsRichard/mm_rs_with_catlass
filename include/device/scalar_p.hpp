#include "kernel_operator.h"
#include "smem_shm_aicore.h"
#include "smem_shm_aicore_common.h"

constexpr uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

template <typename T>
SMEM_INLINE_AICORE void ShmemP(__gm__ T* dst, const T value, int pe)
{
    // address translate
    uint64_t offset = SMEM_GET_SYMMETRIC_SIZE();
    uint64_t dst64 = reinterpret_cast<uint64_t>(dst) + offset * pe;

    __gm__ T* addrGm = reinterpret_cast<__gm__ T*>(dst64);
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }
    *addrGm = value;

    AscendC::GlobalTensor<uint64_t> global;
    global.SetGlobalBuffer((__gm__ uint64_t*)addrGm);

    // 首地址64B对齐，调用DataCacheCleanAndInvalid指令后，会立刻刷新前8个数
    AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);
}