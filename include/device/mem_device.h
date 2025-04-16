#ifndef _SHMEM_MEM_DEVICE_H_
#define _SHMEM_MEM_DEVICE_H_

#include "kernel_operator.h"
#include "low_level_api/smem_shm_aicore_base_api.h"

#include "shmem_device_api.h"

#define SHMEM_TYPE_FUNC(fun)    \
    fun(int);                   \
    fun(half);                  \
    fun(float)


__aicore__ inline __gm__ void* ShmemPtr(__gm__ void* ptr, int pe)
{
    // address translate
    uint64_t offset = smem_shm_get_symmetric_size();
    uint64_t remotePtr = reinterpret_cast<uint64_t>(ptr) + offset * pe;

    return reinterpret_cast<__gm__ void*>(remotePtr);
}


#define SHMEM_TYPENAME_P_AICORE(inType)                                                     \
    __aicore__ inline void ShmemP_##inType(__gm__ inType* dst, const inType value, int pe)  \
    {                                                                                       \
        if (AscendC::GetSubBlockIdx() != 0) {                                               \
            return;                                                                         \
        }                                                                                   \
                                                                                            \
        auto ptr = ShmemPtr(dst, pe);                                                       \
        __gm__ inType* addrGm = reinterpret_cast<__gm__ inType*>(ptr);                      \
                                                                                            \
        *addrGm = value;                                                                    \
        AscendC::GlobalTensor<uint64_t> global;                                             \
        global.SetGlobalBuffer((__gm__ uint64_t*)addrGm);                                   \
                                                                                            \
        /* 首地址64B对齐，调用DataCacheCleanAndInvalid指令后，会立刻刷新前8个数 */              \
        AscendC::DataCacheCleanAndInvalid<uint64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(global);    \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPENAME_P_AICORE);


template <typename T>
__aicore__ inline void ShmemCopyUbuf(__ubuf__ T* srcUb, uint32_t size)
{
    smem_set_copy_ubuf(srcUb, size);
}


template <typename T>
__aicore__ inline void ShmemGetMem(__gm__ T* dst, __gm__ T* src, uint32_t copySize, int pe)
{
    // TODO
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }
}


template <typename T>
__aicore__ inline void ShmemMTEGetMem(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ubSize, uint32_t copySize, int pe, AscendC::TEventID EVENT_ID)
{
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }

    auto ptr = ShmemPtr(src, pe);
    __gm__ T* remotePtr = reinterpret_cast<__gm__ T*>(ptr);

    // blockSize: dataMove Unit
    uint32_t blockSize = ubSize / sizeof(T) * sizeof(T);
    uint32_t remain = copySize % blockSize;

    // TODO: USE DoubleBuffer.
    int repeat_times = copySize / blockSize;
    for (int i = 0; i < repeat_times; i++) {
        smem_copy_gm2ub(buf, remotePtr + i * blockSize * sizeof(T), blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_copy_ub2gm(dst + i * blockSize * sizeof(T), buf, blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
    if (remain > 0) {
        smem_copy_gm2ub(buf, remotePtr + repeat_times * blockSize * sizeof(T), remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_copy_ub2gm(dst + repeat_times * blockSize * sizeof(T), buf, remain);
    }
}


template <typename T>
__aicore__ inline void ShmemMTEPutMem(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ubSize, uint32_t copySize, int pe, AscendC::TEventID EVENT_ID)
{
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }

    auto ptr = ShmemPtr(dst, pe);
    __gm__ T* remotePtr = reinterpret_cast<__gm__ T*>(ptr);

    // blockSize: dataMove Unit
    uint32_t blockSize = ubSize / sizeof(T) * sizeof(T);
    uint32_t remain = copySize % blockSize;

    // TODO: USE DoubleBuffer.
    int repeat_times = copySize / blockSize;
    for (int i = 0; i < repeat_times; i++) {
        smem_copy_gm2ub(buf, src + i * blockSize * sizeof(T), blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_copy_ub2gm(remotePtr + i * blockSize * sizeof(T), buf, blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
    if (remain > 0) {
        smem_copy_gm2ub(buf, src + repeat_times * blockSize * sizeof(T), remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_copy_ub2gm(remotePtr + repeat_times * blockSize * sizeof(T), buf, remain);
    }
}

#endif
