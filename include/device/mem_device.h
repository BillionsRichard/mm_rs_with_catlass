#ifndef _SHMEM_MEM_DEVICE_H_
#define _SHMEM_MEM_DEVICE_H_

#include "kernel_operator.h"
#include "low_level_api/smem_shm_aicore_base_api.h"

#include "shmem_device_api.h"

__BLOCK_LOCAL__ __inline__ int64_t copyUbuf;
__BLOCK_LOCAL__ __inline__ int32_t ubufSize;
__BLOCK_LOCAL__ __inline__ int32_t copyEventID;

// Make Code Style Unified
#define Int int
#define Half half
#define Float float
#define Char char


#define SHMEM_TYPE_FUNC(fun)    \
    fun(Int);                   \
    fun(Half);                  \
    fun(Float)


__aicore__ inline __gm__ void* ShmemPtr(__gm__ void* ptr, int pe)
{
    // Back to root address
    __gm__ void* addrGM = smem_shm_get_extra_context_addr();
    __gm__ ShmemDeviceHostState *deviceState = (__gm__ ShmemDeviceHostState *)addrGM;
    void *mypePtr = deviceState->p2pHeapBase[ShmemMype()];
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(mypePtr);
    
    // Address translate
    uint64_t heapMemSize = smem_shm_get_symmetric_size();
    uint64_t remotePtr = reinterpret_cast<uint64_t>(deviceState->heapBase) + heapMemSize * pe + offset;

    return reinterpret_cast<__gm__ void*>(remotePtr);
}


#define SHMEM_TYPENAME_P_AICORE(inType)                                                     \
    __aicore__ inline void ShmemP##inType(__gm__ inType* dst, const inType value, int pe)   \
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


__aicore__ inline void ShmemSetDefaultMTEConfig()
{
    copyUbuf = 0;
    ubufSize = 256; // Bytes
    copyEventID = EVENT_ID0;
}


template <typename T>
__aicore__ inline void ShmemSetMTEConfig(__ubuf__ T* srcUb, uint32_t size, AscendC::TEventID EVENT_ID)
{
    copyUbuf = reinterpret_cast<uint64_t>(srcUb);
    ubufSize = size;
    copyEventID = EVENT_ID;
}


template <typename T>
__aicore__ inline void ShmemUnSetMTEConfig(__ubuf__ T* srcUb, uint32_t size, AscendC::TEventID EVENT_ID)
{
    copyUbuf = -1;
    ubufSize = -1;
    copyEventID = EVENT_ID0;
}


template <typename T>
__aicore__ inline void ShmemMTEGetMem(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ubSize, uint32_t elemSize, int pe, AscendC::TEventID EVENT_ID)
{
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }

    auto ptr = ShmemPtr(src, pe);
    __gm__ T* remotePtr = reinterpret_cast<__gm__ T*>(ptr);

    // blockSize: dataMove Unit
    uint32_t blockSize = ubSize / sizeof(T) * sizeof(T);
    uint32_t remain = (elemSize * sizeof(T)) % blockSize;

    // TODO: USE DoubleBuffer.
    int repeat_times = (elemSize * sizeof(T)) / blockSize;
    int repeat_elem = blockSize / sizeof(T);
    for (int i = 0; i < repeat_times; i++) {
        smem_copy_gm2ub(buf, remotePtr + i * repeat_elem, blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_copy_ub2gm(dst + i * repeat_elem, buf, blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
    if (remain > 0) {
        smem_copy_gm2ub(buf, remotePtr + repeat_times * repeat_elem, remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_copy_ub2gm(dst + repeat_times * repeat_elem, buf, remain);
    }
}


template <typename T>
__aicore__ inline void ShmemMTEPutMem(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ubSize, uint32_t elemSize, int pe, AscendC::TEventID EVENT_ID)
{
    if (AscendC::GetSubBlockIdx() != 0) {
        return;
    }

    auto ptr = ShmemPtr(dst, pe);
    __gm__ T* remotePtr = reinterpret_cast<__gm__ T*>(ptr);

    // blockSize: dataMove Unit
    uint32_t blockSize = ubSize / sizeof(T) * sizeof(T);
    uint32_t remain = (elemSize * sizeof(T)) % blockSize;

    // TODO: USE DoubleBuffer.
    int repeat_times = (elemSize * sizeof(T)) / blockSize;
    int repeat_elem = blockSize / sizeof(T);
    for (int i = 0; i < repeat_times; i++) {
        smem_copy_gm2ub(buf, src + i * repeat_elem, blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_copy_ub2gm(remotePtr + i * repeat_elem, buf, blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
    if (remain > 0) {
        smem_copy_gm2ub(buf, src + repeat_times * repeat_elem, remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_copy_ub2gm(remotePtr + repeat_times * repeat_elem, buf, remain);
    }
}


#define SHMEM_GET_TYPENAME_MEM(inType)                                                                                  \
    __aicore__ inline void ShmemGet##inType##Mem(__gm__ inType* dst, __gm__ inType* src, uint32_t elemSize, int pe)     \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
                                                                                                                        \
        __gm__ void* addrGM = smem_shm_get_extra_context_addr();                                                        \
        __gm__ ShmemDeviceHostState *deviceState = (__gm__ ShmemDeviceHostState *)addrGM;                               \
        if (deviceState->mteConfig.ubSize == 0) {                                                                       \
            ShmemSetDefaultMTEConfig();                                                                                 \
        } else {                                                                                                        \
            ShmemSetMTEConfig(                                                                                          \
                reinterpret_cast<__ubuf__ inType*>(deviceState->mteConfig.tmpUb),                                       \
                deviceState->mteConfig.ubSize,                                                                          \
                AscendC::TEventID(deviceState->mteConfig.eventID));                                                     \
        }                                                                                                               \
        ShmemMTEGetMem(dst, src, reinterpret_cast<__ubuf__ inType*>(copyUbuf), ubufSize, elemSize, pe, copyEventID);    \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM);


#define SHMEM_PUT_TYPENAME_MEM(inType)                                                                                  \
    __aicore__ inline void ShmemPut##inType##Mem(__gm__ inType* dst, __gm__ inType* src, uint32_t elemSize, int pe)     \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
                                                                                                                        \
        __gm__ void* addrGM = smem_shm_get_extra_context_addr();                                                        \
        __gm__ ShmemDeviceHostState *deviceState = (__gm__ ShmemDeviceHostState *)addrGM;                               \
        if (deviceState->mteConfig.ubSize == 0) {                                                                       \
            ShmemSetDefaultMTEConfig();                                                                                 \
        } else {                                                                                                        \
            ShmemSetMTEConfig(                                                                                          \
                reinterpret_cast<__ubuf__ inType*>(deviceState->mteConfig.tmpUb),                                       \
                deviceState->mteConfig.ubSize,                                                                          \
                AscendC::TEventID(deviceState->mteConfig.eventID));                                                     \
        }                                                                                                               \
        ShmemMTEPutMem(dst, src, reinterpret_cast<__ubuf__ inType*>(copyUbuf), ubufSize, elemSize, pe, copyEventID);    \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM);


#endif
