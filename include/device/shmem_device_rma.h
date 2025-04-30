#ifndef SHMEM_DEVICE_RMA_H
#define SHMEM_DEVICE_RMA_H

#include "kernel_operator.h"
#include "internal/device/shmemi_device_common.h"
#include "shmem_device_team.h"

// Make Code Style Unified
#define Half half
#define Float float
#define Int8 int8_t
#define Int int
#define UInt8 uint8_t
#define Int16 int16_t
#define UInt16 uint16_t
#define Int64 int64_t
#define UInt64 uint64_t
#define Double double
#define Char char
#define Bool bool
#define BFloat16 bfloat16_t


#define SHMEM_TYPE_FUNC(fun)    \
    fun(Half);                  \
    fun(Float);                 \
    fun(Int8);                  \
    fun(Int);                   \
    fun(UInt8);                 \
    fun(Int16);                 \
    fun(UInt16);                \
    fun(Int64);                 \
    fun(UInt64);                \
    fun(Double);                \
    fun(Char);                  \
    fun(Bool);                  \
    fun(BFloat16)


SHMEM_DEVICE __gm__ void* ShmemPtr(__gm__ void* ptr, int pe)
{
    // Get Global State
    __gm__ ShmemiDeviceHostState *deviceState = ShmemiGetState();;

    // Check whether ptr belongs to this rank.
    uint64_t lowerBound = (uint64_t)deviceState->p2pHeapBase[ShmemMype()];
    uint64_t upperBound = lowerBound + deviceState->heapSize;
    if (uint64_t(ptr) < lowerBound || uint64_t(ptr) >= upperBound) {
        AscendC::printf("Rank %d Got Illegal Address !!\n", ShmemMype());
        return nullptr;
    }

    // Back to root address
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(deviceState->heapBase);
    
    // Address translate
    uint64_t remotePtr = reinterpret_cast<uint64_t>(deviceState->p2pHeapBase[pe]) + offset;

    return reinterpret_cast<__gm__ void*>(remotePtr);
}


#define SHMEM_TYPENAME_P_AICORE(inType)                                                     \
    SHMEM_DEVICE void ShmemP##inType(__gm__ inType* dst, const inType value, int pe)   \
    {                                                                                       \
        auto ptr = ShmemPtr(dst, pe);                                                       \
        if (ptr == nullptr) return;                                                         \
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
SHMEM_DEVICE void ShmemMTEGetMem(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ubSize, uint32_t elemSize, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = ShmemPtr(src, pe);
    if (ptr == nullptr) return;
    __gm__ T* remotePtr = reinterpret_cast<__gm__ T*>(ptr);

    // blockSize: dataMove Unit
    uint32_t blockSize = ubSize / sizeof(T) * sizeof(T);
    uint32_t remain = (elemSize * sizeof(T)) % blockSize;

    // TODO: USE DoubleBuffer.
    int repeat_times = (elemSize * sizeof(T)) / blockSize;
    int repeat_elem = blockSize / sizeof(T);
    for (int i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, remotePtr + i * repeat_elem, blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(dst + i * repeat_elem, buf, blockSize);
        if (i != repeat_times - 1) {    // Last PIPE Sync Should be done in Kernel
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        smem_shm_copy_gm2ub(buf, remotePtr + repeat_times * repeat_elem, remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(dst + repeat_times * repeat_elem, buf, remain);
    }
}


template <typename T>
SHMEM_DEVICE void ShmemMTEGetMemNBI(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elemSize, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = ShmemPtr((__gm__ void *)src.GetPhyAddr(), pe);
    if (ptr == nullptr) return;

    AscendC::GlobalTensor<T> remoteBuff;
    remoteBuff.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ptr));

    // blockSize: dataMove Unit
    uint32_t blockSize = buf.GetSize() / sizeof(T) * sizeof(T);
    uint32_t remain = (elemSize * sizeof(T)) % blockSize;

    // TODO: USE DoubleBuffer.
    int repeat_times = (elemSize * sizeof(T)) / blockSize;
    int repeat_elem = blockSize / sizeof(T);
    for (int i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, remoteBuff[i * repeat_elem], blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(dst[i * repeat_elem], buf, blockSize);
        if (i != repeat_times - 1) {    // Last PIPE Sync Should be done in Kernel
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        smem_shm_copy_gm2ub(buf, remoteBuff[repeat_times * repeat_elem], remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(dst[repeat_times * repeat_elem], buf, remain);
    }
}


template <typename T>
SHMEM_DEVICE void ShmemMTEPutMem(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ubSize, uint32_t elemSize, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = ShmemPtr(dst, pe);
    if (ptr == nullptr) return;
    __gm__ T* remotePtr = reinterpret_cast<__gm__ T*>(ptr);

    // blockSize: dataMove Unit
    uint32_t blockSize = ubSize / sizeof(T) * sizeof(T);
    uint32_t remain = (elemSize * sizeof(T)) % blockSize;

    // TODO: USE DoubleBuffer.
    int repeat_times = (elemSize * sizeof(T)) / blockSize;
    int repeat_elem = blockSize / sizeof(T);
    for (int i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, src + i * repeat_elem, blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remotePtr + i * repeat_elem, buf, blockSize);
        if (i != repeat_times - 1) {    // Last PIPE Sync Should be done in Kernel
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        smem_shm_copy_gm2ub(buf, src + repeat_times * repeat_elem, remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remotePtr + repeat_times * repeat_elem, buf, remain);
    }
}


template <typename T>
SHMEM_DEVICE void ShmemMTEPutMemNBI(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elemSize, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = ShmemPtr((__gm__ void *)dst.GetPhyAddr(), pe);
    if (ptr == nullptr) return;

    AscendC::GlobalTensor<T> remoteBuff;
    remoteBuff.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ptr));

    // blockSize: dataMove Unit
    uint32_t blockSize = buf.GetSize() / sizeof(T) * sizeof(T);
    uint32_t remain = (elemSize * sizeof(T)) % blockSize;

    // TODO: USE DoubleBuffer.
    int repeat_times = (elemSize * sizeof(T)) / blockSize;
    int repeat_elem = blockSize / sizeof(T);
    for (int i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, src[i * repeat_elem], blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remoteBuff[i * repeat_elem], buf, blockSize);
        if (i != repeat_times - 1) {    // Last PIPE Sync Should be done in Kernel
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        smem_shm_copy_gm2ub(buf, src[repeat_times * repeat_elem], remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remoteBuff[repeat_times * repeat_elem], buf, remain);
    }
}


#define SHMEM_GET_TYPENAME_MEM(inType)                                                                                  \
    SHMEM_DEVICE void ShmemGet##inType##Mem(__gm__ inType* dst, __gm__ inType* src, uint32_t elemSize, int pe)     \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ void* addrGM = smem_shm_get_extra_context_addr();                                                        \
        __gm__ ShmemiDeviceHostState *deviceState = (__gm__ ShmemiDeviceHostState *)addrGM;                               \
        /* CopyUB Config Set */                                                                                         \
        uint64_t copyUB = deviceState->mteConfig.shmemUB;                                                               \
        uint32_t copyUBSize = deviceState->mteConfig.ubSize;                                                            \
        AscendC::TEventID copyEventID = (AscendC::TEventID)deviceState->mteConfig.eventID;                              \
        ShmemMTEGetMem(dst, src, reinterpret_cast<__ubuf__ inType*>(copyUB), copyUBSize, elemSize, pe, copyEventID);    \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM);


#define SHMEM_GET_TYPENAME_MEM_TENSOR(inType)                                                                           \
    SHMEM_DEVICE void ShmemGet##inType##MemNBI(AscendC::GlobalTensor<inType> dst, AscendC::GlobalTensor<inType> src, uint32_t elemSize, int pe)   \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ void* addrGM = smem_shm_get_extra_context_addr();                                                        \
        __gm__ ShmemiDeviceHostState *deviceState = (__gm__ ShmemiDeviceHostState *)addrGM;                             \
        /* CopyUB Config Set */                                                                                         \
        uint64_t copyUB = deviceState->mteConfig.shmemUB;                                                               \
        /* Create LocalTensor */                                                                                        \
        AscendC::LocalTensor<inType> ubTensor;                                                                          \
        ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                   \
        ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copyUB);                                              \
        ubTensor.address_.dataLen = deviceState->mteConfig.ubSize;                                                      \
        AscendC::TEventID copyEventID = (AscendC::TEventID)deviceState->mteConfig.eventID;                              \
        ShmemMTEGetMemNBI(dst, src, ubTensor, elemSize, pe, copyEventID);                                               \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_TENSOR);


#define SHMEM_PUT_TYPENAME_MEM(inType)                                                                                  \
    SHMEM_DEVICE void ShmemPut##inType##Mem(__gm__ inType* dst, __gm__ inType* src, uint32_t elemSize, int pe)     \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ void* addrGM = smem_shm_get_extra_context_addr();                                                        \
        __gm__ ShmemiDeviceHostState *deviceState = (__gm__ ShmemiDeviceHostState *)addrGM;                               \
        /* CopyUB Config Set */                                                                                         \
        uint64_t copyUB = deviceState->mteConfig.shmemUB;                                                               \
        uint32_t copyUBSize = deviceState->mteConfig.ubSize;                                                            \
        AscendC::TEventID copyEventID = (AscendC::TEventID)deviceState->mteConfig.eventID;                              \
        ShmemMTEPutMem(dst, src, reinterpret_cast<__ubuf__ inType*>(copyUB), copyUBSize, elemSize, pe, copyEventID);    \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM);


#define SHMEM_PUT_TYPENAME_MEM_TENSOR(inType)                                                                           \
    SHMEM_DEVICE void ShmemPut##inType##MemNBI(AscendC::GlobalTensor<inType> dst, AscendC::GlobalTensor<inType> src, uint32_t elemSize, int pe)   \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ void* addrGM = smem_shm_get_extra_context_addr();                                                        \
        __gm__ ShmemiDeviceHostState *deviceState = (__gm__ ShmemiDeviceHostState *)addrGM;                             \
        /* CopyUB Config Set */                                                                                         \
        uint64_t copyUB = deviceState->mteConfig.shmemUB;                                                               \
        /* Create LocalTensor */                                                                                        \
        AscendC::LocalTensor<inType> ubTensor;                                                                          \
        ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                   \
        ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copyUB);                                              \
        ubTensor.address_.dataLen = deviceState->mteConfig.ubSize;                                                      \
        AscendC::TEventID copyEventID = (AscendC::TEventID)deviceState->mteConfig.eventID;                              \
        ShmemMTEPutMemNBI(dst, src, ubTensor, elemSize, pe, copyEventID);                                               \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_TENSOR);


#endif
