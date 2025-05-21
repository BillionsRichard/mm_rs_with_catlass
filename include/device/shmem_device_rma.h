#ifndef SHMEM_DEVICE_RMA_H
#define SHMEM_DEVICE_RMA_H

#include "kernel_operator.h"
#include "internal/device/shmemi_device_common.h"
#include "shmem_device_team.h"

#define SHMEM_TYPE_FUNC(FUNC)        \
    FUNC(half, half);                \
    FUNC(float, float);              \
    FUNC(double, double);            \
    FUNC(int8, int8_t);              \
    FUNC(int16, int16_t);            \
    FUNC(int32, int32_t);            \
    FUNC(int64, int64_t);            \
    FUNC(uint8, uint8_t);            \
    FUNC(uint16, uint16_t);          \
    FUNC(uint32, uint32_t);          \
    FUNC(uint64, uint64_t);          \
    FUNC(char, char);                \
    FUNC(bfloat16, bfloat16_t)

/**
 * @brief Translate an local symmetric address to remote symmetric address on the specified PE.
 *        Firstly, check whether the input address is legal on local PE. Then translate it into remote address 
 *        on specified PE. Otherwise, returns a null pointer.
 *
 * @param ptr               [in] Symmetric address on local PE.
 * @param pe                [in] The number of the remote PE.
 * @return If the input address is legal, returns a remote symmetric address on the specified PE that can be 
 *         accessed using memory loads and stores. Otherwise, a null pointer is returned.
 */
SHMEM_DEVICE __gm__ void* shmem_ptr(__gm__ void* ptr, int pe)
{
    // Get Global State
    __gm__ ShmemiDeviceHostState *deviceState = ShmemiGetState();

    // Check whether ptr belongs to this rank.
    uint64_t lowerBound = (uint64_t)deviceState->p2pHeapBase[shmem_my_pe()];
    uint64_t upperBound = lowerBound + deviceState->heapSize;
    if (uint64_t(ptr) < lowerBound || uint64_t(ptr) >= upperBound) {
        return nullptr;
    }

    // Back to root address
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(deviceState->heapBase);
    
    // Address translate
    uint64_t remotePtr = reinterpret_cast<uint64_t>(deviceState->p2pHeapBase[pe]) + offset;

    return reinterpret_cast<__gm__ void*>(remotePtr);
}


/**
 * @brief Provide a low latency put capability for single element of most basic types.
 *
 * @param dst               [in] Symmetric address of the destination data on local PE.
 * @param value             [in] The element to be put.
 * @param pe                [in] The number of the remote PE.
 * @return void
 */
#define SHMEM_TYPENAME_P_AICORE(NAME, TYPE)                                                 \
    SHMEM_DEVICE void shmem_##NAME##_p(__gm__ TYPE* dst, const TYPE value, int pe)          \
    {                                                                                       \
        auto ptr = shmem_ptr(dst, pe);                                                      \
        if (ptr == nullptr) return;                                                         \
        __gm__ TYPE* addrGm = reinterpret_cast<__gm__ TYPE*>(ptr);                          \
                                                                                            \
        *addrGm = value;                                                                    \
        DcciCacheline((__gm__ uint8_t *)addrGm);                                            \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPENAME_P_AICORE);


/**
 * @brief Provide a low latency get capability for single element of most basic types.
 *
 * @param src               [in] Symmetric address of the destination data on local PE.
 * @param pe                [in] The number of the remote PE.
 * @return A single element of type specified in the input pointer.
 */
#define SHMEM_TYPENAME_G_AICORE(NAME, TYPE)                                                 \
    SHMEM_DEVICE TYPE shmem_##NAME##_g(__gm__ TYPE* src, int32_t pe)                        \
    {                                                                                       \
        auto ptr = shmem_ptr(src, pe);                                                      \
        __gm__ TYPE* addrGm = reinterpret_cast<__gm__ TYPE*>(ptr);                          \
                                                                                            \
        DcciCacheline((__gm__ uint8_t *)addrGm);                                            \
        return *addrGm;                                                                     \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPENAME_G_AICORE);

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local device.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param elemSize          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\MTE3 Event.
 * @return void
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ubSize, uint32_t elemSize, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(src, pe);
    if (ptr == nullptr) return;
    __gm__ T* remotePtr = reinterpret_cast<__gm__ T*>(ptr);

    // blockSize: dataMove Unit
    uint32_t blockSize = ubSize / sizeof(T) * sizeof(T);
    uint32_t remain = (elemSize * sizeof(T)) % blockSize;

    int repeat_times = (elemSize * sizeof(T)) / blockSize;
    int repeat_elem = blockSize / sizeof(T);
    int loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (int i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, remotePtr + i * repeat_elem, blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(dst + i * repeat_elem, buf, blockSize);
        if (i != loop_times - 1) {      // Last PIPE Sync Should be done outside
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

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param elemSize          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\MTE3 Event.
 * @return void
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elemSize, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);
    if (ptr == nullptr) return;

    AscendC::GlobalTensor<T> remoteBuff;
    remoteBuff.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ptr));

    // blockSize: dataMove Unit
    uint32_t blockSize = buf.GetSize() / sizeof(T) * sizeof(T);
    uint32_t remain = (elemSize * sizeof(T)) % blockSize;

    int repeat_times = (elemSize * sizeof(T)) / blockSize;
    int repeat_elem = blockSize / sizeof(T);
    int loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (int i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, remoteBuff[i * repeat_elem], blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(dst[i * repeat_elem], buf, blockSize);
        if (i != loop_times - 1) {      // Last PIPE Sync Should be done outside
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

/**
 * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param elemSize          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\MTE3 Event.
 * @return void
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ubSize, uint32_t elemSize, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(dst, pe);
    if (ptr == nullptr) return;
    __gm__ T* remotePtr = reinterpret_cast<__gm__ T*>(ptr);

    // blockSize: dataMove Unit
    uint32_t blockSize = ubSize / sizeof(T) * sizeof(T);
    uint32_t remain = (elemSize * sizeof(T)) % blockSize;

    int repeat_times = (elemSize * sizeof(T)) / blockSize;
    int repeat_elem = blockSize / sizeof(T);
    int loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (int i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, src + i * repeat_elem, blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remotePtr + i * repeat_elem, buf, blockSize);
        if (i != loop_times - 1) {      // Last PIPE Sync Should be done outside
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

/**
 * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param elemSize          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\MTE3 Event.
 * @return void
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elemSize, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);
    if (ptr == nullptr) return;

    AscendC::GlobalTensor<T> remoteBuff;
    remoteBuff.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ptr));

    // blockSize: dataMove Unit
    uint32_t blockSize = buf.GetSize() / sizeof(T) * sizeof(T);
    uint32_t remain = (elemSize * sizeof(T)) % blockSize;

    int repeat_times = (elemSize * sizeof(T)) / blockSize;
    int repeat_elem = blockSize / sizeof(T);
    int loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (int i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, src[i * repeat_elem], blockSize);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remoteBuff[i * repeat_elem], buf, blockSize);
        if (i != loop_times - 1) {      // Last PIPE Sync Should be done outside
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


/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elemSize          [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @return void
 */
#define SHMEM_GET_TYPENAME_MEM(NAME, TYPE)                                                                              \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, uint32_t elemSize, int32_t pe)     \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ ShmemiDeviceHostState *deviceState = ShmemiGetState();                                                   \
        /* CopyUB Config Set */                                                                                         \
        uint64_t copyUB = deviceState->mteConfig.shmemUB;                                                               \
        uint32_t copyUBSize = deviceState->mteConfig.ubSize;                                                            \
        AscendC::TEventID copyEventID = (AscendC::TEventID)deviceState->mteConfig.eventID;                              \
        shmem_mte_get_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE*>(copyUB), copyUBSize, elemSize, pe, copyEventID);   \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM);


/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param elemSize          [in] Number of elements in the dest and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @return void
 */
#define SHMEM_GET_TYPENAME_MEM_TENSOR(NAME, TYPE)                                                                           \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, uint32_t elemSize, int pe)   \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ ShmemiDeviceHostState *deviceState = ShmemiGetState();                                                   \
        /* CopyUB Config Set */                                                                                         \
        uint64_t copyUB = deviceState->mteConfig.shmemUB;                                                               \
        /* Create LocalTensor */                                                                                        \
        AscendC::LocalTensor<TYPE> ubTensor;                                                                          \
        ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                   \
        ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copyUB);                                              \
        ubTensor.address_.dataLen = deviceState->mteConfig.ubSize;                                                      \
        AscendC::TEventID copyEventID = (AscendC::TEventID)deviceState->mteConfig.eventID;                              \
        shmem_mte_get_mem_nbi(dst, src, ubTensor, elemSize, pe, copyEventID);                                               \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_TENSOR);


/**
 * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param elemSize          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @return void
 */
#define SHMEM_PUT_TYPENAME_MEM(NAME, TYPE)                                                                              \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, uint32_t elemSize, int32_t pe)         \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ ShmemiDeviceHostState *deviceState = ShmemiGetState();                                                   \
        /* CopyUB Config Set */                                                                                         \
        uint64_t copyUB = deviceState->mteConfig.shmemUB;                                                               \
        uint32_t copyUBSize = deviceState->mteConfig.ubSize;                                                            \
        AscendC::TEventID copyEventID = (AscendC::TEventID)deviceState->mteConfig.eventID;                              \
        shmem_mte_put_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE*>(copyUB), copyUBSize, elemSize, pe, copyEventID);      \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM);


/**
 * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param elemSize          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @return void
 */
#define SHMEM_PUT_TYPENAME_MEM_TENSOR(NAME, TYPE)                                                                           \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, uint32_t elemSize, int pe)   \
    {                                                                                                                   \
        /* ROCE */                                                                                                      \
        /* RDMA */                                                                                                      \
        /* MTE  */                                                                                                      \
        /* Global State Get */                                                                                          \
        __gm__ ShmemiDeviceHostState *deviceState = ShmemiGetState();                                                   \
        /* CopyUB Config Set */                                                                                         \
        uint64_t copyUB = deviceState->mteConfig.shmemUB;                                                               \
        /* Create LocalTensor */                                                                                        \
        AscendC::LocalTensor<TYPE> ubTensor;                                                                          \
        ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                   \
        ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copyUB);                                              \
        ubTensor.address_.dataLen = deviceState->mteConfig.ubSize;                                                      \
        AscendC::TEventID copyEventID = (AscendC::TEventID)deviceState->mteConfig.eventID;                              \
        shmem_mte_put_mem_nbi(dst, src, ubTensor, elemSize, pe, copyEventID);                                               \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_TENSOR);

#endif
