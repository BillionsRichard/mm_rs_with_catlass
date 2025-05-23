#ifndef SHMEM_DEVICE_LOW_LEVEL_RMA_H
#define SHMEM_DEVICE_LOW_LEVEL_RMA_H

#include "kernel_operator.h"
#include "internal/device/shmemi_device_common.h"
#include "device/shmem_device_team.h"


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
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local device.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ubSize            [in] The size of temp Buffer on UB. (In Bytes)
 * @param elemSize          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
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
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data 
 *        on symmetric memory from the specified PE to address on the local device.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ubSize            [in] The size of temp Buffer on UB. (In Bytes)
 * @param copyParams        [in] Params to describe how non-contiguous data is managed in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ubSize, const non_contiguous_copy_param& copyParams, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(src, pe);
    if (ptr == nullptr) return;
    __gm__ T* remotePtr = reinterpret_cast<__gm__ T*>(ptr);

    AscendC::GlobalTensor<T> srcTensor;
    AscendC::LocalTensor<T> ubTensor;
    AscendC::GlobalTensor<T> dstTensor;
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    srcTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(remotePtr));
    dstTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint32_t ubStride = (copyParams.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams dataCopyParamsGM2UB(
        copyParams.repeat,
        copyParams.length * sizeof(T),
        (copyParams.srcLd - copyParams.length) * sizeof(T),
        (ubStride - copyParams.length) / ELE_NUM_PER_UNIT,
        0
    );
    smem_shm_copy_gm2ub(ubTensor, srcTensor, dataCopyParamsGM2UB);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams dataCopyParamsUB2GM(
        copyParams.repeat,
        copyParams.length * sizeof(T),
        (ubStride - copyParams.length) / ELE_NUM_PER_UNIT,
        (copyParams.dstLd - copyParams.length) * sizeof(T),
        0
    );
    smem_shm_copy_ub2gm(dstTensor, ubTensor, dataCopyParamsUB2GM);
}


/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param elemSize          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
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
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data 
 *        on symmetric memory from the specified PE to address on the local device.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param copyParams        [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, const non_contiguous_copy_param& copyParams, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);
    if (ptr == nullptr) return;

    AscendC::GlobalTensor<T> remoteBuff;
    remoteBuff.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint32_t ubStride = (copyParams.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams dataCopyParamsGM2UB(
        copyParams.repeat,
        copyParams.length * sizeof(T),
        (copyParams.srcLd - copyParams.length) * sizeof(T),
        (ubStride - copyParams.length) / ELE_NUM_PER_UNIT,
        0
    );
    smem_shm_copy_gm2ub(buf, remoteBuff, dataCopyParamsGM2UB);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams dataCopyParamsUB2GM(
        copyParams.repeat,
        copyParams.length * sizeof(T),
        (ubStride - copyParams.length) / ELE_NUM_PER_UNIT,
        (copyParams.dstLd - copyParams.length) * sizeof(T),
        0
    );
    smem_shm_copy_ub2gm(dst, buf, dataCopyParamsUB2GM);
}


/**
 * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ubSize            [in] The size of temp Buffer on UB. (In Bytes)
 * @param elemSize          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
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
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data 
 *        on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ubSize            [in] The size of temp Buffer on UB. (In Bytes)
 * @param copyParams        [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ubSize, const non_contiguous_copy_param& copyParams, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(dst, pe);
    if (ptr == nullptr) return;
    __gm__ T* remotePtr = reinterpret_cast<__gm__ T*>(ptr);

    AscendC::GlobalTensor<T> srcTensor;
    AscendC::LocalTensor<T> ubTensor;
    AscendC::GlobalTensor<T> dstTensor;
    ubTensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ubTensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    srcTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src));
    dstTensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(remotePtr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint32_t ubStride = (copyParams.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams dataCopyParamsGM2UB(
        copyParams.repeat,
        copyParams.length * sizeof(T),
        (copyParams.srcLd - copyParams.length) * sizeof(T),
        (ubStride - copyParams.length) / ELE_NUM_PER_UNIT,
        0
    );
    smem_shm_copy_gm2ub(ubTensor, srcTensor, dataCopyParamsGM2UB);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams dataCopyParamsUB2GM(
        copyParams.repeat,
        copyParams.length * sizeof(T),
        (ubStride - copyParams.length) / ELE_NUM_PER_UNIT,
        (copyParams.dstLd - copyParams.length) * sizeof(T),
        0
    );
    smem_shm_copy_ub2gm(dstTensor, ubTensor, dataCopyParamsUB2GM);
}


/**
 * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param elemSize          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
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
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data 
 *        on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param copyParams        [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, const non_contiguous_copy_param& copyParams, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);
    if (ptr == nullptr) return;

    AscendC::GlobalTensor<T> remoteBuff;
    remoteBuff.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint32_t ubStride = (copyParams.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams dataCopyParamsGM2UB(
        copyParams.repeat,
        copyParams.length * sizeof(T),
        (copyParams.srcLd - copyParams.length) * sizeof(T),
        (ubStride - copyParams.length) / ELE_NUM_PER_UNIT,
        0
    );
    smem_shm_copy_gm2ub(buf, src, dataCopyParamsGM2UB);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams dataCopyParamsUB2GM(
        copyParams.repeat,
        copyParams.length * sizeof(T),
        (ubStride - copyParams.length) / ELE_NUM_PER_UNIT,
        (copyParams.dstLd - copyParams.length) * sizeof(T),
        0
    );
    smem_shm_copy_ub2gm(remoteBuff, buf, dataCopyParamsUB2GM);
}


/**
 * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local UB of the source data.
 * @param elemSize          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(__gm__ T* dst, __ubuf__ T* src, uint32_t elemSize, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(dst, pe);
    if (ptr == nullptr) return;
    __gm__ T* remotePtr = reinterpret_cast<__gm__ T*>(ptr);

    smem_shm_copy_ub2gm(remotePtr, src, elemSize * sizeof(T));
}


/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local device.
 *
 * @param dst               [in] Pointer on local UB of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param elemSize          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(__ubuf__ T* dst, __gm__ T* src, uint32_t elemSize, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(src, pe);
    if (ptr == nullptr) return;
    __gm__ T* remotePtr = reinterpret_cast<__gm__ T*>(ptr);

    smem_shm_copy_gm2ub(dst, remotePtr, elemSize * sizeof(T));
}


#endif