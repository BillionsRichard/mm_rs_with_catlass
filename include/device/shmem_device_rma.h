#ifndef SHMEM_DEVICE_RMA_H
#define SHMEM_DEVICE_RMA_H

#include "kernel_operator.h"
#include "internal/device/shmemi_device_common.h"
#include "shmem_device_team.h"

/**
 * @brief Standard RMA Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |half       | half      |
 * |float      | float     |
 * |double     | double    |
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 * |int64      | int64     |
 * |uint8      | uint8     |
 * |uint16     | uint16    |
 * |uint32     | uint32    |
 * |uint64     | uint64    |
 * |char       | char      |
 * |bfloat16   | bfloat16  |
*/
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
    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();

    // Check whether ptr belongs to this rank.
    uint64_t lower_bound = (uint64_t)device_state->p2p_heap_base[shmem_my_pe()];
    uint64_t upper_bound = lower_bound + device_state->heap_size;
    if (uint64_t(ptr) < lower_bound || uint64_t(ptr) >= upper_bound) {
        return nullptr;
    }

    // Back to root address
    uint64_t offset = reinterpret_cast<uint64_t>(ptr) - reinterpret_cast<uint64_t>(device_state->heap_base);
    
    // Address translate
    uint64_t remote_ptr = reinterpret_cast<uint64_t>(device_state->p2p_heap_base[pe]) + offset;

    return reinterpret_cast<__gm__ void*>(remote_ptr);
}


#define SHMEM_TYPENAME_P_AICORE(NAME, TYPE)                                                 \
    /**                                                                                     \
    * @brief Provide a low latency put capability for single element of most basic types.   \
    *                                                                                       \
    * @param dst               [in] Symmetric address of the destination data on local PE.  \
    * @param value             [in] The element to be put.                                  \
    * @param pe                [in] The number of the remote PE.                            \
    */                                                                                      \
    SHMEM_DEVICE void shmem_##NAME##_p(__gm__ TYPE* dst, const TYPE value, int pe)          \
    {                                                                                       \
        auto ptr = shmem_ptr(dst, pe);                                                      \
        if (ptr == nullptr) return;                                                         \
        __gm__ TYPE* addr_gm = reinterpret_cast<__gm__ TYPE*>(ptr);                          \
                                                                                            \
        *addr_gm = value;                                                                    \
        dcci_cacheline((__gm__ uint8_t *)addr_gm);                                            \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPENAME_P_AICORE);


#define SHMEM_TYPENAME_G_AICORE(NAME, TYPE)                                                 \
    /**                                                                                     \
    * @brief Provide a low latency get capability for single element of most basic types.   \
    *                                                                                       \
    * @param src               [in] Symmetric address of the destination data on local PE.  \
    * @param pe                [in] The number of the remote PE.                            \
    * @return A single element of type specified in the input pointer.                      \
    */                                                                                      \
    SHMEM_DEVICE TYPE shmem_##NAME##_g(__gm__ TYPE* src, int32_t pe)                        \
    {                                                                                       \
        auto ptr = shmem_ptr(src, pe);                                                      \
        __gm__ TYPE* addr_gm = reinterpret_cast<__gm__ TYPE*>(ptr);                          \
                                                                                            \
        dcci_cacheline((__gm__ uint8_t *)addr_gm);                                            \
        return *addr_gm;                                                                     \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPENAME_G_AICORE);


/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local device.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size            [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(src, pe);
    if (ptr == nullptr) return;
    __gm__ T* remote_ptr = reinterpret_cast<__gm__ T*>(ptr);

    // block_size: dataMove Unit
    uint32_t block_size = ub_size / sizeof(T) * sizeof(T);
    uint32_t remain = (elem_size * sizeof(T)) % block_size;

    int repeat_times = (elem_size * sizeof(T)) / block_size;
    int repeat_elem = block_size / sizeof(T);
    int loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (int i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, remote_ptr + i * repeat_elem, block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(dst + i * repeat_elem, buf, block_size);
        if (i != loop_times - 1) {      // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        smem_shm_copy_gm2ub(buf, remote_ptr + repeat_times * repeat_elem, remain);
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
 * @param ub_size            [in] The size of temp Buffer on UB. (In Bytes)
 * @param copy_params        [in] Params to describe how non-contiguous data is managed in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, const non_contiguous_copy_param& copy_params, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(src, pe);
    if (ptr == nullptr) return;
    __gm__ T* remote_ptr = reinterpret_cast<__gm__ T*>(ptr);

    AscendC::GlobalTensor<T> src_tensor;
    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> dst_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(remote_ptr));
    dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dst));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint32_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(
        copy_params.repeat,
        copy_params.length * sizeof(T),
        (copy_params.src_ld - copy_params.length) * sizeof(T),
        (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
        0
    );
    smem_shm_copy_gm2ub(ub_tensor, src_tensor, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(
        copy_params.repeat,
        copy_params.length * sizeof(T),
        (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
        (copy_params.dst_ld - copy_params.length) * sizeof(T),
        0
    );
    smem_shm_copy_ub2gm(dst_tensor, ub_tensor, data_copy_params_ub2gm);
}


/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param elem_size          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);
    if (ptr == nullptr) return;

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ptr));

    // block_size: dataMove Unit
    uint32_t block_size = buf.GetSize() / sizeof(T) * sizeof(T);
    uint32_t remain = (elem_size * sizeof(T)) % block_size;

    int repeat_times = (elem_size * sizeof(T)) / block_size;
    int repeat_elem = block_size / sizeof(T);
    int loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (int i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, remote_buff[i * repeat_elem], block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(dst[i * repeat_elem], buf, block_size);
        if (i != loop_times - 1) {      // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        smem_shm_copy_gm2ub(buf, remote_buff[repeat_times * repeat_elem], remain);
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
 * @param copy_params        [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_get_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, const non_contiguous_copy_param& copy_params, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)src.GetPhyAddr(), pe);
    if (ptr == nullptr) return;

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint32_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(
        copy_params.repeat,
        copy_params.length * sizeof(T),
        (copy_params.src_ld - copy_params.length) * sizeof(T),
        (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
        0
    );
    smem_shm_copy_gm2ub(buf, remote_buff, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(
        copy_params.repeat,
        copy_params.length * sizeof(T),
        (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
        (copy_params.dst_ld - copy_params.length) * sizeof(T),
        0
    );
    smem_shm_copy_ub2gm(dst, buf, data_copy_params_ub2gm);
}


/**
 * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size            [in] The size of temp Buffer on UB. (In Bytes)
 * @param elem_size          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(dst, pe);
    if (ptr == nullptr) return;
    __gm__ T* remote_ptr = reinterpret_cast<__gm__ T*>(ptr);

    // block_size: dataMove Unit
    uint32_t block_size = ub_size / sizeof(T) * sizeof(T);
    uint32_t remain = (elem_size * sizeof(T)) % block_size;

    int repeat_times = (elem_size * sizeof(T)) / block_size;
    int repeat_elem = block_size / sizeof(T);
    int loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (int i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, src + i * repeat_elem, block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remote_ptr + i * repeat_elem, buf, block_size);
        if (i != loop_times - 1) {      // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        smem_shm_copy_gm2ub(buf, src + repeat_times * repeat_elem, remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remote_ptr + repeat_times * repeat_elem, buf, remain);
    }
}


/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data 
 *        on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param ub_size            [in] The size of temp Buffer on UB. (In Bytes)
 * @param copy_params        [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t ub_size, const non_contiguous_copy_param& copy_params, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr(dst, pe);
    if (ptr == nullptr) return;
    __gm__ T* remote_ptr = reinterpret_cast<__gm__ T*>(ptr);

    AscendC::GlobalTensor<T> src_tensor;
    AscendC::LocalTensor<T> ub_tensor;
    AscendC::GlobalTensor<T> dst_tensor;
    ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(buf);
    src_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(src));
    dst_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(remote_ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint32_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(
        copy_params.repeat,
        copy_params.length * sizeof(T),
        (copy_params.src_ld - copy_params.length) * sizeof(T),
        (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
        0
    );
    smem_shm_copy_gm2ub(ub_tensor, src_tensor, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(
        copy_params.repeat,
        copy_params.length * sizeof(T),
        (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
        (copy_params.dst_ld - copy_params.length) * sizeof(T),
        0
    );
    smem_shm_copy_ub2gm(dst_tensor, ub_tensor, data_copy_params_ub2gm);
}


/**
 * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] Pointer on local UB.
 * @param elem_size          [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, uint32_t elem_size, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);
    if (ptr == nullptr) return;

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ptr));

    // block_size: dataMove Unit
    uint32_t block_size = buf.GetSize() / sizeof(T) * sizeof(T);
    uint32_t remain = (elem_size * sizeof(T)) % block_size;

    int repeat_times = (elem_size * sizeof(T)) / block_size;
    int repeat_elem = block_size / sizeof(T);
    int loop_times = remain > 0 ? repeat_times + 1 : repeat_times;
    for (int i = 0; i < repeat_times; i++) {
        smem_shm_copy_gm2ub(buf, src[i * repeat_elem], block_size);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remote_buff[i * repeat_elem], buf, block_size);
        if (i != loop_times - 1) {      // Last PIPE Sync Should be done outside
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
    }
    if (remain > 0) {
        smem_shm_copy_gm2ub(buf, src[repeat_times * repeat_elem], remain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        smem_shm_copy_ub2gm(remote_buff[repeat_times * repeat_elem], buf, remain);
    }
}


/**
 * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data 
 *        on local PE to symmetric address on the specified PE.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] LocalTensor on local UB.
 * @param copy_params        [in] Params to describe how non-contiguous data is organized in src and dst.
 * @param pe                [in] PE number of the remote PE.
 * @param EVENT_ID          [in] ID used to Sync MTE2\\MTE3 Event.
 */
template <typename T>
SHMEM_DEVICE void shmem_mte_put_mem_nbi(AscendC::GlobalTensor<T> dst, AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T> buf, const non_contiguous_copy_param& copy_params, int pe, AscendC::TEventID EVENT_ID)
{
    auto ptr = shmem_ptr((__gm__ void *)dst.GetPhyAddr(), pe);
    if (ptr == nullptr) return;

    AscendC::GlobalTensor<T> remote_buff;
    remote_buff.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(ptr));

    uint32_t ELE_NUM_PER_UNIT = 32 / sizeof(T);
    uint32_t ub_stride = (copy_params.length + ELE_NUM_PER_UNIT - 1) / ELE_NUM_PER_UNIT * ELE_NUM_PER_UNIT;
    AscendC::DataCopyExtParams data_copy_params_gm2ub(
        copy_params.repeat,
        copy_params.length * sizeof(T),
        (copy_params.src_ld - copy_params.length) * sizeof(T),
        (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
        0
    );
    smem_shm_copy_gm2ub(buf, src, data_copy_params_gm2ub);

    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);

    AscendC::DataCopyExtParams data_copy_params_ub2gm(
        copy_params.repeat,
        copy_params.length * sizeof(T),
        (ub_stride - copy_params.length) / ELE_NUM_PER_UNIT,
        (copy_params.dst_ld - copy_params.length) * sizeof(T),
        0
    );
    smem_shm_copy_ub2gm(remote_buff, buf, data_copy_params_ub2gm);
}


#define SHMEM_GET_TYPENAME_MEM(NAME, TYPE)                                                                                      \
    /**                                                                                                                         \
    * @fn SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, uint32_t elem_size, int32_t pe)       \
    * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE. \
    *                                                                                                                           \
    * @param dst               [in] Pointer on local device of the destination data.                                            \
    * @param src               [in] Pointer on Symmetric memory of the source data.                                             \
    * @param elem_size          [in] Number of elements in the dest and source arrays.                                           \
    * @param pe                [in] PE number of the remote PE.                                                                 \
    */                                                                                                                          \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, uint32_t elem_size, int32_t pe)             \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                           \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                       \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                                    \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                      \
        shmem_mte_get_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE*>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);       \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM);


#define SHMEM_GET_TYPENAME_MEM_DETAILED(NAME, TYPE)                                                                             \
    /**                                                                                                                         \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                                \
     *        on symmetric memory from the specified PE to address on the local device.                                         \
     *                                                                                                                          \
     * @param dst               [in] Pointer on local device of the destination data.                                           \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                            \
     * @param copy_params        [in] Params to describe how non-contiguous data is managed in src and dst.                      \
     * @param pe                [in] PE number of the remote PE.                                                                \
     */                                                                                                                         \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, const non_contiguous_copy_param& copy_params, int32_t pe)         \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                           \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                       \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                                    \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                      \
        shmem_mte_get_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE*>(copy_ub), copy_ub_size, copy_params, pe, copy_event_id);     \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_DETAILED);


#define SHMEM_GET_TYPENAME_MEM_TENSOR(NAME, TYPE)                                                                               \
    /**                                                                                                                         \
    * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE. \
    *                                                                                                                           \
    * @param dst               [in] GlobalTensor on local device of the destination data.                                       \
    * @param src               [in] GlobalTensor on Symmetric memory of the source data.                                        \
    * @param elem_size          [in] Number of elements in the dest and source arrays.                                           \
    * @param pe                [in] PE number of the remote PE.                                                                 \
    */                                                                                                                          \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, uint32_t elem_size, int pe)   \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                           \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                       \
        /* Create LocalTensor */                                                                                                \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                                    \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                           \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                                      \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                                              \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                      \
        shmem_mte_get_mem_nbi(dst, src, ub_tensor, elem_size, pe, copy_event_id);                                                   \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_TENSOR);


#define SHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED(NAME, TYPE)                                                                      \
    /**                                                                                                                         \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                                \
     *        on symmetric memory from the specified PE to address on the local device.                                         \
     *                                                                                                                          \
     * @param dst               [in] GlobalTensor on local device of the destination data.                                      \
     * @param src               [in] GlobalTensor on Symmetric memory of the source data.                                       \
     * @param copy_params        [in] Params to describe how non-contiguous data is managed in src and dst.                      \
     * @param pe                [in] PE number of the remote PE.                                                                \
     */                                                                                                                         \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, const non_contiguous_copy_param& copy_params, int pe)  \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                           \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                       \
        /* Create LocalTensor */                                                                                                \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                                    \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                           \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                                      \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                                              \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                      \
        shmem_mte_get_mem_nbi(dst, src, ub_tensor, copy_params, pe, copy_event_id);                                                 \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED);


#define SHMEM_PUT_TYPENAME_MEM(NAME, TYPE)                                                                                      \
    /**                                                                                                                         \
    * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.               \
    *                                                                                                                           \
    * @param dst               [in] Pointer on Symmetric memory of the destination data.                                        \
    * @param src               [in] Pointer on local device of the source data.                                                 \
    * @param elem_size          [in] Number of elements in the destination and source arrays.                                    \
    * @param pe                [in] PE number of the remote PE.                                                                 \
    */                                                                                                                          \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, uint32_t elem_size, int32_t pe)             \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                           \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                       \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                                    \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                      \
        shmem_mte_put_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE*>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id);       \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM);


#define SHMEM_PUT_TYPENAME_MEM_DETAILED(NAME, TYPE)                                                                             \
    /**                                                                                                                         \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                                \
     *        on local PE to symmetric address on the specified PE.                                                             \
     *                                                                                                                          \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                                       \
     * @param src               [in] Pointer on local device of the source data.                                                \
     * @param copy_params        [in] Params to describe how non-contiguous data is managed in src and dst.                      \
     * @param pe                [in] PE number of the remote PE.                                                                \
     */                                                                                                                         \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, const non_contiguous_copy_param& copy_params, int32_t pe)        \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                           \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                       \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                                    \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                      \
        shmem_mte_put_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE*>(copy_ub), copy_ub_size, copy_params, pe, copy_event_id);     \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_DETAILED);


#define SHMEM_PUT_TYPENAME_MEM_TENSOR(NAME, TYPE)                                                                               \
    /**                                                                                                                         \
    * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.               \
    *                                                                                                                           \
    * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                                   \
    * @param src               [in] GlobalTensor on local device of the source data.                                            \
    * @param elem_size          [in] Number of elements in the destination and source arrays.                                    \
    * @param pe                [in] PE number of the remote PE.                                                                 \
    */                                                                                                                          \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, uint32_t elem_size, int pe)   \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                           \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                       \
        /* Create LocalTensor */                                                                                                \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                                    \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                           \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                                      \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                                              \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                      \
        shmem_mte_put_mem_nbi(dst, src, ub_tensor, elem_size, pe, copy_event_id);                                                   \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_TENSOR);



#define SHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED(NAME, TYPE)                                                                      \
    /**                                                                                                                         \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                                \
     *        on local PE to symmetric address on the specified PE.                                                             \
     *                                                                                                                          \
     * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                                  \
     * @param src               [in] GlobalTensor on local device of the source data.                                           \
     * @param copy_params        [in] Params to describe how non-contiguous data is managed in src and dst.                      \
     * @param pe                [in] PE number of the remote PE.                                                                \
     */                                                                                                                         \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, const non_contiguous_copy_param& copy_params, int pe)  \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                           \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                       \
        /* Create LocalTensor */                                                                                                \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                                    \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                           \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                                      \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                                              \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                      \
        shmem_mte_put_mem_nbi(dst, src, ub_tensor, copy_params, pe, copy_event_id);                                                 \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED);


#endif
