/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_DEVICE_RMA_H
#define SHMEM_DEVICE_RMA_H

#include "kernel_operator.h"
#include "internal/device/shmemi_device_common.h"
#include "low_level/shmem_device_low_level_rma.h"
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
        __gm__ TYPE* addr_gm = reinterpret_cast<__gm__ TYPE*>(ptr);                         \
                                                                                            \
        *addr_gm = value;                                                                   \
        dcci_cacheline((__gm__ uint8_t *)addr_gm);                                          \
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
        __gm__ TYPE* addr_gm = reinterpret_cast<__gm__ TYPE*>(ptr);                         \
                                                                                            \
        dcci_cacheline((__gm__ uint8_t *)addr_gm);                                          \
        return *addr_gm;                                                                    \
    }

SHMEM_TYPE_FUNC(SHMEM_TYPENAME_G_AICORE);


#define SHMEM_GET_TYPENAME_MEM(NAME, TYPE)                                                                                      \
    /**                                                                                                                         \
    * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE. \
    *                                                                                                                           \
    * @param dst               [in] Pointer on local device of the destination data.                                            \
    * @param src               [in] Pointer on Symmetric memory of the source data.                                             \
    * @param elem_size         [in] Number of elements in the dest and source arrays.                                           \
    * @param pe                [in] PE number of the remote PE.                                                                 \
    */                                                                                                                          \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, uint32_t elem_size, int32_t pe)            \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                   \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                               \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_get_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE*>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id); \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM);


#define SHMEM_GET_TYPENAME_MEM_DETAILED(NAME, TYPE)                                                                             \
    /**                                                                                                                         \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                                \
     *        on symmetric memory from the specified PE to address on the local device.                                         \
     *                                                                                                                          \
     * @param dst               [in] Pointer on local device of the destination data.                                           \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                            \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.                      \
     * @param pe                [in] PE number of the remote PE.                                                                \
     */                                                                                                                         \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, const non_contiguous_copy_param& copy_params, int32_t pe)         \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                   \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                               \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_get_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE*>(copy_ub), copy_ub_size, copy_params, pe, copy_event_id);   \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_DETAILED);


#define SHMEM_GET_TYPENAME_MEM_TENSOR(NAME, TYPE)                                                                               \
    /**                                                                                                                         \
    * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local PE. \
    *                                                                                                                           \
    * @param dst               [in] GlobalTensor on local device of the destination data.                                       \
    * @param src               [in] GlobalTensor on Symmetric memory of the source data.                                        \
    * @param elem_size         [in] Number of elements in the dest and source arrays.                                           \
    * @param pe                [in] PE number of the remote PE.                                                                 \
    */                                                                                                                          \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, uint32_t elem_size, int pe)   \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                   \
        /* Create LocalTensor */                                                                                                \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                                   \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                          \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                                    \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                                          \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_get_mem_nbi(dst, src, ub_tensor, elem_size, pe, copy_event_id);                                               \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_TENSOR);


#define SHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED(NAME, TYPE)                                                                      \
    /**                                                                                                                         \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                                \
     *        on symmetric memory from the specified PE to address on the local device.                                         \
     *                                                                                                                          \
     * @param dst               [in] GlobalTensor on local device of the destination data.                                      \
     * @param src               [in] GlobalTensor on Symmetric memory of the source data.                                       \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.                      \
     * @param pe                [in] PE number of the remote PE.                                                                \
     */                                                                                                                         \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, const non_contiguous_copy_param& copy_params, int pe)  \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                   \
        /* Create LocalTensor */                                                                                                \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                                   \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                          \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                                    \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                                          \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_get_mem_nbi(dst, src, ub_tensor, copy_params, pe, copy_event_id);                                             \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_TENSOR_DETAILED);


#define SHMEM_PUT_TYPENAME_MEM(NAME, TYPE)                                                                                      \
    /**                                                                                                                         \
    * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.               \
    *                                                                                                                           \
    * @param dst               [in] Pointer on Symmetric memory of the destination data.                                        \
    * @param src               [in] Pointer on local device of the source data.                                                 \
    * @param elem_size         [in] Number of elements in the destination and source arrays.                                    \
    * @param pe                [in] PE number of the remote PE.                                                                 \
    */                                                                                                                          \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, uint32_t elem_size, int32_t pe)            \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                   \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                               \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_put_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE*>(copy_ub), copy_ub_size, elem_size, pe, copy_event_id); \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM);


#define SHMEM_PUT_TYPENAME_MEM_DETAILED(NAME, TYPE)                                                                             \
    /**                                                                                                                         \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                                \
     *        on local PE to symmetric address on the specified PE.                                                             \
     *                                                                                                                          \
     * @param dst               [in] Pointer on Symmetric memory of the destination data.                                       \
     * @param src               [in] Pointer on local device of the source data.                                                \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.                      \
     * @param pe                [in] PE number of the remote PE.                                                                \
     */                                                                                                                         \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, const non_contiguous_copy_param& copy_params, int32_t pe)        \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                   \
        uint32_t copy_ub_size = device_state->mte_config.ub_size;                                                               \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_put_mem_nbi(dst, src, reinterpret_cast<__ubuf__ TYPE*>(copy_ub), copy_ub_size, copy_params, pe, copy_event_id);   \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_DETAILED);


#define SHMEM_PUT_TYPENAME_MEM_TENSOR(NAME, TYPE)                                                                               \
    /**                                                                                                                         \
    * @brief Asynchronous interface. Copy a contiguous data on local PE to symmetric address on the specified PE.               \
    *                                                                                                                           \
    * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                                   \
    * @param src               [in] GlobalTensor on local device of the source data.                                            \
    * @param elem_size         [in] Number of elements in the destination and source arrays.                                    \
    * @param pe                [in] PE number of the remote PE.                                                                 \
    */                                                                                                                          \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, uint32_t elem_size, int pe)   \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                   \
        /* Create LocalTensor */                                                                                                \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                                   \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                          \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                                    \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                                          \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_put_mem_nbi(dst, src, ub_tensor, elem_size, pe, copy_event_id);                                               \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_TENSOR);


#define SHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED(NAME, TYPE)                                                                      \
    /**                                                                                                                         \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                                \
     *        on local PE to symmetric address on the specified PE.                                                             \
     *                                                                                                                          \
     * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                                  \
     * @param src               [in] GlobalTensor on local device of the source data.                                           \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.                      \
     * @param pe                [in] PE number of the remote PE.                                                                \
     */                                                                                                                         \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, const non_contiguous_copy_param& copy_params, int pe)  \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        /* CopyUB Config Set */                                                                                                 \
        uint64_t copy_ub = device_state->mte_config.shmem_ub;                                                                   \
        /* Create LocalTensor */                                                                                                \
        AscendC::LocalTensor<TYPE> ub_tensor;                                                                                   \
        ub_tensor.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);                                          \
        ub_tensor.address_.bufferAddr = reinterpret_cast<uint64_t>(copy_ub);                                                    \
        ub_tensor.address_.dataLen = device_state->mte_config.ub_size;                                                          \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_put_mem_nbi(dst, src, ub_tensor, copy_params, pe, copy_event_id);                                             \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_TENSOR_DETAILED);


#define SHMEM_GET_TYPENAME_MEM_UB(NAME, TYPE)                                                                                   \
    /**                                                                                                                         \
     * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local UB.\
     *                                                                                                                          \
     * @param dst               [in] Pointer on local UB of the destination data.                                               \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                            \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                                   \
     * @param pe                [in] PE number of the remote PE.                                                                \
     */                                                                                                                         \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__ubuf__ TYPE* dst, __gm__ TYPE* src, uint32_t elem_size, int pe)              \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_get_mem_nbi(dst, src, elem_size, pe, copy_event_id);                                                          \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_UB);


#define SHMEM_GET_TYPENAME_MEM_UB_TENSOR(NAME, TYPE)                                                                            \
    /**                                                                                                                         \
     * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified PE to address on the local UB.\
     *                                                                                                                          \
     * @param dst               [in] LocalTensor on local UB of the destination data.                                           \
     * @param src               [in] GlobalTensor on Symmetric memory of the source data.                                       \
     * @param elem_size         [in] Number of elements in the destination and source arrays.                                   \
     * @param pe                [in] PE number of the remote PE.                                                                \
     */                                                                                                                         \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(AscendC::LocalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, uint32_t elem_size, int pe)    \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_get_mem_nbi(dst, src, elem_size, pe, copy_event_id);                                                          \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_UB_TENSOR);


#define SHMEM_GET_TYPENAME_MEM_UB_DETAILED(NAME, TYPE)                                                                          \
    /**                                                                                                                         \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                                \
     *        on symmetric memory from the specified PE to address on the local UB.                                             \
     *                                                                                                                          \
     * @param dst               [in] Pointer on local UB of the destination data.                                               \
     * @param src               [in] Pointer on Symmetric memory of the source data.                                            \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.                      \
     * @param pe                [in] PE number of the remote PE.                                                                \
     */                                                                                                                         \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__ubuf__ TYPE* dst, __gm__ TYPE* src, const non_contiguous_copy_param& copy_params, int pe) \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_get_mem_nbi(dst, src, copy_params, pe, copy_event_id);                                                        \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_UB_DETAILED);


#define SHMEM_GET_TYPENAME_MEM_UB_TENSOR_DETAILED(NAME, TYPE)                                                                   \
    /**                                                                                                                         \
     * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                                \
     *        on symmetric memory from the specified PE to address on the local UB.                                             \
     *                                                                                                                          \
     * @param dst               [in] LocalTensor on local UB of the destination data.                                           \
     * @param src               [in] GlobalTensor on Symmetric memory of the source data.                                       \
     * @param copy_params       [in] Params to describe how non-contiguous data is managed in src and dst.                      \
     * @param pe                [in] PE number of the remote PE.                                                                \
     */                                                                                                                         \
    SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(AscendC::LocalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, const non_contiguous_copy_param& copy_params, int pe)  \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_get_mem_nbi(dst, src, copy_params, pe, copy_event_id);                                                        \
    }

SHMEM_TYPE_FUNC(SHMEM_GET_TYPENAME_MEM_UB_TENSOR_DETAILED);


#define SHMEM_PUT_TYPENAME_MEM_UB(NAME, TYPE)                                                                                   \
    /**                                                                                                                         \
    * @brief Asynchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.               \
    *                                                                                                                           \
    * @param dst               [in] Pointer on Symmetric memory of the destination data.                                        \
    * @param src               [in] Pointer on local UB of the source data.                                                     \
    * @param elem_size         [in] Number of elements in the destination and source arrays.                                    \
    * @param pe                [in] PE number of the remote PE.                                                                 \
    */                                                                                                                          \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(__gm__ TYPE* dst, __ubuf__ TYPE* src, uint32_t elem_size, int32_t pe)          \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_put_mem_nbi(dst, src, elem_size, pe, copy_event_id);                                                          \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_UB);


#define SHMEM_PUT_TYPENAME_MEM_UB_TENSOR(NAME, TYPE)                                                                            \
    /**                                                                                                                         \
    * @brief Asynchronous interface. Copy a contiguous data on local UB to symmetric address on the specified PE.               \
    *                                                                                                                           \
    * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                                   \
    * @param src               [in] LocalTensor on local UB of the source data.                                                 \
    * @param elem_size         [in] Number of elements in the destination and source arrays.                                    \
    * @param pe                [in] PE number of the remote PE.                                                                 \
    */                                                                                                                          \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src, uint32_t elem_size, int32_t pe)    \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_put_mem_nbi(dst, src, elem_size, pe, copy_event_id);                                                          \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_UB_TENSOR);


#define SHMEM_PUT_TYPENAME_MEM_UB_DETAILED(NAME, TYPE)                                                                          \
    /**                                                                                                                         \
    * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                                 \
    *        on local UB to symmetric address on the specified PE.                                                              \
    *                                                                                                                           \
    * @param dst               [in] Pointer on Symmetric memory of the destination data.                                        \
    * @param src               [in] Pointer on local UB of the source data.                                                     \
    * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.                     \
    * @param pe                [in] PE number of the remote PE.                                                                 \
    */                                                                                                                          \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(__gm__ TYPE* dst, __ubuf__ TYPE* src, const non_contiguous_copy_param& copy_params, int32_t pe) \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_put_mem_nbi(dst, src, copy_params, pe, copy_event_id);                                                        \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_UB_DETAILED);


#define SHMEM_PUT_TYPENAME_MEM_UB_TENSOR_DETAILED(NAME, TYPE)                                                                   \
    /**                                                                                                                         \
    * @brief Asynchronous interface. Provide a high-performance way to copy non-contiguous data                                 \
    *        on local UB to symmetric address on the specified PE.                                                              \
    *                                                                                                                           \
    * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.                                   \
    * @param src               [in] LocalTensor on local UB of the source data.                                                 \
    * @param copy_params       [in] Params to describe how non-contiguous data is organized in src and dst.                     \
    * @param pe                [in] PE number of the remote PE.                                                                 \
    */                                                                                                                          \
    SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::LocalTensor<TYPE> src, const non_contiguous_copy_param& copy_params, int32_t pe)  \
    {                                                                                                                           \
        /* ROCE */                                                                                                              \
        /* RDMA */                                                                                                              \
        /* MTE  */                                                                                                              \
        /* Global State Get */                                                                                                  \
        __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();                                                   \
        AscendC::TEventID copy_event_id = (AscendC::TEventID)device_state->mte_config.event_id;                                 \
        shmem_mte_put_mem_nbi(dst, src, copy_params, pe, copy_event_id);                                                        \
    }

SHMEM_TYPE_FUNC(SHMEM_PUT_TYPENAME_MEM_UB_TENSOR_DETAILED);


#endif
