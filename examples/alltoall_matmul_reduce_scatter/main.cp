/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "utils.h"
#include "shmem.h"

// Kernel specific include
#include "catcoc/dgemm/kernel/alltoall_matmul_reduce_scatter.hpp"


void MallocMem(void** devPtr, uint64_t size) {
    if (aclrtMalloc(devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        std::cout << "Malloc device memory failed." << std::endl;
        exit(-1);
    }
}

void FreeMem(void* devPtr) {
    if (aclrtFree(devPtr) != ACL_SUCCESS) {
        std::cout << "Free device memory failed." << std::endl;
        exit(-1);
    }
}

// Forward declaration for the kernel launch function
void launch_kernel(uint32_t m, uint32_t n, uint32_t k, int rank_id, int rank_size,
                   void* d_a_local, void* d_b_local, void* d_d_local, void* d_workspace);

// Host-side verification function
bool verify_result(const aclFloat16* host_output, const aclFloat16* host_golden, size_t size) {
    bool pass = true;
    for (size_t i = 0; i < size; ++i) {
        float val_out = aclFloat16ToFloat(host_output[i]);
        float val_gold = aclFloat16ToFloat(host_golden[i]);
        float diff = std::abs(val_out - val_gold);
        float tolerance = 1e-3; // Relative tolerance
        float relative_diff = diff / (std::abs(val_gold) + tolerance);
        if (relative_diff > 0.01) { // 1% relative error
            std::cout << "Verification FAILED at index " << i << ". Output: " << val_out
                      << ", Golden: " << val_gold << ", Diff: " << diff << std::endl;
            pass = false;
            break;
        }
    }
    return pass;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " M N K" << std::endl;
        return -1;
    }

    uint32_t m = std::stoul(argv[1]);
    uint32_t n = std::stoul(argv[2]);
    uint32_t k = std::stoul(argv[3]);

    // 1. SHMEM and ACL initialization
    shmem_init();
    int rank_id = shmem_my_pe();
    int rank_size = shmem_n_pes();
    std::cout << "rank_id: " << rank_id << ", rank_size: " << rank_size << std::endl;

    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t device_id = rank_id;
    CHECK_ACL(aclrtSetDevice(device_id));
    CHECK_ACL(aclrtCreateContext(&context, device_id));

    // 2. Problem definition and memory allocation
    uint32_t m_local = m / rank_size;
    uint32_t k_local = k / rank_size;

    uint64_t size_a_local = (uint64_t)m_local * k;
    uint64_t size_b_local = (uint64_t)k_local * n;
    uint64_t size_d_local = (uint64_t)m_local * n;

    // Workspace for A' after alltoall: M * K_local * sizeof(Element)
    // Workspace for P_i after matmul: M * N * sizeof(ElementAccum)
    uint64_t workspace_size = (uint64_t)m * k * sizeof(aclFloat16) + (uint64_t)m * n * sizeof(float);

    void *d_a_local, *d_b_local, *d_d_local, *d_workspace;
    MallocMem(&d_a_local, size_a_local * sizeof(aclFloat16));
    MallocMem(&d_b_local, size_b_local * sizeof(aclFloat16));
    MallocMem(&d_d_local, size_d_local * sizeof(aclFloat16));
    
    // Allocate symmetric workspace
    d_workspace = shmem_malloc(workspace_size);
    if (d_workspace == nullptr) {
        std::cout << "Failed to allocate symmetric workspace." << std::endl;
        return -1;
    }

    // 3. Data generation
    std::vector<aclFloat16> h_a_global(m * k);
    std::vector<aclFloat16> h_b_global(k * n);
    std::vector<float> h_c_global_fp32(m * n);

    // Initialize global matrices on host
    for (size_t i = 0; i < m * k; ++i) h_a_global[i] = aclFloatToFloat16(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    for (size_t i = 0; i < k * n; ++i) h_b_global[i] = aclFloatToFloat16(static_cast<float>(rand()) / static_cast<float>(RAND_MAX));
    
    // Slice and copy to device
    std::vector<aclFloat16> h_a_local(size_a_local);
    // A is sliced on M dimension
    for(uint32_t i = 0; i < m_local; ++i) {
        for(uint32_t j = 0; j < k; ++j) {
            h_a_local[i * k + j] = h_a_global[(rank_id * m_local + i) * k + j];
        }
    }
    CHECK_ACL(aclrtMemcpy(d_a_local, size_a_local * sizeof(aclFloat16), h_a_local.data(), size_a_local * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));

    std::vector<aclFloat16> h_b_local(size_b_local);
    // B is sliced on K dimension
    for(uint32_t i = 0; i < k_local; ++i) {
        for(uint32_t j = 0; j < n; ++j) {
            h_b_local[i * n + j] = h_b_global[(rank_id * k_local + i) * n + j];
        }
    }
    CHECK_ACL(aclrtMemcpy(d_b_local, size_b_local * sizeof(aclFloat16), h_b_local.data(), size_b_local * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE));


    // 4. Golden Calculation (on host)
    // C_global(m,n) = A_global(m,k) * B_global(k,n)
    for (uint32_t mi = 0; mi < m; ++mi) {
        for (uint32_t ni = 0; ni < n; ++ni) {
            float acc = 0.0f;
            for (uint32_t ki = 0; ki < k; ++ki) {
                acc += aclFloat16ToFloat(h_a_global[mi * k + ki]) * aclFloat16ToFloat(h_b_global[ki * n + ni]);
            }
            h_c_global_fp32[mi * n + ni] = acc;
        }
    }

    // Slice golden C for local rank verification
    std::vector<aclFloat16> h_d_golden(size_d_local);
    for(uint32_t i = 0; i < m_local; ++i) {
        for(uint32_t j = 0; j < n; ++j) {
            h_d_golden[i * n + j] = aclFloatToFloat16(h_c_global_fp32[(rank_id * m_local + i) * n + j]);
        }
    }

    // 5. Kernel launch
    launch_kernel(m, n, k, rank_id, rank_size, d_a_local, d_b_local, d_d_local, d_workspace);

    shmem_barrier_all();

    // 6. Verification
    std::vector<aclFloat16> h_d_output(size_d_local);
    CHECK_ACL(aclrtMemcpy(h_d_output.data(), size_d_local * sizeof(aclFloat16), d_d_local, size_d_local * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST));

    bool pass = verify_result(h_d_output.data(), h_d_golden.data(), size_d_local);
    if (rank_id == 0) {
        std::cout << "Verification " << (pass ? "PASSED" : "FAILED") << std::endl;
    }

    // 7. Cleanup
    FreeMem(d_a_local);
    FreeMem(d_b_local);
    FreeMem(d_d_local);
    shmem_free(d_workspace);

    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclFinalize());
    shmem_finalize();

    return 0;
}


void launch_kernel(uint32_t m, uint32_t n, uint32_t k, int rank_id, int rank_size,
                   void* d_a_local, void* d_b_local, void* d_d_local, void* d_workspace)
{
    using Element = aclFloat16;
    using ElementAccum = float;

    // Define Catlass/Catcoc types
    using MmadL1TileShape = Catlass::Shape<128, 128, 32>;
    using ArchTag = Catlass::arch::Sm80; // Placeholder, should match target arch

    using BlockMmad = Catlass::gemm::block::BlockMmad<
        ArchTag, MmadL1TileShape,
        Catlass::gemm::MakeGemmType<Element, Catlass::layout::RowMajor, Element, Catlass::layout::RowMajor, ElementAccum, Catlass::layout::RowMajor>
    >;

    using TileRemoteCopyGet = Catcoc::CommEpilogue::Tile::TileRemoteCopy<
        ArchTag,
        Catlass::TypeHolder<Element, Catlass::layout::RowMajor>,
        Catlass::TypeHolder<Element, Catlass::layout::RowMajor>,
        Catcoc::detail::CopyDirect::Get
    >;

    using EpilogueComm = Catcoc::CommEpilogue::Block::CommBlockEpilogue<
        Catcoc::CommEpilogue::EpilogueAtlasA2CommRemoteCopy<2, Catcoc::detail::CopyMode::P2P, false>,
        Catlass::TypeHolder<Element, Catlass::layout::RowMajor>,
        Catlass::TypeHolder<Element, Catlass::layout::RowMajor>,
        Catlass::Shape<1, 1>, Catlass::Shape<64, 64>, Catlass::Shape<64, 64>,
        TileRemoteCopyGet
    >;

    using MmadScheduler = Catlass::gemm::BlockMmadSchedulerDefault<BlockMmad>;
    using CommScheduler = Catcoc::CommEpilogue::BlockSchedulerDefault<EpilogueComm>; // Assuming this exists

    using MyKernel = Catcoc::DGemm::Kernel::AlltoallMatmulReduceScatter<
        BlockMmad, EpilogueComm, EpilogueComm, MmadScheduler, CommScheduler, 2
    >;

    // Setup kernel parameters
    uint32_t m_local = m / rank_size;
    uint32_t k_local = k / rank_size;

    Catlass::layout::RowMajor layout_a = {m_local, k, k};
    Catlass::layout::RowMajor layout_b = {k_local, n, n};
    Catlass::layout::RowMajor layout_d = {m_local, n, n};

    typename MyKernel::Params params(
        {m, n, k},
        rank_id,
        rank_size,
        SHMEM_TEAM_WORLD,
        d_a_local, layout_a,
        d_b_local, layout_b,
        d_d_local, layout_d,
        d_workspace,
        {},{} // Default epilogue params
    );

    // Launch kernel
    aclrtStream stream;
    CHECK_ACL(aclrtCreateStream(&stream));
    
    // Define grid and block dimensions
    dim3 grid(1, 1, 1); // Grid for AIC and AIV
    dim3 block(cce::aicore::GetAicCoreNum(), 1, 1);

    MyKernel<<<grid, block, stream>>>(params);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtDestroyStream(stream));

    if (rank_id == 0) {
        std::cout << "Kernel launched and synchronized." << std::endl;
    }
}

