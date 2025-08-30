/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>

#include <iostream>
#include <vector>
#include <cstring>

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

// shmem_host
#include "shmem_api.h"

// utils
#include "utils.h"

#include "catcoc/catcoc.hpp"
#include "catcoc/comm_epilogue/comm_dispatch_policy.hpp"
#include "catcoc/comm_epilogue/block/comm_block_epilogue.hpp"
#include "catcoc/comm_epilogue/block/comm_block_swizzle.hpp"
#include "catcoc/comm_epilogue/tile/tile_remote_copy.hpp"
#include "catcoc/detail/remote_copy_type.hpp"
#include "catcoc/dgemm/kernel/matmul_reduce_scatter_quant_perchn.hpp"

static uint32_t gNpuNum = 8;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

using namespace AscendC;
using namespace Catcoc;

constexpr uint32_t BLOCK_NUM = 20;
constexpr int32_t BLOCK_SIZE_16 = 16;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;

CATLASS_GLOBAL
void ShmemMatmulReduceScatterQuantPerchn(
    uint64_t fftsAddr,
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD,
    GM_ADDR bias, GM_ADDR scale, GM_ADDR symmetricPtr,
    uint32_t m, uint32_t n, uint32_t k, shmem_team_t teamIdx = 0
)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    // Define ArchTag
    using ArchTag = Catlass::Arch::AtlasA2;

    Catlass::GemmCoord problemShape{m, n, k};

    // Prepare comm address
    uint32_t rankIdx = shmem_team_my_pe(teamIdx);
    uint32_t rankSize = shmem_team_n_pes(teamIdx);
    using ElementC = half;

    // Block level, Define the layout of each input matrix
    Catlass::layout::RowMajor layoutA{m, k, k};
    Catlass::layout::RowMajor layoutB{k, n, n};
    Catlass::layout::RowMajor layoutD{m / rankSize, n, n};
    Catlass::layout::VectorLayout layoutBias{n};

    // Block level, define BlockMmad
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2PingpongBias<enableUnitFlag>;
    using L1TileShape = Catlass::GemmShape<128, 256, 256>;
    using L0TileShape = Catlass::GemmShape<128, 256, 64>;
    using AType = Catlass::Gemm::GemmType<int8_t, LayoutA>;
    using BType = Catlass::Gemm::GemmType<int8_t, LayoutB>;
    using CType = Catlass::Gemm::GemmType<half, LayoutC>;
    using DType = Catlass::Gemm::GemmType<half, LayoutD>;
    using BiasType = Catlass::Gemm::GemmType<int32_t, Catlass::layout::VectorLayout>;
    using ScaleType = Catlass::Gemm::GemmType<uint64_t , Catlass::layout::VectorLayout>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmadQuantPerchn<MmadDispatchPolicy,
        L1TileShape, L0TileShape, AType, BType, CType, BiasType, ScaleType>;

    using BlockMmadScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
    using BlockEpilogueScheduler = CommEpilogue::Block::BlockCommSwizzle<0, true>;

    using RemoteSrcType = CType;
    using RemoteDstType = DType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Get>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using CommBlockShape = Catlass::MatrixShape<64, 256>;
    using CommCoreSplit = Catlass::MatrixShape<20, 1>;

    constexpr uint32_t UB_STAGES = 2;
    using EpilogueReduceScatterTileShape = Catlass::MatrixShape<32, 256>;
    using EpilogueReduceScatterDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES,
            Catcoc::detail::CopyMode::Scatter>;
    using BlockEpilogueReduceScatter = CommEpilogue::Block::CommBlockEpilogue<
        EpilogueReduceScatterDispatch,
        RemoteSrcType, RemoteDstType,
        CommCoreSplit,
        CommBlockShape,
        EpilogueReduceScatterTileShape, TileRemoteCopy, TileScheduler
    >;

    constexpr uint32_t workspaceStages = 2;
    constexpr uint32_t commInterval = 10;
    using MatmulReduceScatterKernel = DGemm::Kernel::MatmulReduceScatterQuantPerchn<
        BlockMmad,
        BlockEpilogueReduceScatter,
        BlockMmadScheduler,
        BlockEpilogueScheduler,
        workspaceStages
    >;
    Catlass::GemmCoord problemShapeInRank = problemShape / Catlass::MakeCoord<uint32_t>(rankSize, 1, 1);
    BlockMmadScheduler mmadBlockScheduler(problemShapeInRank, Catlass::MakeCoord(L1TileShape::M, L1TileShape::N));

    typename BlockEpilogueReduceScatter::Params reduceScatterParams{};

    // Prepare params
    typename MatmulReduceScatterKernel::Params params{
        problemShape,
        rankIdx, rankSize, teamIdx,
        gmA, layoutA,
        gmB, layoutB,
        bias, scale,
        symmetricPtr,
        reduceScatterParams,
        gmD, layoutD,
        commInterval
    };

    // Call kernel
    MatmulReduceScatterKernel matmulCommKernel;
    matmulCommKernel(params);

    shmem_barrier_all();
}

struct Options {
    static constexpr auto helper = "Usage: matmul_allreduce m n k transA transB\n";

    int rankSize;
    int rankId;
    std::string ipPort{};
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::vector<int> deviceIdList{};

    int Parse(int argc, char **argv)
    {
        enum ArgsIndex {
            RANK_SIZE_INDEX = 1,
            RANK_ID_INDEX,
            IP_PORT_INDEX,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            DEVICE_LIST_INDEX,
            INDEX_MAX
        };

        if (argc > INDEX_MAX) {
            printf(helper);
            return -1;
        }

        rankSize = std::atoi(argv[RANK_SIZE_INDEX]);
        rankId = std::atoi(argv[RANK_ID_INDEX]);
        ipPort = argv[IP_PORT_INDEX];
        m = std::atoi(argv[M_INDEX]);
        n = std::atoi(argv[N_INDEX]);
        k = std::atoi(argv[K_INDEX]);
        if (argc > DEVICE_LIST_INDEX) {
            char *idListStr = argv[DEVICE_LIST_INDEX];
            for (char *idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) {
                deviceIdList.push_back(std::atoi(idToken));
            }
        } else {
            for (int i = 0; i < rankSize; ++i) {
                deviceIdList.push_back(i);
            }
        }
        return 0;
    }
};


int main(int argc, char **argv)
{
    int status = SHMEM_SUCCESS;
    Options options;
    options.Parse(argc, argv);
    int rankSize = options.rankSize;
    int rankId = options.rankId;
    std::string ipPort = options.ipPort;
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t k = options.k;
    int32_t deviceId = options.deviceIdList[rankId];

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipPort << "\n";

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));
    status = shmem_set_conf_store_tls(false, nullptr, 0);
    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rankId, rankSize, gNpuMallocSpace, ipPort.c_str(), &attributes);
    status = shmem_init_attr(attributes);
    status = shmem_init_status();

    size_t aSize = static_cast<size_t>(m) * k * sizeof(int8_t);
    size_t bSize = static_cast<size_t>(k) * n * sizeof(int8_t);
    size_t cSize = static_cast<size_t>(m) * n * sizeof(half);
    size_t biasSize = static_cast<size_t>(n) * sizeof(int32_t);
    size_t scaleSize = static_cast<size_t>(n) * sizeof(uint64_t);
    size_t cSizeScatter = cSize / options.rankSize;

    uint8_t *aDevice;
    ACL_CHECK(aclrtMalloc((void **)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *aHost;
    ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
    ReadFile("./output/a_gm.bin", aHost, aSize);
    ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bDevice;
    ACL_CHECK(aclrtMalloc((void **)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *bHost;
    ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
    ReadFile("./output/b_gm.bin", bHost, bSize);
    ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *biasDevice;
    ACL_CHECK(aclrtMalloc((void **)(&biasDevice), biasSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *biasHost;
    ACL_CHECK(aclrtMallocHost((void **)(&biasHost), biasSize));
    ReadFile("./output/bias_gm.bin", biasHost, biasSize);
    ACL_CHECK(aclrtMemcpy(biasDevice, biasSize, biasHost, biasSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *scaleDevice;
    ACL_CHECK(aclrtMalloc((void **)(&scaleDevice), scaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *scaleHost;
    ACL_CHECK(aclrtMallocHost((void **)(&scaleHost), scaleSize));
    ReadFile("./output/scale_gm.bin", scaleHost, scaleSize);
    ACL_CHECK(aclrtMemcpy(scaleDevice, scaleSize, scaleHost, scaleSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *cDevice;
    ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSizeScatter, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *cHost;
    ACL_CHECK(aclrtMallocHost((void **)(&cHost), cSize));
    ReadFile("./output/c_gm.bin", cHost, cSize);
    ACL_CHECK(aclrtMemcpy(cDevice, cSizeScatter, cHost, cSizeScatter, ACL_MEMCPY_HOST_TO_DEVICE));

    void *symmPtr = shmem_malloc((204 * 1024 * 1024) * sizeof(__fp16));
    uint8_t *symmetricPtr = (uint8_t *)symmPtr;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    for (int i = 0; i < 1; i++) {
        ShmemMatmulReduceScatterQuantPerchn<<<BLOCK_NUM, nullptr, stream>>>(shmemx_get_ffts_config(),
            aDevice, bDevice, cDevice, biasDevice, scaleDevice, symmetricPtr, m, n, k, SHMEM_TEAM_WORLD);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(cHost, cSizeScatter, cDevice, cSizeScatter, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", cHost, cSizeScatter, options.rankId * cSizeScatter);
    if (rankId == 0) {
        std::printf("test finished\n");
    }

    shmem_free(symmPtr);

    ACL_CHECK(aclrtFreeHost(aHost));
    ACL_CHECK(aclrtFreeHost(bHost));
    ACL_CHECK(aclrtFreeHost(cHost));
    ACL_CHECK(aclrtFreeHost(biasHost));
    ACL_CHECK(aclrtFreeHost(scaleHost));
    ACL_CHECK(aclrtFree(aDevice));
    ACL_CHECK(aclrtFree(bDevice));
    ACL_CHECK(aclrtFree(cDevice));
    ACL_CHECK(aclrtFreeHost(biasDevice));
    ACL_CHECK(aclrtFreeHost(scaleDevice));

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}