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
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

// shmem_host
#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_rma.h"
#include "host/shmem_host_team.h"

// utils
#include "utils.h"

#include "catcoc/catcoc.hpp"
#include "catcoc/comm_epilogue/comm_dispatch_policy.hpp"
#include "catcoc/comm_epilogue/block/comm_block_epilogue.hpp"
#include "catcoc/comm_epilogue/block/comm_block_swizzle.hpp"
#include "catcoc/comm_epilogue/tile/tile_remote_copy.hpp"
#include "catcoc/detail/remote_copy_type.hpp"
#include "catcoc/dgemm/block/block_swizzle_allgather.hpp"
#include "catcoc/dgemm/kernel/allgather_matmul.hpp"

static uint32_t gNpuNum = 8;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

using namespace AscendC;
using namespace Catcoc;

constexpr uint32_t BLOCK_NUM = 20;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;

using ElementA = half;
using ElementB = half;
using ElementC = half;
using ElementD = half;

CATLASS_GLOBAL
void ShmemAllGatherMatmul(
    uint64_t fftsAddr,
    GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD, GM_ADDR gmSymmetric,
    uint32_t m, uint32_t n, uint32_t k)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    // Define ArchTag
    using ArchTag = Catlass::Arch::AtlasA2;

    // Prepare comm address
    uint32_t rank = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    Catlass::GemmCoord problemShape{m, n, k};
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutD layoutD{m * rankSize, n};

    // Block level, define BlockMmad
    constexpr bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;
    using L1TileShape = Catlass::GemmShape<128, 256, 256>;
    using L0TileShape = Catlass::GemmShape<128, 256, 64>;
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType
    >;

    using BlockMmadScheduler = Catcoc::DGemm::Block::GemmIdentityBlockSwizzleAllGather<7, 1, 2>;
    using BlockRemapScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
    using BlockEpilogueScheduler = Catcoc::CommEpilogue::Block::BlockCommSwizzle<0>;

    using RemoteSrcType = CType;
    using RemoteDstType = DType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Put>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using CommBlockShape = Catlass::MatrixShape<64, UINT_MAX>;
    using CommCoreSplit = Catlass::MatrixShape<20, 1>;

    constexpr uint32_t UB_STAGES = 2;
    using EpilogueAllGatherTileShape = Catlass::MatrixShape<32, 256>;
    using EpilogueAllGatherDispatch = CommEpilogue::EpilogueAtlasA2CommToShareMem<UB_STAGES,
        Catcoc::detail::CopyMode::Gather>;
    using BlockEpilogueAllGather = CommEpilogue::Block::CommBlockEpilogue<
        EpilogueAllGatherDispatch,
        RemoteSrcType, RemoteDstType,
        CommCoreSplit,
        CommBlockShape,
        EpilogueAllGatherTileShape, TileRemoteCopy, TileScheduler,
        BlockRemapScheduler
    >;

    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 3;
    using AllGatherMatmulKernel = DGemm::Kernel::AllGatherMatmul<
        BlockMmad,
        BlockEpilogueAllGather,
        BlockMmadScheduler,
        BlockEpilogueScheduler,
        WORKSPACE_STAGES
    >;

    Catlass::GemmCoord remapProblemShape{problemShape.m(), problemShape.k(), problemShape.k()};
    BlockRemapScheduler remapBlockScheduler(remapProblemShape,
        Catlass::MakeCoord(L1TileShape::M, problemShape.k()));

    Catlass::layout::RowMajor layoutSymmetric{
        L1TileShape::M * COMM_INTERVAL * rankSize * WORKSPACE_STAGES,
        k,
        k
    };

    // Prepare params
    typename AllGatherMatmulKernel::Params params{
        problemShape,
        rank, rankSize,
        COMM_INTERVAL,
        gmA, layoutA,
        gmB, layoutB,
        gmD, layoutD,
        gmSymmetric,
        {
            reinterpret_cast<__gm__ ElementC *>(gmSymmetric),
            layoutSymmetric,
            remapBlockScheduler
        }
    };

    // Call kernel
    AllGatherMatmulKernel matmulCommKernel;
    matmulCommKernel(params);
}

struct Options {
    static constexpr auto HELPER =
       "Usage: allgather_matmul rank_size rank_id ip_port m n k [device_id_list]\n";

    int rankSize;
    int rankId;
    std::string ipPort;
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::string dataPath;
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
            DATA_PATH_INDEX,
            DEVICE_LIST_INDEX,
            INDEX_MAX
        };

        if (argc > INDEX_MAX) {
            printf(HELPER);
            return -1;
        }

        rankSize = std::atoi(argv[RANK_SIZE_INDEX]);
        rankId = std::atoi(argv[RANK_ID_INDEX]);
        ipPort = argv[IP_PORT_INDEX];
        m = std::atoi(argv[M_INDEX]);
        n = std::atoi(argv[N_INDEX]);
        k = std::atoi(argv[K_INDEX]);
        dataPath = argv[DATA_PATH_INDEX];
        if (argc > DEVICE_LIST_INDEX) {
            char *idListStr = argv[DEVICE_LIST_INDEX];
            for (char *idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) {
                deviceIdList.push_back(std::atoi(idToken));
            }
        } else {
            for (size_t i = 0; i < rankSize; ++i) {
                deviceIdList.push_back(i);
            }
        }
        return 0;
    }

    std::string GetDataPath(std::string const &fileName = "") const
    {
        return dataPath + "/" + fileName;
    }
};

int main(int argc, char **argv)
{
    int status = SHMEM_SUCCESS;
    Options options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }
    int rankSize = options.rankSize;
    int rankId = options.rankId;
    std::string ipPort = options.ipPort;
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t k = options.k;
    int32_t deviceId = options.deviceIdList[rankId];

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipPort << std::endl;

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));
    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rankId, rankSize, gNpuMallocSpace, ipPort.c_str(), &attributes);
    status = shmem_init_attr(attributes);
    status = shmem_init_status();

    size_t aSize = static_cast<size_t>(m) * k * sizeof(__fp16);
    size_t bSize = static_cast<size_t>(k) * n * sizeof(__fp16);
    size_t dSize = static_cast<size_t>(m) * rankSize * n * sizeof(__fp16);

    uint8_t *aDevice;
    ACL_CHECK(aclrtMalloc((void **)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *aHost;
    ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
    ReadFile(options.GetDataPath("rank_" + std::to_string(rankId) + "_a.bin"), aHost, aSize);
    ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bDevice;
    ACL_CHECK(aclrtMalloc((void **)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *bHost;
    ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
    ReadFile(options.GetDataPath("rank_" + std::to_string(rankId) + "_b.bin"), bHost, bSize);
    ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *dDevice;
    ACL_CHECK(aclrtMalloc((void **)(&dDevice), dSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *dHost;
    ACL_CHECK(aclrtMallocHost((void **)(&dHost), dSize));

    void *symmPtr = shmem_malloc((204 * 1024 * 1024) * sizeof(__fp16));
    uint8_t *gmSymmetric = (uint8_t *)symmPtr;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "Before calling AG_MM kernel " << std::endl;
    for (int i = 0; i < 1; i++) {
        ShmemAllGatherMatmul<<<BLOCK_NUM, nullptr, stream>>>(
            shmemx_get_ffts_config(),
            aDevice, bDevice, dDevice, gmSymmetric,
            m, n, k
        );
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "After calling AG_MM kernel " << std::endl;

    ACL_CHECK(aclrtMemcpy(dHost, dSize, dDevice, dSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(options.GetDataPath("shmem_output.bin"), dHost, dSize);
    if (rankId == 0) {
        std::printf("test finished\n");
    }

    shmem_free(symmPtr);

    ACL_CHECK(aclrtFreeHost(aHost));
    ACL_CHECK(aclrtFreeHost(bHost));
    ACL_CHECK(aclrtFreeHost(dHost));
    ACL_CHECK(aclrtFree(aDevice));
    ACL_CHECK(aclrtFree(bDevice));
    ACL_CHECK(aclrtFree(dDevice));

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}