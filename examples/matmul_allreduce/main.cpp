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


// misc
#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>


// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/matmul_epilogue.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

// from shmem-templates
#include "kernel/matmul_epilogue_comm.hpp"

// shmem_host
#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_rma.h"
#include "host/shmem_host_team.h"

// utils
#include "utils.h"

static uint32_t gNpuNum = 8;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

using namespace AscendC;
using namespace Catlass;
using fp16_t = op::fp16_t;


struct CoCTiling {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t m0 = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    uint32_t swizzleDirect = 0;
    uint32_t swizzleOffset = 0;
    uint32_t ubMoveNum = 0;
    uint32_t pValue = 0;
    uint32_t commNpuSplit = 0;
    uint32_t commDataSplit = 0;
    uint32_t lenPerLoop = 0;
};

constexpr uint32_t BLOCK_NUM = 20;
constexpr int32_t BLOCK_SIZE_16 = 16;

using LayoutA = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutC = layout::RowMajor;

CATLASS_GLOBAL
void ShmemMatmulAllReduce(
    uint64_t fftsAddr, GemmCoord problemShape, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR symmetricPtr, CoCTiling cocTiling)
{
    // Set FFTS address
    shmemx_set_ffts_config(fftsAddr);

    // Define ArchTag
    using ArchTag = Arch::AtlasA2;

    // unzip cocTiling
    uint32_t m = cocTiling.m;
    uint32_t n = cocTiling.n;
    uint32_t k = cocTiling.k;
    uint32_t m0 = cocTiling.m0;
    uint32_t k0 = cocTiling.k0;
    uint32_t n0 = cocTiling.n0;
    uint32_t swizzleOffset = cocTiling.swizzleOffset;
    uint32_t swizzleDirect = cocTiling.swizzleDirect;
    uint32_t pValue = cocTiling.pValue;
    uint32_t commDataSplit = cocTiling.commDataSplit;
    uint32_t commNpuSplit = cocTiling.commNpuSplit;
    uint32_t ubMoveNum = cocTiling.ubMoveNum;
    uint32_t lenPerLoop = cocTiling.lenPerLoop;

    // Prepare comm address
    uint32_t rank = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();
    using ElementC = half;

    // Block level, Define the layout of each input matrix
    layout::RowMajor layoutA{m, k, k};
    layout::RowMajor layoutB{k, n, n};
    layout::RowMajor layoutC{m, n, n};

    GemmCoord blockShape{m0, n0, k0};

    // Block level, define BlockMmad
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;
    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<half, LayoutC>;
    using BlockMmad = Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    using ElementStore = half;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
    using CommBlockSwizzle = Gemm::Block::CommBlockSwizzleDynamic;

    // Block level, define BlockAllReduceEpilogue(ReduceScatter + AllGather)
    using BlockAllReduceEpilogue = Epilogue::Block::EpilogueAllReduce<BlockScheduler, CommBlockSwizzle>;

    // Kernel level
    using MatmulAllReduceKernel = Gemm::Kernel::MatmulEpilogueComm<BlockMmad, BlockAllReduceEpilogue, BlockScheduler>;

    // Prepare EpilogueComm params
    uint32_t maxUbPingPongSize = ubMoveNum / 2;

    BlockScheduler matmulBlockScheduler(problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));

    MatrixCoord commBlockShape{lenPerLoop / n0, n0};
    MatrixCoord commProcessShape{maxUbPingPongSize / n0, n0};

    CommBlockSwizzle commSwizzle{commBlockShape, rank, rankSize, 0, commDataSplit, commNpuSplit};

    AscendC::GlobalTensor<ElementStore> refC;
    refC.SetGlobalBuffer((__gm__ ElementStore *)c);
    typename BlockAllReduceEpilogue::Params epilogueCommParams{
            refC, layout::RowMajor(m, n, n),
            0,
            (__gm__ ElementC *)symmetricPtr,
            commBlockShape, commProcessShape,
            matmulBlockScheduler, commSwizzle};

    // Prepare params
    typename MatmulAllReduceKernel::Params params{
        problemShape, blockShape,
        pValue, rank, rankSize,
        a, layoutA,
        b, layoutB,
        symmetricPtr,
        epilogueCommParams};

    // Call kernel
    MatmulAllReduceKernel matmulCommKernel;
    matmulCommKernel(params);
}

struct Options {
    static constexpr auto helper = 
       "Usage: matmul_allreduce m n k transA transB [--block m0 n0 k0 --ubMoveNum ubMoveNum --pValue pValue --split commNpuSplit commDataSplit lenPerLoop --swizzle swizzleOffset swizzleDirect]\n";

    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t m0 = 128;
    uint32_t k0 = 256;
    uint32_t n0 = 256;
    uint32_t transA = 0;
    uint32_t transB = 0;
    uint32_t swizzleDirect = 1;
    uint32_t swizzleOffset = 7;
    uint32_t ubMoveNum = 16 * 1024;
    uint32_t pValue = 3;
    uint32_t commNpuSplit = 2;
    uint32_t commDataSplit = 1;
    uint32_t lenPerLoop = m0 * n0 / 2;

    int Parse(int argc, char **argv)
    {
        if (argc < 6) {
            printf(helper);
            return -1;
        }

        uint32_t argIndex = 1;
        m = std::atoi(argv[argIndex++]);
        n = std::atoi(argv[argIndex++]);
        k = std::atoi(argv[argIndex++]);
        transA = std::atoi(argv[argIndex++]);
        transB = std::atoi(argv[argIndex++]);

        while (argIndex < argc) {
            std::string flag = std::string(argv[argIndex++]);

            if (flag == "--pValue") {
                pValue = std::atoi(argv[argIndex++]);
            } else if (flag == "--ubMoveNum") {
                ubMoveNum = std::atoi(argv[argIndex++]);
            } else if (flag == "--split") {
                commNpuSplit = std::atoi(argv[argIndex++]);
                commDataSplit = std::atoi(argv[argIndex++]);
                lenPerLoop = std::atoi(argv[argIndex++]);
            } else if (flag == "--block") {
                m0 = std::atoi(argv[argIndex++]);
                n0 = std::atoi(argv[argIndex++]);
                k0 = std::atoi(argv[argIndex++]);
            } else if (flag == "--swizzle") {
                swizzleOffset = std::atoi(argv[argIndex++]);
                swizzleDirect = std::atoi(argv[argIndex++]);
            } else {
                printf(helper);
                return -1;
            }
        }

        return 0;
    }
};

int main(int argc, char **argv)
{
    int status = SHMEM_SUCCESS;
    int rankSize = atoi(argv[1]);
    int rankId = atoi(argv[2]);
    std::string ipport = argv[3];

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipport << std::endl;

    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = atoi(argv[4]) + rankId % gNpuNum;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rankId, rankSize, gNpuMallocSpace, ipport.c_str(), &attributes);
    status = shmem_init_attr(attributes);
    status = shmem_init_status();

    Options options;
    uint32_t m = atoi(argv[5]);
    uint32_t k = atoi(argv[6]);
    uint32_t n = atoi(argv[7]);
    uint32_t m0 = 128;
    uint32_t k0 = 256;
    uint32_t n0 = 256;
    uint32_t swizzleDirect = 1;
    uint32_t swizzleOffset = 7;
    uint32_t ubMoveNum = 16 * 1024;
    uint32_t pValue = 3;
    uint32_t commNpuSplit = 2;
    uint32_t commDataSplit = 1;
    uint32_t lenPerLoop = m0 * n0 / 2;

    // m, n, k
    GemmCoord problemShape{m, n, k};

    size_t aSize = static_cast<size_t>(m) * k * sizeof(__fp16);
    size_t bSize = static_cast<size_t>(k) * n * sizeof(__fp16);
    size_t cSize = static_cast<size_t>(m) * n * sizeof(__fp16);
    size_t workspaceSize = static_cast<size_t>(m) * n * sizeof(__fp16);

    uint8_t *aDevice;
    ACL_CHECK(aclrtMalloc((void **)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *aHost;
    ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
    std::string dataPath = argv[8];
    std::string aPath = dataPath + "/rank_" + std::to_string(rankId) + "_a.bin";
    ReadFile(aPath.c_str(), aHost, aSize);
    ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bDevice;
    ACL_CHECK(aclrtMalloc((void **)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *bHost;
    ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
    std::string bPath = dataPath + "/rank_" + std::to_string(rankId) + "_b.bin";
    ReadFile(bPath.c_str(), bHost, bSize);
    ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *cDevice;
    ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *cHost;
    ACL_CHECK(aclrtMallocHost((void **)(&cHost), cSize));
    memset(cHost, 0, cSize);  // 零初始化 C 矩阵
    ACL_CHECK(aclrtMemcpy(cDevice, cSize, cHost, cSize, ACL_MEMCPY_HOST_TO_DEVICE));

    void *symmPtr = shmem_malloc((204 * 1024 * 1024) * sizeof(__fp16));
    uint8_t *symmetricPtr = (uint8_t *)symmPtr;

    CoCTiling cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    cocTiling.m0 = m0;
    cocTiling.n0 = n0;
    cocTiling.k0 = k0;
    cocTiling.swizzleOffset = swizzleOffset;
    cocTiling.swizzleDirect = swizzleDirect;
    cocTiling.pValue = pValue;
    cocTiling.ubMoveNum = ubMoveNum;
    cocTiling.commNpuSplit = commNpuSplit;
    cocTiling.commDataSplit = commDataSplit;
    cocTiling.lenPerLoop = lenPerLoop;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    std::cout << "Before calling MM_AR kernel " << std::endl;
    for (int i = 0; i < 1; i++) {
        ShmemMatmulAllReduce<<<BLOCK_NUM, nullptr, stream>>>(shmemx_get_ffts_config(), problemShape, aDevice, bDevice, cDevice, symmetricPtr, cocTiling);
    }
    std::cout << "After calling MM_AR kernel " << std::endl;
    
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    if (rankId == 0) {
        std::string cPath = dataPath + "/shmem_output.bin";
        WriteFile(cPath.c_str(), cHost, cSize);
        std::printf("test finished\n");
    }

    shmem_free(symmPtr);

    ACL_CHECK(aclrtFreeHost(aHost));
    ACL_CHECK(aclrtFreeHost(bHost));
    ACL_CHECK(aclrtFreeHost(cHost));
    ACL_CHECK(aclrtFree(aDevice));
    ACL_CHECK(aclrtFree(bDevice));
    ACL_CHECK(aclrtFree(cDevice));

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}