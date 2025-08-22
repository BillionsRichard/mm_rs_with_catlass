#include <acl/acl.h>

#include <iostream>
#include <vector>
#include <cstring>

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
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
#include "utils/utils.h"

#include "catcoc/catcoc.hpp"
#include "catcoc/comm_epilogue/comm_dispatch_policy.hpp"
#include "catcoc/comm_epilogue/block/comm_block_epilogue.hpp"
#include "catcoc/comm_epilogue/block/comm_block_swizzle.hpp"
#include "catcoc/comm_epilogue/tile/tile_remote_copy.hpp"
#include "catcoc/detail/remote_copy_type.hpp"
#include "catcoc/dgemm/block/block_swizzle_allgather.hpp"
#include "catcoc/dgemm/kernel/allgather_dequant_matmul.hpp"

static uint32_t gNpuNum = 8;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

using namespace AscendC;
using namespace Catcoc;

constexpr uint32_t BLOCK_NUM = 20;
constexpr int32_t BLOCK_SIZE_16 = 16;

using ElementA = int8_t;
using ElementB = int8_t;
using ElementC = int32_t;
using ElementD = half;
using ElementScale = float;
using ElementPerTokenScale = float;
using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;
using LayoutScale = Catlass::layout::VectorLayout;
using LayoutPerTokenScale = Catlass::layout::VectorLayout;

CATLASS_GLOBAL
void ShmemAllGatherMatmul(uint64_t fftsAddr, GM_ADDR aDevice, GM_ADDR bDevice, GM_ADDR cDevice, GM_ADDR symmetricPtr,
    GM_ADDR dDevice, GM_ADDR deviceScale, GM_ADDR devicePerTokenScale, uint32_t m, uint32_t n, uint32_t k)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    // Define ArchTag
    using ArchTag = Catlass::Arch::AtlasA2;

    Catlass::GemmCoord problemShape{m, n, k};

    // Prepare comm address
    uint32_t rank = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    // Block level, Define the layout of each input matrix
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m * rankSize, n};
    LayoutD layoutD{m * rankSize, n};
    LayoutScale layoutScale{n};
    LayoutPerTokenScale layoutPerTokenScale{m};

    // Block level, define BlockMmad
    constexpr bool enableUnitFlag = false;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = Catlass::GemmShape<128, 256, 256>;
    using L0TileShape = Catlass::GemmShape<128, 256, 64>;
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using BlockMmad =
        Catlass::Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    using BlockSchedulerForAllgather = typename Catcoc::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
    using CommBlockScheduler = CommEpilogue::Block::BlockCommSwizzle<0>;

    using RemoteSrcType = AType;
    using RemoteDstType = AType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Put>;
    using TileSchedulerForAllgather = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using CommBlockShape = Catlass::MatrixShape<64, UINT_MAX>;
    using CommCoreSplit = Catlass::MatrixShape<20, 1>;

    constexpr uint32_t UB_STAGES = 2;
    using AllGatherTileShape = Catlass::MatrixShape<32, 256>;
    using AllGatherDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<UB_STAGES, Catcoc::detail::CopyMode::Gather>;
    using BlockEpilogueAllGather = CommEpilogue::Block::CommBlockEpilogue<AllGatherDispatch,
        RemoteSrcType,
        RemoteDstType,
        CommCoreSplit,
        CommBlockShape,
        AllGatherTileShape,
        TileRemoteCopy,
        TileSchedulerForAllgather>;

    constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Catlass::Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleType = Catlass::Gemm::GemmType<ElementScale, LayoutScale>;
    using PerTokenScaleType = Catlass::Gemm::GemmType<ElementPerTokenScale, LayoutPerTokenScale>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;

    using RowBroadcastMulType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using BroadcastOneBlkType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;

    using EpilogueTileShape = Catlass::MatrixShape<32, 256>;
    using TileRowBroadcastMul =
        Catlass::Epilogue::Tile::TileRowBroadcastMul<ArchTag, RowBroadcastMulType, EpilogueTileShape>;
    using TileBroadcastOneBlk =
        Catlass::Epilogue::Tile::TileBroadcastOneBlk<ArchTag, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Catlass::Epilogue::Tile::TileOneBlkColumnBroadcastMul<ArchTag, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using TileCopy = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, PerTokenScaleType, DType>;
    using TileSchedulerForDequant = Catlass::Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogueDequant = Catlass::Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy,
        CType,
        ScaleType,
        PerTokenScaleType,
        DType,
        TileRowBroadcastMul,
        TileBroadcastOneBlk,
        TileOneBlkColumnBroadcastMul,
        TileCopy,
        TileSchedulerForDequant>;

    using BlockSchedulerForDequant = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 3;
    using AllGatherMatmulKernel = DGemm::Kernel::AllGatherDequantMatmul<BlockMmad,
        BlockEpilogueAllGather,
        BlockSchedulerForAllgather,
        BlockEpilogueDequant,
        BlockSchedulerForDequant,
        CommBlockScheduler,
        WORKSPACE_STAGES>;

    typename BlockEpilogueAllGather::Params allGatherParams{};
    typename BlockEpilogueDequant::Params dequantParams{
        reinterpret_cast<__gm__ ElementScale *>(deviceScale),
        layoutScale,
        reinterpret_cast<__gm__ ElementPerTokenScale *>(devicePerTokenScale),
        layoutPerTokenScale,
        reinterpret_cast<__gm__ ElementD *>(dDevice),
        layoutD,
    };

    // Prepare params
    typename AllGatherMatmulKernel::Params params{
        problemShape,
        rank,
        rankSize,
        aDevice,
        layoutA,
        bDevice,
        layoutB,
        cDevice,
        layoutC,
        symmetricPtr,
        allGatherParams,
        dequantParams,
        COMM_INTERVAL
    };

    // Call kernel
    AllGatherMatmulKernel matmulCommKernel;
    matmulCommKernel(params);
}

struct Options {
    static constexpr auto helper = "Usage: allgather_matmul m n k transA transB\n";

    int rankSize;
    int rankId;
    std::string ipPort;
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
    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rankId, rankSize, gNpuMallocSpace, ipPort.c_str(), &attributes);
    status = shmem_init_attr(attributes);
    status = shmem_init_status();
    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    size_t aSize = static_cast<size_t>(m) * k * sizeof(int8_t);
    size_t bSize = static_cast<size_t>(k) * n * sizeof(int8_t);
    size_t cSize = static_cast<size_t>(m) * rankSize * n * sizeof(int32_t);

    size_t scaleSize = static_cast<size_t>(n) * sizeof(ElementScale);
    size_t perTokenScaleSize = static_cast<size_t>(m) * sizeof(ElementPerTokenScale);
    size_t dSize = static_cast<size_t>(m) * rankSize * n * sizeof(half);

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

    uint8_t *cDevice;
    ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *cHost;
    ACL_CHECK(aclrtMallocHost((void **)(&cHost), cSize));
    ReadFile("./output/c_gm.bin", cHost, cSize);
    ACL_CHECK(aclrtMemcpy(cDevice, cSize, cHost, cSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *dDevice;
    ACL_CHECK(aclrtMalloc((void **)(&dDevice), dSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *dHost;
    ACL_CHECK(aclrtMallocHost((void **)(&dHost), dSize));
    ReadFile("./output/d_gm.bin", dHost, dSize);
    ACL_CHECK(aclrtMemcpy(dDevice, dSize, dHost, dSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceScale;
    ACL_CHECK(aclrtMalloc((void **)(&deviceScale), scaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *scaleHost;
    ACL_CHECK(aclrtMallocHost((void **)(&scaleHost), scaleSize));
    ReadFile("./output/scale_gm.bin", scaleHost, scaleSize);
    ACL_CHECK(aclrtMemcpy(deviceScale, scaleSize, scaleHost, scaleSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *devicePerTokenScale;
    ACL_CHECK(aclrtMalloc((void **)(&devicePerTokenScale), perTokenScaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *perTokenScaleHost;
    ACL_CHECK(aclrtMallocHost((void **)(&perTokenScaleHost), perTokenScaleSize));
    ReadFile("./output/perTokenScale_gm.bin", perTokenScaleHost, perTokenScaleSize);
    ACL_CHECK(aclrtMemcpy(
        devicePerTokenScale, perTokenScaleSize, perTokenScaleHost, perTokenScaleSize, ACL_MEMCPY_HOST_TO_DEVICE));

    void *symmPtr = shmem_malloc((204 * 1024 * 1024) * sizeof(__fp16));
    uint8_t *symmetricPtr = (uint8_t *)symmPtr;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    for (int i = 0; i < 1; i++) {
        ShmemAllGatherMatmul<<<BLOCK_NUM, nullptr, stream>>>(
            fftsAddr, aDevice, bDevice, cDevice, symmetricPtr, dDevice, deviceScale, devicePerTokenScale, m, n, k);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(dHost, dSize, dDevice, dSize, ACL_MEMCPY_DEVICE_TO_HOST));
    if (rankId == 0) {
        WriteFile("./output/output.bin", dHost, dSize);
        std::printf("test finished\n");
    }
    shmem_free(symmPtr);

    ACL_CHECK(aclrtFreeHost(aHost));
    ACL_CHECK(aclrtFreeHost(bHost));
    ACL_CHECK(aclrtFreeHost(cHost));
    ACL_CHECK(aclrtFreeHost(dHost));
    ACL_CHECK(aclrtFreeHost(scaleHost));
    ACL_CHECK(aclrtFreeHost(perTokenScaleHost));
    ACL_CHECK(aclrtFree(aDevice));
    ACL_CHECK(aclrtFree(bDevice));
    ACL_CHECK(aclrtFree(cDevice));
    ACL_CHECK(aclrtFree(dDevice));
    ACL_CHECK(aclrtFree(devicePerTokenScale));
    ACL_CHECK(aclrtFree(deviceScale));

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());
    return 0;
}