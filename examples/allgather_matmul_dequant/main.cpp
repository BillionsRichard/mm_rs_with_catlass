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
#include "utils.h"

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
using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;
using LayoutScale = Catlass::layout::VectorLayout;

CATLASS_GLOBAL
void ShmemAllGatherMatmul(uint64_t fftsAddr, GM_ADDR aDevice, GM_ADDR bDevice, GM_ADDR cDevice, GM_ADDR symmetricPtr,
    GM_ADDR dDevice, GM_ADDR fused_scale, uint32_t m, uint32_t n, uint32_t k)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Catlass::Arch::AtlasA2;
    Catlass::GemmCoord problemShape{m, n, k};
    uint32_t rank = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();

    // Define layouts
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutC layoutC{m * rankSize, n};
    LayoutD layoutD{m * rankSize, n};
    LayoutScale layoutScale{n};

    // Define GEMM
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<false>;
    using L1TileShape = Catlass::GemmShape<128, 256, 256>;
    using L0TileShape = Catlass::GemmShape<128, 256, 64>;
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    // Define Communication (AllGather for Matrix A)
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
        RemoteSrcType, RemoteDstType, CommCoreSplit, CommBlockShape, AllGatherTileShape, TileRemoteCopy, TileSchedulerForAllgather>;

    // Define Dequantization Epilogue (Simplified)
    using EpilogueDispatchPolicy = Catlass::Epilogue::EpilogueAtlasA2Dequant;
    using ScaleType = Catlass::Gemm::GemmType<ElementScale, LayoutScale>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;
    using BroadcastMulType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using EpilogueTileShape = Catlass::MatrixShape<32, 256>;
    using TileBroadcastMul = Catlass::Epilogue::Tile::TileRowBroadcastMul<ArchTag, BroadcastMulType, EpilogueTileShape>;
    using TileCopy = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType, ScaleType, DType>;
    using TileSchedulerForDequant = Catlass::Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using BlockEpilogueDequant = Catlass::Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy,
        CType, ScaleType, DType, TileBroadcastMul, TileCopy, TileSchedulerForDequant>;

    // Define Schedulers
    using BlockSchedulerForDequant = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

    // Define Kernel
    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 3;
    using AllGatherMatmulKernel = DGemm::Kernel::AllGatherDequantMatmul<BlockMmad,
        BlockEpilogueAllGather, BlockSchedulerForAllgather, BlockEpilogueDequant,
        BlockSchedulerForDequant, CommBlockScheduler, WORKSPACE_STAGES>;

    // Prepare parameters
    typename BlockEpilogueAllGather::Params allGatherParams{};
    typename BlockEpilogueDequant::Params dequantParams{
        reinterpret_cast<__gm__ ElementScale *>(fused_scale),
        layoutScale,
        reinterpret_cast<__gm__ ElementD *>(dDevice),
        layoutD,
    };

    typename AllGatherMatmulKernel::Params params{
        problemShape, rank, rankSize, aDevice, layoutA, bDevice, layoutB,
        cDevice, layoutC, symmetricPtr, allGatherParams, dequantParams, COMM_INTERVAL
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
            printf(helper);
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
    Options options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }
    int rankSize = options.rankSize;
    int rankId = options.rankId;
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t k = options.k;

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceIdList[rankId]));
    ACL_CHECK(aclrtCreateStream(&stream));
    shmem_init_attr_t *attributes;
    shmem_set_attr(rankId, rankSize, gNpuMallocSpace, options.ipPort.c_str(), &attributes);
    shmem_init_attr(attributes);
    shmem_init_status();

    size_t aSize = static_cast<size_t>(m) * k * sizeof(int8_t);
    size_t bSize = static_cast<size_t>(k) * n * sizeof(int8_t);
    size_t cSize = static_cast<size_t>(m) * rankSize * n * sizeof(int32_t);
    size_t scaleSize = static_cast<size_t>(n) * sizeof(ElementScale);
    size_t dSize = static_cast<size_t>(m) * rankSize * n * sizeof(half);

    uint8_t *aDevice, *bDevice, *cDevice, *dDevice, *fusedScaleDevice;
    uint8_t *aHost, *bHost, *fusedScaleHost;

    std::string aFileName = "a_gm_rank_" + std::to_string(rankId) + ".bin";
    ACL_CHECK(aclrtMalloc((void **)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
    ReadFile(options.GetDataPath(aFileName), aHost, aSize);
    ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc((void **)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
    ReadFile(options.GetDataPath("b_gm.bin"), bHost, bSize);
    ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc((void **)(&dDevice), dSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACL_CHECK(aclrtMalloc((void **)(&fusedScaleDevice), scaleSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&fusedScaleHost), scaleSize));
    ReadFile(options.GetDataPath("scale_gm.bin"), fusedScaleHost, scaleSize);
    ACL_CHECK(aclrtMemcpy(fusedScaleDevice, scaleSize, fusedScaleHost, scaleSize, ACL_MEMCPY_HOST_TO_DEVICE));

    void *symmPtr = shmem_malloc((204 * 1024 * 1024) * sizeof(__fp16));

    ACL_CHECK(aclrtSynchronizeStream(stream));
    ShmemAllGatherMatmul<<<BLOCK_NUM, nullptr, stream>>>(
        shmemx_get_ffts_config(), aDevice, bDevice, cDevice, (uint8_t *)symmPtr, dDevice, fusedScaleDevice, m, n, k);
    ACL_CHECK(aclrtSynchronizeStream(stream));

    if (rankId == 0) {
        uint8_t *dHost;
        ACL_CHECK(aclrtMallocHost((void **)(&dHost), dSize));
        ACL_CHECK(aclrtMemcpy(dHost, dSize, dDevice, dSize, ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(options.GetDataPath("output.bin"), dHost, dSize);
        ACL_CHECK(aclrtFreeHost(dHost));
        std::printf("test finished\n");
    }

    shmem_free(symmPtr);
    ACL_CHECK(aclrtFreeHost(aHost));
    ACL_CHECK(aclrtFreeHost(bHost));
    ACL_CHECK(aclrtFreeHost(fusedScaleHost));
    ACL_CHECK(aclrtFree(aDevice));
    ACL_CHECK(aclrtFree(bDevice));
    ACL_CHECK(aclrtFree(cDevice));
    ACL_CHECK(aclrtFree(dDevice));
    ACL_CHECK(aclrtFree(fusedScaleDevice));

    shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceIdList[rankId]));
    ACL_CHECK(aclFinalize());
    return 0;
}