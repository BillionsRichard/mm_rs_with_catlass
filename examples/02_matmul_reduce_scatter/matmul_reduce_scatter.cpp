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
#include "catcoc/dgemm/kernel/matmul_reduce_scatter.hpp"

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
void ShmemMatmulReduceScatter(
    uint64_t fftsAddr,
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR symmetricPtr,
    uint32_t m, uint32_t n, uint32_t k
)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    // Define ArchTag
    using ArchTag = Catlass::Arch::AtlasA2;

    Catlass::GemmCoord problemShape{m, n, k};

    // Prepare comm address
    uint32_t rank = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();
    using ElementC = half;

    // Block level, Define the layout of each input matrix
    Catlass::layout::RowMajor layoutA{m, k, k};
    Catlass::layout::RowMajor layoutB{k, n, n};
    Catlass::layout::RowMajor layoutC{m / rankSize, n, n};

    // Block level, define BlockMmad
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = Catlass::GemmShape<128, 256, 256>;
    using L0TileShape = Catlass::GemmShape<128, 256, 64>;
    using AType = Catlass::Gemm::GemmType<half, LayoutA>;
    using BType = Catlass::Gemm::GemmType<half, LayoutB>;
    using CType = Catlass::Gemm::GemmType<half, LayoutC>;
    using DType = Catlass::Gemm::GemmType<half, LayoutD>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<MmadDispatchPolicy,
        L1TileShape, L0TileShape, AType, BType, CType>;

    using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
    using CommBlockScheduler = CommEpilogue::Block::BlockCommSwizzle<0>;

    using RemoteSrcType = CType;
    using RemoteDstType = DType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Get>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using CommBlockShape = Catlass::MatrixShape<64, 256>;
    using CommCoreSplit = Catlass::MatrixShape<20, 1>;

    constexpr uint32_t ubStages = 2;
    using ReduceScatterTileShape = Catlass::MatrixShape<32, 256>;
    using ReduceScatterDispatch = CommEpilogue::EpilogueAtlasA2CommToLocalMem<ubStages,
        Catcoc::detail::CopyMode::Scatter>;
    using BlockEpilogueReduceScatter = CommEpilogue::Block::CommBlockEpilogue<
        ReduceScatterDispatch,
        RemoteSrcType, RemoteDstType,
        CommCoreSplit,
        CommBlockShape,
        ReduceScatterTileShape, TileRemoteCopy, TileScheduler,
        BlockScheduler
    >;

    constexpr uint32_t workspaceStages = 2;
    constexpr uint32_t commInterval = 10;
    using MatmulReduceScatterKernel = DGemm::Kernel::MatmulReduceScatter<
        BlockMmad,
        BlockEpilogueReduceScatter,
        BlockScheduler,
        CommBlockScheduler,
        workspaceStages
    >;
    Catlass::GemmCoord problemShapeInRank = problemShape / Catlass::MakeCoord<uint32_t>(rankSize, 1, 1);
    BlockScheduler matmulBlockScheduler(problemShapeInRank, Catlass::MakeCoord(L1TileShape::M, L1TileShape::N));

    Catlass::layout::RowMajor layoutPeerMemStore{
        L1TileShape::M * commInterval * BLOCK_NUM * workspaceStages, L1TileShape::N,
        L1TileShape::N
    };

    typename BlockEpilogueReduceScatter::Params reduceScatterParams{
        reinterpret_cast<__gm__ ElementC *>(symmetricPtr),
        layoutPeerMemStore,
        matmulBlockScheduler
    };

    // Prepare params
    typename MatmulReduceScatterKernel::Params params{
        problemShape,
        rank, rankSize,
        a, layoutA,
        b, layoutB,
        symmetricPtr,
        reduceScatterParams,
        c, layoutC,
        commInterval
    };

    // Call kernel
    MatmulReduceScatterKernel matmulCommKernel;
    matmulCommKernel(params);
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
    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rankId, rankSize, gNpuMallocSpace, ipPort.c_str(), &attributes);
    status = shmem_init_attr(attributes);
    status = shmem_init_status();

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    size_t aSize = static_cast<size_t>(m) * k * sizeof(__fp16);
    size_t bSize = static_cast<size_t>(k) * n * sizeof(__fp16);
    size_t cSize = static_cast<size_t>(m) * n * sizeof(__fp16);
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
        ShmemMatmulReduceScatter<<<BLOCK_NUM, nullptr, stream>>>(fftsAddr,
            aDevice, bDevice, cDevice, symmetricPtr, m, n, k);
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