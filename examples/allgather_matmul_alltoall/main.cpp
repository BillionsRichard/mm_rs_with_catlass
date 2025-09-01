#include <acl/acl.h>

#include <iostream>
#include <vector>
#include <cstring>

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp" ///--->include in
#include "catlass/gemm/dispatch_policy.hpp"
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
#include "catcoc/dgemm/kernel/allgather_matmul_alltoall.hpp"

static uint32_t gNpuNum = 8;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

using namespace AscendC;
using namespace Catcoc;

constexpr uint32_t BLOCK_NUM = 20;
constexpr int32_t BLOCK_SIZE_16 = 16;

using ElementA = half;
using ElementB = half;
using ElementC = half;
using ElementD = half;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;


CATLASS_GLOBAL
void ShmemAllGatherMatmulAlltoall(
    uint64_t fftsAddr,
    GM_ADDR aDevice,
    GM_ADDR bDevice,
    GM_ADDR cDevice,
    GM_ADDR symmetricPtr,
    uint32_t m, uint32_t n, uint32_t k, shmem_team_t teamIdx = 0)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    // Define ArchTag
    using ArchTag = Catlass::Arch::AtlasA2;

    uint32_t n_per_rank = n / rankSize;
    Catlass::GemmCoord problemShape{m, n, k}; // Logical problem shape

    // Prepare comm address
    uint32_t rank = shmem_team_my_pe(teamIdx);
    uint32_t rankSize = shmem_team_n_pes(teamIdx);

    // Define layouts based on the new data flow
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n_per_rank};
    LayoutC layoutC{m, n}; // Final output layout

    // Block level, define BlockMmad for half precision
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong; // No bias
    using L1TileShape = Catlass::GemmShape<128, 128, 128>; // Adjusted for half
    using L0TileShape = Catlass::GemmShape<128, 128, 32>;  // Adjusted for half
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>; // Accumulator is also half
    using BlockMmad =
        Catlass::Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    // Schedulers
    using BlockSchedulerForAllgather = typename Catcoc::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
    using CommBlockScheduler = CommEpilogue::Block::BlockCommSwizzle<0>;

    // AllGather Epilogue (similar to before, but types might change)
    using RemoteSrcTypeAG = AType;
    using RemoteDstTypeAG = AType;
    using TileRemoteCopyAG = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcTypeAG, RemoteDstTypeAG, Catcoc::detail::CopyDirect::Put>;
    using TileSchedulerForAllgather = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using CommBlockShapeAG = Catlass::MatrixShape<64, UINT_MAX / 2>;
    using CommCoreSplitAG = Catlass::MatrixShape<20, 1>;
    using AllGatherTileShape = Catlass::MatrixShape<32, 128>;
    using AllGatherDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<2, Catcoc::detail::CopyMode::Gather>;
    using BlockEpilogueAllGather = CommEpilogue::Block::CommBlockEpilogue<AllGatherDispatch, RemoteSrcTypeAG, RemoteDstTypeAG, CommCoreSplitAG, CommBlockShapeAG, AllGatherTileShape, TileRemoteCopyAG, TileSchedulerForAllgather>;

    // Alltoall Epilogue (This part is new and based on educated guesses)
    // After matmul and transpose, the data to be sent is [rankSize, n_per_rank, m]
    // The type will be half
    using RemoteTypeAT = Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor>;
    using TileRemoteCopyAT = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteTypeAT, RemoteTypeAT, Catcoc::detail::CopyDirect::Put>;
    using TileSchedulerForAlltoall = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using CommBlockShapeAT = Catlass::MatrixShape<64, 64>; // Example shape
    using CommCoreSplitAT = Catlass::MatrixShape<20, 1>;
    using AlltoallTileShape = Catlass::MatrixShape<32, 64>;
    using AlltoallDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<2, Catcoc::detail::CopyMode::Alltoall>; // Assuming an Alltoall mode exists
    using BlockEpilogueAlltoall = CommEpilogue::Block::CommBlockEpilogue<AlltoallDispatch, RemoteTypeAT, RemoteTypeAT, CommCoreSplitAT, CommBlockShapeAT, AlltoallTileShape, TileRemoteCopyAT, TileSchedulerForAlltoall>;

    // Define the final kernel
    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 3;
    using AllGatherMatmulAlltoallKernel = DGemm::Kernel::AllGatherMatmulAlltoall<BlockMmad,
        BlockEpilogueAllGather,
        BlockSchedulerForAllgather,
        BlockEpilogueAlltoall,
        CommBlockScheduler,
        WORKSPACE_STAGES>;

    // Prepare params for the kernel
    typename BlockEpilogueAllGather::Params allGatherParams{};
    typename BlockEpilogueAlltoall::Params alltoallParams{};

    typename AllGatherMatmulAlltoallKernel::Params params{
        problemShape,
        rank,
        rankSize, teamIdx,
        aDevice,
        layoutA,
        bDevice,
        layoutB,
        cDevice,
        layoutC,
        symmetricPtr,
        allGatherParams,
        alltoallParams,
        COMM_INTERVAL
    };

    // Call kernel
    AllGatherMatmulAlltoallKernel matmulCommKernel;
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

    uint32_t n_per_rank = n / rankSize;
    size_t aSize = static_cast<size_t>(m) * k * sizeof(half);
    size_t bSize = static_cast<size_t>(k) * n_per_rank * sizeof(half);
    size_t cSize = static_cast<size_t>(m) * n * sizeof(half);

    uint8_t *aDevice;
    ACL_CHECK(aclrtMalloc((void **)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *aHost;
    ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
    std::string a_filename = "a_gm_rank" + std::to_string(rankId) + ".bin";
    ReadFile(options.GetDataPath(a_filename), aHost, aSize);
    ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bDevice;
    ACL_CHECK(aclrtMalloc((void **)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *bHost;
    ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
    std::string b_filename = "b_gm_rank" + std::to_string(rankId) + ".bin";
    ReadFile(options.GetDataPath(b_filename), bHost, bSize);
    ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *cDevice;
    ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *cHost;
    ACL_CHECK(aclrtMallocHost((void **)(&cHost), cSize));
    ACL_CHECK(aclrtMemset(cDevice, cSize, 0, cSize)); // Ensure output buffer is zeroed out

    void *symmPtr = shmem_malloc((204 * 1024 * 1024) * sizeof(half));
    uint8_t *symmetricPtr = (uint8_t *)symmPtr;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    auto ffts_cfg = shmemx_get_ffts_config();
    for (int i = 0; i < 1; i++) {
        ShmemAllGatherMatmulAlltoall<<<BLOCK_NUM, nullptr, stream>>>(
            ffts_cfg, aDevice, bDevice, cDevice, symmetricPtr, m, n, k);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));

    // After this point, cHost contains the output C_gpu_i for the current rank.
    // This data can be written to a file for later comparison or compared in-place
    // against the corresponding slice of the C_golden matrix.
    std.string output_filename = "output_rank" + std::to_string(rankId) + ".bin";
    WriteFile(options.GetDataPath(output_filename), cHost, cSize);
    std::printf("test finished\n");
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