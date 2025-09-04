#include <acl/acl.h>
#include <iostream>
#include <vector>
#include <cstring>
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_rma.h"
#include "host/shmem_host_team.h"
#include "utils.h"
#include "catcoc/catcoc.hpp"
#include "catcoc/comm_epilogue/comm_dispatch_policy.hpp"
#include "catcoc/comm_epilogue/block/comm_block_epilogue.hpp"
#include "catcoc/comm_epilogue/block/comm_block_swizzle.hpp"
#include "catcoc/comm_epilogue/tile/tile_remote_copy.hpp"
#include "catcoc/detail/remote_copy_type.hpp"
#include "catcoc/dgemm/block/block_swizzle_allgather.hpp"
#include "catcoc/dgemm/kernel/allgather_matmul_alltoall.hpp"

using namespace AscendC;
using namespace Catcoc;
using ElementA = half;
using ElementB = half;
using ElementC = half;
using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;

CATLASS_GLOBAL
void ShmemAllGatherMatmulAlltoall(uint64_t fftsAddr, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, uint32_t m, uint32_t n, uint32_t k, shmem_team_t teamIdx = 0) {
    AscendC::SetSyncBaseAddr(fftsAddr);
    using ArchTag = Catlass::Arch::AtlasA2;
    uint32_t rank = shmem_team_my_pe(teamIdx);
    uint32_t rankSize = shmem_team_n_pes(teamIdx);
    Catlass::GemmCoord problemShape{m, n, k};
    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n / rankSize};
    LayoutC layoutC{m, n};
    
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<true>;
    using L1TileShape = Catlass::GemmShape<128, 256, 256>;
    using L0TileShape = Catlass::GemmShape<128, 256, 64>;
    
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, Catlass::layout::RowMajor>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    
    using BlockSchedulerForMatmul = typename Catcoc::DGemm::Block::GemmBlockSwizzleAllGatherMesh<7, 1>;
    using CommBlockScheduler = CommEpilogue::Block::BlockCommSwizzle<0>;
    
    using RemoteType = Catlass::Gemm::GemmType<half, Catlass::layout::RowMajor>;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteType, RemoteType, Catcoc::detail::CopyDirect::Put>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using CommBlockShape = Catlass::MatrixShape<64, 64>;
    using CommCoreSplit = Catlass::MatrixShape<20, 1>;
    using CommTileShape = Catlass::MatrixShape<32, 64>;
    
    using AllGatherDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<2, Catcoc::detail::CopyMode::Gather>;
    using BlockEpilogueAllGather = CommEpilogue::Block::CommBlockEpilogue<AllGatherDispatch, RemoteType, RemoteType, CommCoreSplit, CommBlockShape, CommTileShape, TileRemoteCopy, TileScheduler>;
    
    using DType = CType;
    // using ACL_CHECK = Catlass::Epilogue::EpilogueAtlasA2<2>;
    using CopyTileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;
    using TileCopy = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType, CType, DType>;
    // using BlockEpilogueScatter = Catlass::Epilogue::Block::BlockEpilogue<ACL_CHECK, CType, CType, DType, TileCopy, TileCopy, TileCopy, TileCopy, CopyTileScheduler>;

    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 3;
    using Kernel = DGemm::Kernel::AllGatherMatmulAlltoall<BlockMmad, BlockEpilogueAllGather, BlockSchedulerForMatmul, CommBlockScheduler>;
    
    typename BlockEpilogueAllGather::Params agParams{};
    // typename BlockEpilogueScatter::Params scatterParams{};
    
    typename Kernel::Params params{problemShape, rank, rankSize, teamIdx, a, layoutA, b, layoutB, c, layoutC, ws, agParams, COMM_INTERVAL};
    Kernel kernel;
    kernel(params);
}

struct Options {
    int rankSize, rankId;
    std::string ipPort;
    uint32_t m=0, n=0, k=0;
    std::string dataPath;
    std::vector<int> deviceIdList;
    int Parse(int argc, char **argv) {
        if (argc < 8) { printf("Usage: %s rank_size rank_id ip_port m n k data_path [device_list]\n", argv[0]); return -1; }
        rankSize = std::atoi(argv[1]);
        rankId = std::atoi(argv[2]);
        ipPort = argv[3];
        m = std::atoi(argv[4]);
        n = std::atoi(argv[5]);
        k = std::atoi(argv[6]);
        dataPath = argv[7];
        if (argc > 8) {
            char *idListStr = argv[8];
            for (char *idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) { deviceIdList.push_back(std::atoi(idToken)); }
        } else {
            for (size_t i = 0; i < rankSize; ++i) deviceIdList.push_back(i);
        }
        return 0;
    }
};

int main(int argc, char **argv) {
    Options options;
    if (options.Parse(argc, argv) != 0) { std::cerr << "Invalid args\n"; return 1; }
    int rankSize = options.rankSize, rankId = options.rankId;
    uint32_t m = options.m, n = options.n, k = options.k;
    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(options.deviceIdList[rankId]));
    ACL_CHECK(aclrtCreateStream(&stream));
    shmem_init_attr_t *attributes;
    ACL_CHECK(shmem_set_attr(rankId, rankSize, 1024UL * 1024 * 1024, options.ipPort.c_str(), &attributes));
    ACL_CHECK(shmem_init_attr(attributes));
    ACL_CHECK(shmem_init_status());
    
    uint32_t n_per_rank = n / rankSize;
    size_t aSize = (size_t)m * k * sizeof(ElementA);
    size_t bSize = (size_t)k * n_per_rank * sizeof(ElementB);
    size_t cSize = (size_t)m * n * sizeof(ElementC);
    uint8_t *aDev, *aHost, *bDev, *bHost, *cDev, *cHost;
    ACL_CHECK(aclrtMalloc((void **)(&aDev), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
    ReadFile(options.dataPath + "/a_gm_rank" + std::to_string(rankId) + ".bin", aHost, aSize);
    ACL_CHECK(aclrtMemcpy(aDev, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMalloc((void **)(&bDev), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
    ReadFile(options.dataPath + "/b_gm_rank" + std::to_string(rankId) + ".bin", bHost, bSize);
    ACL_CHECK(aclrtMemcpy(bDev, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMalloc((void **)(&cDev), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&cHost), cSize));
    ACL_CHECK(aclrtMemset(cDev, cSize, 0, cSize));
    
    using L1TileShape = Catlass::GemmShape<128, 128, 64>;
    constexpr uint32_t WORKSPACE_STAGES = 2;
    constexpr uint32_t COMM_INTERVAL = 3;
    uint32_t commSizeM = COMM_INTERVAL * L1TileShape::M;
    size_t ag_bytes_per_stage = rankSize * commSizeM * k * sizeof(ElementA);
    size_t scatter_bytes_per_stage = commSizeM * n * sizeof(ElementC);
    size_t totalWorkspace = (ag_bytes_per_stage + scatter_bytes_per_stage) * WORKSPACE_STAGES;
    
    void *symmPtr = shmem_malloc(totalWorkspace);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ShmemAllGatherMatmulAlltoall<<<20, nullptr, stream>>>(shmemx_get_ffts_config(), aDev, bDev, cDev, (uint8_t*)symmPtr, m, n, k);
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(cHost, cSize, cDev, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile(options.dataPath + "/output_rank" + std::to_string(rankId) + ".bin", cHost, cSize);
    
    shmem_free(symmPtr);
    ACL_CHECK(aclrtFreeHost(aHost));
    ACL_CHECK(aclrtFreeHost(bHost));
    ACL_CHECK(aclrtFreeHost(cHost));
    ACL_CHECK(aclrtFree(aDev));
    ACL_CHECK(aclrtFree(bDev));
    ACL_CHECK(aclrtFree(cDev));
    ACL_CHECK(shmem_finalize());
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(options.deviceIdList[rankId]));
    ACL_CHECK(aclFinalize());
    return 0;
}
