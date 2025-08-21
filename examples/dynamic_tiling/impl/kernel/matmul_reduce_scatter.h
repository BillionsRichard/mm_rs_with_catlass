#ifndef MATMUL_REDUCE_SCATTER_KERNEL_H
#define MATMUL_REDUCE_SCATTER_KERNEL_H

#include "info.h"

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

#include "catcoc/catcoc.hpp"
#include "catcoc/comm_epilogue/comm_dispatch_policy.hpp"
#include "catcoc/comm_epilogue/block/comm_block_epilogue.hpp"
#include "catcoc/comm_epilogue/block/comm_block_swizzle.hpp"
#include "catcoc/comm_epilogue/tile/tile_remote_copy.hpp"
#include "catcoc/detail/remote_copy_type.hpp"
#include "catcoc/dgemm/kernel/matmul_reduce_scatter.hpp"

using namespace AscendC;
using namespace Catcoc;

template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD,
    class ElementSymmetric, class LayoutSymmetric,
    uint32_t M0, uint32_t N0, uint32_t K0
>
CATLASS_DEVICE
void MatmulReduceScatterImpl(
    Catlass::GemmCoord& problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA& layoutA,
    GM_ADDR gmB, LayoutB& layoutB,
    GM_ADDR gmD, LayoutD& layoutD,
    uint32_t rank, uint32_t rankSize, uint32_t commInterval,
    Catlass::MatrixCoord& commCoreSplit,
    Catlass::MatrixCoord& commBlockShape,
    Catlass::MatrixCoord& commTileShape,
    GM_ADDR symmetricPtr, LayoutSymmetric& layoutSymmetric, shmem_team_t teamIdx = 0
)
{
    constexpr bool ENABLE_UNIT_FLAG = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<ENABLE_UNIT_FLAG>;

    using L1TileShape = Catlass::GemmShape<M0, N0, K0>;
    using L0TileShape = Catlass::GemmShape<M0, N0, 64>;

    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementSymmetric, LayoutSymmetric>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<
        MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType
    >;

    using BlockMmadScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
    using BlockEpilogueScheduler = Catcoc::CommEpilogue::Block::BlockCommSwizzle<0, true>;

    using RemoteSrcType = CType;
    using RemoteDstType = DType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Get>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    constexpr bool isDynamic = true;
    using EpilogueReduceScatterDispatch = CommEpilogue::EpilogueAtlasA2CommToLocalMem<UB_STAGES,
        Catcoc::detail::CopyMode::Scatter, isDynamic>;
    using BlockEpilogueReduceScatter = CommEpilogue::Block::CommBlockEpilogue<
        EpilogueReduceScatterDispatch,
        RemoteSrcType, RemoteDstType,
        void,
        void,
        void, TileRemoteCopy, TileScheduler,
        BlockMmadScheduler
    >;

    using MatmulReduceScatterKernel = DGemm::Kernel::MatmulReduceScatter<
        BlockMmad,
        BlockEpilogueReduceScatter,
        BlockMmadScheduler,
        BlockEpilogueScheduler,
        WORKSPACE_STAGES
    >;

    Catlass::GemmCoord problemShapeInRank = problemShape / Catlass::MakeCoord<uint32_t>(rankSize, 1, 1);
    BlockMmadScheduler matmulBlockScheduler(problemShapeInRank, Catlass::MakeCoord<uint32_t>(M0, N0));

    // uint32_t rank = shmem_team_my_pe(teamIdx);
    // uint32_t rankSize = shmem_team_n_pes(teamIdx);

    // Prepare params
    typename MatmulReduceScatterKernel::Params params{
        problemShape,
        rank, rankSize, teamIdx,
        commInterval,
        gmA, layoutA,
        gmB, layoutB,
        gmD, layoutD,
        symmetricPtr,
        {
            reinterpret_cast<__gm__ ElementSymmetric *>(symmetricPtr),
            layoutSymmetric,
            matmulBlockScheduler,
            commCoreSplit,
            commBlockShape,
            commTileShape
        }
    };

    // Call kernel
    MatmulReduceScatterKernel matmulCommKernel;
    matmulCommKernel(params);
}

template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD,
    class ElementSymmetric, class LayoutSymmetric
>
CATLASS_DEVICE
void MatmulReduceScatterImpl_M0_256(
    Catlass::GemmCoord& problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA& layoutA,
    GM_ADDR gmB, LayoutB& layoutB,
    GM_ADDR gmD, LayoutD& layoutD,
    uint32_t rank, uint32_t rankSize, uint32_t commInterval,
    Catlass::MatrixCoord& commCoreSplit,
    Catlass::MatrixCoord& commBlockShape,
    Catlass::MatrixCoord& commTileShape,
    GM_ADDR symmetricPtr, LayoutSymmetric& layoutSymmetric
)
{
    MatmulReduceScatterImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, ElementSymmetric, LayoutSymmetric, 256, 128, 256>(
        problemShape, l1TileShape,
        gmA, layoutA, gmB, layoutB, gmD, layoutD,
        rank, rankSize, commInterval,
        commCoreSplit, commBlockShape, commTileShape,
        symmetricPtr, layoutSymmetric
    );
}

template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD,
    class ElementSymmetric, class LayoutSymmetric
>
CATLASS_DEVICE
void MatmulReduceScatterImpl_M0_128(
    Catlass::GemmCoord& problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA& layoutA,
    GM_ADDR gmB, LayoutB& layoutB,
    GM_ADDR gmD, LayoutD& layoutD,
    uint32_t rank, uint32_t rankSize, uint32_t commInterval,
    Catlass::MatrixCoord& commCoreSplit,
    Catlass::MatrixCoord& commBlockShape,
    Catlass::MatrixCoord& commTileShape,
    GM_ADDR symmetricPtr, LayoutSymmetric& layoutSymmetric
)
{
    MatmulReduceScatterImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, ElementSymmetric, LayoutSymmetric, 128, 256, 256>(
        problemShape, l1TileShape,
        gmA, layoutA, gmB, layoutB, gmD, layoutD,
        rank, rankSize, commInterval,
        commCoreSplit, commBlockShape, commTileShape,
        symmetricPtr, layoutSymmetric
    );
}
template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementD, class LayoutD,
    class ElementSymmetric, class LayoutSymmetric
>
CATLASS_GLOBAL
void MatmulReduceScatter(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmD, GM_ADDR symmetricPtr, CocTilingParams cocTiling, shmem_team_t teamIdx = 0
)
{
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Catlass::Arch::AtlasA2;
    Catlass::Arch::Resource<ArchTag> resource;

    uint32_t m = cocTiling.m;
    uint32_t n = cocTiling.n;
    uint32_t k = cocTiling.k;
    uint32_t m0 = cocTiling.m0;
    uint32_t n0 = cocTiling.n0;
    uint32_t k0 = cocTiling.k0;
    uint32_t commInterval = cocTiling.commInterval;
    uint32_t commTileM = cocTiling.commTileM;
    uint32_t commNpuSplit = cocTiling.commNpuSplit;
    uint32_t commDataSplit = cocTiling.commDataSplit;
    uint32_t commBlockM = cocTiling.commBlockM;

    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::GemmCoord l1TileShape{m0, n0, k0};

    Catlass::MatrixCoord commCoreSplit{commDataSplit, commNpuSplit};
    Catlass::MatrixCoord commBlockShape{commBlockM, n0};
    Catlass::MatrixCoord commTileShape{commTileM / 2, n0};

    uint32_t rank = shmem_team_my_pe(teamIdx);
    uint32_t rankSize = shmem_team_n_pes(teamIdx);

    LayoutA layoutA{m, k};
    LayoutB layoutB{k, n};
    LayoutD layoutD{m / rankSize, n};
    LayoutSymmetric layoutSymmetric{m0 * commInterval * BLOCK_NUM * WORKSPACE_STAGES, n0, n0};
    
    if (m0 == 128){
        MatmulReduceScatterImpl_M0_128<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, ElementSymmetric, LayoutSymmetric>(
            problemShape, l1TileShape,
            gmA, layoutA, gmB, layoutB, gmD, layoutD,
            rank, rankSize, commInterval,
            commCoreSplit, commBlockShape, commTileShape,
            symmetricPtr, layoutSymmetric
        );
    } else {
        MatmulReduceScatterImpl_M0_256<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementD, LayoutD, ElementSymmetric, LayoutSymmetric>(
            problemShape, l1TileShape,
            gmA, layoutA, gmB, layoutB, gmD, layoutD,
            rank, rankSize, commInterval,
            commCoreSplit, commBlockShape, commTileShape,
            symmetricPtr, layoutSymmetric
        );
    }
}

#endif // MATMUL_REDUCE_SCATTER_KERNEL_H