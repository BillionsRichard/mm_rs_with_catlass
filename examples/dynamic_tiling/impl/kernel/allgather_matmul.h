#ifndef ALLGATHER_MATMUL_KERNEL_H
#define ALLGATHER_MATMUL_KERNEL_H
 
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
#include "catcoc/dgemm/block/block_swizzle_allgather.hpp"
#include "catcoc/dgemm/kernel/allgather_matmul.hpp"
 
using namespace AscendC;
using namespace Catcoc;
 
template <
    class ArchTag,
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    class ElementD, class LayoutD
>
CATLASS_DEVICE
void AllGatherMatmulImpl(
    Catlass::GemmCoord& problemShape,
    Catlass::GemmCoord& l1TileShape,
    GM_ADDR gmA, LayoutA& layoutA,
    GM_ADDR gmB, LayoutB& layoutB,
    GM_ADDR gmC, LayoutC& layoutC,
    uint32_t commInterval,
    Catlass::MatrixCoord& commCoreSplit,
    Catlass::MatrixCoord& commBlockShape,
    Catlass::MatrixCoord& commTileShape,
    GM_ADDR symmetricPtr, LayoutC& layoutD
)
{
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
 
    // TODO: 当前catlass中MmadAtlasA2Pingpong没有动态版本, 预留了l1TileShape
    using L1TileShape = Catlass::GemmShape<M0, N0, K0>;
    using L0TileShape = Catlass::GemmShape<M0, N0, 64>;
 
    using AType = Catlass::Gemm::GemmType<ElementA, LayoutA>;
    using BType = Catlass::Gemm::GemmType<ElementB, LayoutB>;
    using CType = Catlass::Gemm::GemmType<ElementC, LayoutC>;
    using DType = Catlass::Gemm::GemmType<ElementD, LayoutD>;
 
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;
    using BlockScheduler = typename Catcoc::DGemm::Block::GemmIdentityBlockSwizzleAllGather<7, 1, 2>;
    using BlockRemapper = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
    using CommBlockScheduler = CommEpilogue::Block::BlockCommSwizzle<0>;
 
    using RemoteSrcType = CType;
    using RemoteDstType = DType;
    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, RemoteSrcType, RemoteDstType, CopyDirect::Put>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    constexpr bool isDynamic = true;
    using AllGatherDispatch = CommEpilogue::EpilogueAtlasA2CommToShareMem<UB_STAGES,
        Catcoc::detail::CopyMode::Gather, isDynamic>;
    using BlockEpilogueAllGather = CommEpilogue::Block::CommBlockEpilogue<
        AllGatherDispatch,
        RemoteSrcType, RemoteDstType,
        void,
        void,
        void, TileRemoteCopy, TileScheduler,
        BlockRemapper
    >;
 
    using AllGatherMatmulKernel = DGemm::Kernel::AllGatherMatmul<
        BlockMmad,
        BlockEpilogueAllGather,
        BlockScheduler,
        CommBlockScheduler,
        WORKSPACE_STAGES
    >;
 
    // Prepare params
    uint32_t rank = shmem_my_pe();
    uint32_t rankSize = shmem_n_pes();
 
    Catlass::GemmCoord commProblemShape{problemShape.m(), problemShape.k(), problemShape.k()};
    BlockRemapper remapper(commProblemShape, 
        Catlass::MakeCoord(L1TileShape::M, problemShape.k()));
 
    typename BlockEpilogueAllGather::Params allGatherParams{
        reinterpret_cast<__gm__ ElementC *>(symmetricPtr),
        layoutD,
        remapper,
        commCoreSplit,
        commBlockShape,
        commTileShape
    };
 
    typename AllGatherMatmulKernel::Params params{
        problemShape,
        rank, rankSize,
        gmA, layoutA,
        gmB, layoutB,
        symmetricPtr,
        allGatherParams,
        gmC, layoutC,
        commInterval
    };
 
    // Call kernel
    AllGatherMatmulKernel allGatherMatmul;
    allGatherMatmul(params);
}
 
template <
    class ElementA, class LayoutA,
    class ElementB, class LayoutB,
    class ElementC, class LayoutC,
    class ElementD, class LayoutD
>
CATLASS_GLOBAL
void AllGatherMatmul(
    uint64_t fftsAddr, GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmC, GM_ADDR symmetricPtr, CocTilingParams cocTiling
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
    uint32_t rankSize = cocTiling.rankSize;
 
    Catlass::GemmCoord problemShape{m, n, k};
    Catlass::GemmCoord l1TileShape{m0, n0, k0};
 
    Catlass::MatrixCoord commCoreSplit{commDataSplit, commNpuSplit};
    Catlass::MatrixCoord commBlockShape{commBlockM, UINT_MAX / 2};
    Catlass::MatrixCoord commTileShape{commTileM / 2, n0};

    uint32_t strideA;
    if constexpr (std::is_same_v<LayoutA, Catlass::layout::RowMajor>) {
        strideA = k;
    } else if constexpr (std::is_same_v<LayoutA, Catlass::layout::ColumnMajor>) {
        strideA = m;
    }

    uint32_t strideB;
    if constexpr (std::is_same_v<LayoutB, Catlass::layout::RowMajor>) {
        strideB = n;
    } else if constexpr (std::is_same_v<LayoutB, Catlass::layout::ColumnMajor>) {
        strideB = k;
    }

    LayoutA layoutA{m, k, strideA};
    LayoutB layoutB{k, n, strideB};
    LayoutC layoutC{m * rankSize, n, n};
    LayoutD layoutD{m0 * commInterval * rankSize * WORKSPACE_STAGES, k, k};
 
    AllGatherMatmulImpl<ArchTag, ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementD, LayoutD>
        (problemShape, l1TileShape, gmA, layoutA, gmB, layoutB, gmC, layoutC, 
         commInterval, commCoreSplit, commBlockShape, commTileShape, symmetricPtr, layoutD
        );
}
 
 
#endif // ALLGATHER_MATMUL_KERNEL_H
