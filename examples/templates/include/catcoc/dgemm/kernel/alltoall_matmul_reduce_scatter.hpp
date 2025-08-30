#ifndef CATCOC_DGEMM_KERNEL_ALLTOALL_MATMUL_REDUCE_SCATTER_HPP
#define CATCOC_DGEMM_KERNEL_ALLTOALL_MATMUL_REDUCE_SCATTER_HPP

#include "catcoc/catcoc.hpp"

#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catcoc/dist_coord.hpp"

namespace Catcoc::DGemm::Kernel {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;
using Catcoc::DistMatrixCoord;

template <
    class BlockMmad,
    class BlockEpilogueAlltoall,
    class BlockEpilogueReduceScatter,
    class BlockMmadScheduler,
    class BlockCommScheduler,
    uint32_t WORKSPACE_STAGES
>
class AlltoallMatmulReduceScatter {
public:
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC; // Matmul accumulator type
    using ElementD = typename BlockEpilogueReduceScatter::ElementDst; // Final output type
    using LayoutD = typename BlockEpilogueReduceScatter::LayoutDst;

    using BlockEpilogueAlltoallParams = typename BlockEpilogueAlltoall::Params;
    using BlockEpilogueReduceScatterParams = typename BlockEpilogueReduceScatter::Params;

    struct Params {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        int32_t teamIdx;
        __gm__ ElementA *ptrA; LayoutA layoutA;
        __gm__ ElementB *ptrB; LayoutB layoutB;
        __gm__ ElementD *ptrD; LayoutD layoutD;
        GM_ADDR ptrSymmetric;
        BlockEpilogueAlltoallParams alltoallParams;
        BlockEpilogueReduceScatterParams reduceScatterParams;
        CATLASS_DEVICE Params() = default;
        CATLASS_DEVICE Params( GemmCoord const &problemShape_, uint32_t rankIdx_, uint32_t rankSize_, int32_t teamIdx_,
            GM_ADDR ptrA_, LayoutA const &layoutA_, GM_ADDR ptrB_, LayoutB const &layoutB_, GM_ADDR ptrD_, LayoutD const &layoutD_,
            GM_ADDR ptrSymmetric_, BlockEpilogueAlltoallParams const &alltoallParams_,
            BlockEpilogueReduceScatterParams const &reduceScatterParams_
        ) : problemShape(problemShape_), rankIdx(rankIdx_), rankSize(rankSize_), teamIdx(teamIdx_),
            ptrA(reinterpret_cast<__gm__ ElementA *>(ptrA_)), layoutA(layoutA_),
            ptrB(reinterpret_cast<__gm__ ElementB *>(ptrB_)), layoutB(layoutB_),
            ptrD(reinterpret_cast<__gm__ ElementD *>(ptrD_)), layoutD(layoutD_),
            ptrSymmetric(ptrSymmetric_), alltoallParams(alltoallParams_), reduceScatterParams(reduceScatterParams_) {}
    };

    CATLASS_DEVICE AlltoallMatmulReduceScatter() {
        flagAivFinishAlltoall = Catlass::Arch::CrossCoreFlag(0);
        flagAicFinishMatmul = Catlass::Arch::CrossCoreFlag(1);
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const &params) {
        uint32_t aic_idx = AscendC::GetBlockIdx();
        uint32_t aic_num = AscendC::GetBlockNum();
        Catlass::Arch::CrossCoreWaitFlag(flagAivFinishAlltoall);
        
        GemmCoord matmul_problem_shape = {params.problemShape.m(), params.problemShape.n(), params.problemShape.k() / params.rankSize};
        BlockMmadScheduler mmad_scheduler(matmul_problem_shape, L1TileShape::ToCoordMN());
        uint32_t core_loops = mmad_scheduler.GetCoreLoops();
        BlockMmad block_mmad(resource);

        __gm__ ElementA* workspace_A_prime = reinterpret_cast<__gm__ ElementA*>(params.ptrSymmetric);
        auto layout_A_prime = Catlass::layout::RowMajor{matmul_problem_shape.m(), matmul_problem_shape.k()};
        
        uint64_t P_offset = (uint64_t)matmul_problem_shape.m() * matmul_problem_shape.k() * sizeof(ElementA);
        __gm__ ElementC* workspace_P = reinterpret_cast<__gm__ ElementC*>((__gm__ uint8_t*)params.ptrSymmetric + P_offset);
        auto layout_P = Catlass::layout::RowMajor{matmul_problem_shape.m(), matmul_problem_shape.n()};

        for (uint32_t loop_idx = aic_idx; loop_idx < core_loops; loop_idx += aic_num) {
            GemmCoord block_coord = mmad_scheduler.GetBlockCoord(loop_idx);
            GemmCoord actual_block_shape = mmad_scheduler.GetActualBlockShape(block_coord);
            GemmCoord offset_coord = block_coord * L1TileShape::ToCoord();
            
            auto gm_block_A = workspace_A_prime[layout_A_prime.GetOffset(offset_coord.GetCoordMK())];
            auto gm_block_B = params.ptrB[params.layoutB.GetOffset(offset_coord.GetCoordKN())];
            auto gm_block_C = workspace_P[layout_P.GetOffset(offset_coord.GetCoordMN())];

            block_mmad(gm_block_A, layout_A_prime, gm_block_B, params.layoutB, gm_block_C, layout_P, actual_block_shape);
        }
        
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishMatmul);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params const &params) {
        // Phase 1: All-to-All (Get-based)
        BlockEpilogueAlltoall alltoall_epilogue(resource, params.alltoallParams);
        __gm__ ElementA* workspace_A_prime = reinterpret_cast<__gm__ ElementA*>(params.ptrSymmetric);
        auto layout_A_prime = Catlass::layout::RowMajor{params.problemShape.m(), params.problemShape.k() / params.rankSize};

        // Each rank (dst_rank) gets data from all other ranks (src_rank)
        // This simplified loop illustrates the logic. A full implementation would use a scheduler.
        for (uint32_t src_rank = 0; src_rank < params.rankSize; ++src_rank) {
            uint32_t dst_rank = params.rankIdx;
            MatrixCoord copy_shape = {params.problemShape.m() / params.rankSize, params.problemShape.k() / params.rankSize};
            
            // Source is on remote rank `src_rank`. It's the `dst_rank`-th column slice of their A.
            MatrixCoord src_offset_on_remote = {0, dst_rank * copy_shape.column()};
            
            // Destination is my local workspace. The data from `src_rank` becomes my `src_rank`-th row slice.
            MatrixCoord dst_offset_in_local = {src_rank * copy_shape.row(), 0};

            auto gm_dst_tile = workspace_A_prime[layout_A_prime.GetOffset(dst_offset_in_local)];
            auto layout_dst_tile = layout_A_prime.GetTileLayout(copy_shape);
            
            alltoall_epilogue(
                gm_dst_tile, layout_dst_tile, 
                params.ptrA, params.layoutA, src_offset_on_remote,
                copy_shape, src_rank, params.teamIdx
            );
        }
        shmemx_barrier_all_vec();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishAlltoall);

        // Phase 2: Wait for Matmul
        Catlass::Arch::CrossCoreWaitFlag(flagAicFinishMatmul);
        shmemx_barrier_all_vec();

        // Phase 3: Reduce-Scatter (Get-based)
        BlockEpilogueReduceScatter reduce_scatter_epilogue(resource, params.reduceScatterParams);
        uint64_t P_offset = (uint64_t)params.problemShape.m() * (params.problemShape.k() / params.rankSize) * sizeof(ElementA);
        __gm__ ElementC* workspace_P = reinterpret_cast<__gm__ ElementC*>((__gm__ uint8_t*)params.ptrSymmetric + P_offset);
        auto layout_P = Catlass::layout::RowMajor{params.problemShape.m(), params.problemShape.n()};
        
        MatrixCoord my_out_offset_base = {params.rankIdx * (params.problemShape.m() / params.rankSize), 0};
        
        AscendC::SetAtomicAdd<ElementD>();
        AscendC::PipeBarrier<PIPE_ALL>();
        // Simplified loop, a real implementation would use a scheduler.
        for (uint32_t remote_rank = 0; remote_rank < params.rankSize; ++remote_rank) {
            MatrixCoord copy_shape = {params.problemShape.m() / params.rankSize, params.problemShape.n()};
            
            MatrixCoord src_offset_on_remote = my_out_offset_base;

            auto gm_dst_tile = params.ptrD[params.layoutD.GetOffset(my_out_offset_base)];
            auto layout_dst_tile = params.layoutD.GetTileLayout(copy_shape);
            
            reduce_scatter_epilogue(
                gm_dst_tile, layout_dst_tile,
                workspace_P, layout_P, src_offset_on_remote,
                copy_shape, remote_rank, params.teamIdx
            );
        }
        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_ALL>();
        shmemx_barrier_all_vec();
    }

private:
    Catlass::Arch::CrossCoreFlag flagAivFinishAlltoall;
    Catlass::Arch::CrossCoreFlag flagAicFinishMatmul;
    Catlass::Arch::Resource<ArchTag> resource;
}; 

}  // namespace Catcoc::DGemm::Kernel
#endif  // CATCOC_DGEMM_KERNEL_ALLTOALL_MATMUL_REDUCE_SCATTER_HPP