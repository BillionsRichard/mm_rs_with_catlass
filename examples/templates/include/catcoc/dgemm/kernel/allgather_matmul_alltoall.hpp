// Prevent multiple inclusions of the header file
#ifndef CATCOC_DGEMM_KERNEL_ALLGATHER_MATMUL_ALLTOALL_HPP
#define CATCOC_DGEMM_KERNEL_ALLGATHER_MATMUL_ALLTOALL_HPP

// Include dependent headers
#include "catcoc/catcoc.hpp"
#include "catcoc/gemm/block/block_mmad_pingpong.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catcoc::DGemm::Kernel {

// Use type aliases to simplify code
using Catlass::MatrixCoord;
using Catlass::GemmCoord;

//
// AllGatherMatmulAlltoall is a kernel implementation of a fused operator.
// It fuses AllGather, Matmul, and Alltoall operations.
//
template <
    class BlockMmad_,
    class BlockEpilogueAllGather_,
    class BlockSchedulerForMatmul_,
    class BlockEpilogueAlltoall_,
    class CommScheduler_,
    uint32_t WORKSPACE_STAGES_
>
class AllGatherMatmulAlltoall {
public:
    // --- Type Alias Definitions ---
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC; // Matmul result type
    using LayoutC = typename BlockMmad::LayoutC; // Final output layout

    using AllGather = BlockEpilogueAllGather_;
    using AllGatherParams = typename AllGather::Params;
    using Alltoall = BlockEpilogueAlltoall_;
    using AlltoallParams = typename Alltoall::Params;

    using BlockSchedulerForMatmul = BlockSchedulerForMatmul_;
    using CommScheduler = CommScheduler_;
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    //
    // Params struct: used to pass all the necessary parameters for the operator from the host side
    //
    struct Params {
        GemmCoord problemShape; // Logical problem shape (M, N, K)
        uint32_t rankIdx;
        uint32_t rankSize;
        int32_t teamIdx;

        GM_ADDR ptrA; LayoutA layoutA;
        GM_ADDR ptrB; LayoutB layoutB;
        GM_ADDR ptrC; LayoutC layoutC; // Final output
        GM_ADDR ptrSymmetric;          // Workspace for communication

        AllGatherParams allGatherParams;
        AlltoallParams alltoallParams;

        uint32_t commInterval;

        CATLASS_DEVICE Params() {}
        CATLASS_DEVICE Params(
            GemmCoord const &problemShape_,
            uint32_t rank_, uint32_t rankSize_, int32_t teamIdx_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrC_, LayoutC const &layoutC_,
            GM_ADDR ptrSymmetric_,
            AllGatherParams const &allGatherParams_,
            AlltoallParams const &alltoallParams_,
            uint32_t commInterval_
        ) : problemShape(problemShape_),
            rankIdx(rank_), rankSize(rankSize_), teamIdx(teamIdx_),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrB(ptrB_), layoutB(layoutB_),
            ptrC(ptrC_), layoutC(layoutC_),
            ptrSymmetric(ptrSymmetric_),
            allGatherParams(allGatherParams_),
            alltoallParams(alltoallParams_),
            commInterval(commInterval_) {}
    };

    //
    // Kernel constructor: initializes Flags for inter-core synchronization
    //
    CATLASS_DEVICE AllGatherMatmulAlltoall() {
        for (uint32_t i = 0; i < WORKSPACE_STAGES; ++i) {
            flagAicFinishMma[i] = Catlass::Arch::CrossCoreFlag(i);
            flagAivFinishAllGather[i] = Catlass::Arch::CrossCoreFlag(i);
            flagAivFinishAlltoall[i] = Catlass::Arch::CrossCoreFlag(i);
        }
    }

    //
    // Kernel execution entry point
    //
    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params &params);

    //
    // Kernel implementation for AIC (AI Core): responsible for Matmul
    //
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params &params) {
        // Problem dimensions
        uint32_t M = params.problemShape.m();
        uint32_t N = params.problemShape.n();
        uint32_t K = params.problemShape.k();
        uint32_t N_per_rank = N / params.rankSize;

        // Core setup
        uint32_t aicoreIndex = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();

        // Global memory tensor for B
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));

        // Symmetric memory workspace setup (must match AIV)
        uint8_t* smem_base_ptr = reinterpret_cast<uint8_t*>(params.ptrSymmetric);
        AscendC::GlobalTensor<ElementA> smem_allgather_out;
        smem_allgather_out.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(smem_base_ptr));
        auto layout_smem_ag_out = Catlass::layout::RowMajor{M * params.rankSize, K};

        uint32_t matmul_out_offset = M * K * params.rankSize * sizeof(ElementA);
        AscendC::GlobalTensor<ElementC> smem_matmul_out;
        smem_matmul_out.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(smem_base_ptr + matmul_out_offset));
        auto layout_smem_mm_out = Catlass::layout::RowMajor{M * params.rankSize, N_per_rank};

        // --- Main Loop Logic ---
        // Wait for AllGather to be finished by AIV
        Catlass::Arch::CrossCoreWaitFlag(flagAivFinishAllGather[0]);

        // Batched Matmul
        BlockMmad blockMmad(resource);
        GemmCoord problem_shape_matmul = {M * params.rankSize, N_per_rank, K};
        GemmCoord block_shape_matmul = L1TileShape::ToCoord();
        BlockSchedulerForMatmul matmul_scheduler(problem_shape_matmul, block_shape_matmul.GetCoordMN());
        uint32_t matmul_loops = matmul_scheduler.GetCoreLoops();

        for (uint32_t i = aicoreIndex; i < matmul_loops; i += aicoreNum) {
            GemmCoord block_coord = matmul_scheduler.GetBlockCoord(i);
            GemmCoord actual_block_shape = matmul_scheduler.GetActualBlockShape(block_coord);
            GemmCoord offset_coord = block_coord * block_shape_matmul;

            // Source A is from the all-gathered buffer in symmetric memory
            int64_t offsetA = layout_smem_ag_out.GetOffset(offset_coord.GetCoordMK());
            // Source B is from the local weight matrix in global memory
            int64_t offsetB = params.layoutB.GetOffset(offset_coord.GetCoordKN());
            // Destination C is the matmul output buffer in symmetric memory
            int64_t offsetC = layout_smem_mm_out.GetOffset(offset_coord.GetCoordMN());

            blockMmad(
                smem_allgather_out[offsetA], layout_smem_ag_out,
                gmB[offsetB], params.layoutB,
                smem_matmul_out[offsetC], layout_smem_mm_out,
                actual_block_shape
            );
        }

        // Signal AIV that Matmul is complete
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishMma[0]);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    //
    // Kernel implementation for AIV (AI Vector): responsible for communication and data movement
    //
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params &params) {
        // Problem dimensions
        uint32_t M = params.problemShape.m();
        uint32_t N = params.problemShape.n();
        uint32_t K = params.problemShape.k();
        uint32_t N_per_rank = N / params.rankSize;

        // Core and scheduler setup
        uint32_t aivIndex = AscendC::GetSubBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

        // Global memory tensors
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));

        // Symmetric memory workspace setup
        AscendC::GlobalTensor<ElementA> smem_allgather_out;
        smem_allgather_out.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrSymmetric));
        auto layout_smem_ag_out = Catlass::layout::RowMajor{M * params.rankSize, K};

        // The rest of the workspace is for matmul_out, transpose_out, alltoall_out
        // We need to manage offsets carefully.
        uint8_t* smem_base_ptr = reinterpret_cast<uint8_t*>(params.ptrSymmetric);
        uint32_t matmul_out_offset = M * K * params.rankSize * sizeof(ElementA);
        AscendC::GlobalTensor<ElementC> smem_matmul_out;
        smem_matmul_out.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(smem_base_ptr + matmul_out_offset));
        auto layout_smem_mm_out = Catlass::layout::RowMajor{M * params.rankSize, N_per_rank};

        // --- Main Loop Logic ---
        // For now, we assume a simple, non-pipelined execution for clarity.
        // A full implementation would use the WORKSPACE_STAGES for pipelining.

        // 1. AllGather
        // The AIV cores collectively perform AllGather on matrix A.
        AllGather allgather_op(resource, params.allGatherParams);

        // All AIV cores must participate in the AllGather.
        // We can use a simple loop over the M dimension, where each core handles a slice.
        // This is a simplified view; a real implementation uses a scheduler.
        MatrixCoord allgather_problem_shape = {M, K};
        typename AllGather::EpilogueTileSwizzle allgather_swizzle(allgather_problem_shape, AllGather::TileShape::ToCoord());
        uint32_t allgather_loops = allgather_swizzle.GetLoops();

        for(uint32_t i = aicoreIndex; i < allgather_loops; i += aicoreNum) {
             auto tileCoord = allgather_swizzle.GetTileCoord(i);
             auto actualTileShape = allgather_swizzle.GetActualTileShape(tileCoord);
             auto ag_offset = tileCoord * AllGather::TileShape::ToCoord();

             // The source is the local slice of A
             auto gmBlockSrc = gmA[params.layoutA.GetOffset(ag_offset)];
             auto layoutBlockSrc = params.layoutA.GetTileLayout(actualTileShape);

             // The destination is the symmetric memory buffer
             auto gmBlockDst = smem_allgather_out[layout_smem_ag_out.GetOffset(ag_offset)];
             auto layoutBlockDst = layout_smem_ag_out.GetTileLayout(actualTileShape);

             allgather_op(gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst, actualTileShape, params.rankIdx, params.teamIdx);
        }

        // Synchronize all ranks and cores after AllGather
        shmemx_barrier_all_vec();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishAllGather[0]);

        // 2. Wait for Matmul (from AIC)
        Catlass::Arch::CrossCoreWaitFlag(flagAicFinishMma[0]);
        shmemx_barrier_all_vec(); // Ensure all ranks have finished matmul before transpose

        // 3. Transpose the matmul output
        // Shape in: [rankSize, M, N/rankSize], Shape out: [rankSize, N/rankSize, M]
        uint32_t transpose_out_offset = matmul_out_offset + M * N * sizeof(ElementC);
        AscendC::GlobalTensor<ElementC> smem_transpose_out;
        smem_transpose_out.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(smem_base_ptr + transpose_out_offset));
        
        // This is a simplified transpose, a real implementation would be more optimized.
        // We iterate through the source tensor and copy elements to the destination with swapped indices.
        for (uint32_t r = 0; r < params.rankSize; ++r) {
            for (uint32_t m_idx = aicoreIndex; m_idx < M; m_idx += aicoreNum) {
                for (uint32_t n_idx = 0; n_idx < N_per_rank; ++n_idx) {
                    // src_idx is for [r, m_idx, n_idx] in a [rankSize*M, N_per_rank] layout
                    int64_t src_idx = (r * M + m_idx) * N_per_rank + n_idx;
                    // dst_idx is for [r, n_idx, m_idx] in a [rankSize*N_per_rank, M] layout
                    int64_t dst_idx = (r * N_per_rank + n_idx) * M + m_idx;
                    smem_transpose_out[dst_idx] = smem_matmul_out[src_idx];
                }
            }
        }
        shmemx_barrier_all_vec();

        // 4. Alltoall
        uint32_t alltoall_out_offset = transpose_out_offset + M * N * sizeof(ElementC);
        AscendC::GlobalTensor<ElementC> smem_alltoall_out;
        smem_alltoall_out.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(smem_base_ptr + alltoall_out_offset));

        Alltoall alltoall_op(resource, params.alltoallParams);
        // The data is already in [rankSize, N_per_rank, M] layout in smem_transpose_out.
        // Alltoall will exchange the first two dimensions.
        // Simplified loop, a real one would use a scheduler.
        for (uint32_t i = aicoreIndex; i < (N_per_rank * M); i += aicoreNum) {
            alltoall_op.Exchange(
                smem_transpose_out[i], // src
                smem_alltoall_out[i],  // dst
                params.rankIdx,
                params.teamIdx
            );
        }
        shmemx_barrier_all_vec();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishAlltoall[0]);
        
        // 5. Final copy to C
        // The output of alltoall is in [rankSize, N_per_rank, M], which can be viewed as [N, M]
        // We need to transpose it to [M, N] for the final output C.
        auto layout_alltoall_out = Catlass::layout::RowMajor{N, M};
        for (uint32_t n_idx = aicoreIndex; n_idx < N; n_idx += aicoreNum) {
            for (uint32_t m_idx = 0; m_idx < M; ++m_idx) {
                int64_t src_idx = n_idx * M + m_idx;
                int64_t dst_idx = m_idx * N + n_idx;
                gmC[dst_idx] = smem_alltoall_out[src_idx];
            }
        }

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // --- Member Variables ---
    Catlass::Arch::CrossCoreFlag flagAicFinishMma[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishAllGather[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishAlltoall[WORKSPACE_STAGES];
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catcoc::DGemm::Kernel

#endif // CATCOC_DGEMM_KERNEL_MATMUL_REDUCE_SCATTER_DEQUANT_HPP
