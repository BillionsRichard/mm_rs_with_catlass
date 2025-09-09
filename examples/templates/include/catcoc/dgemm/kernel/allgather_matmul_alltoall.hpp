// Prevent multiple inclusions of the header file
#ifndef CATCOC_DGEMM_KERNEL_ALLGATHER_MATMUL_ALLTOALL_HPP
#define CATCOC_DGEMM_KERNEL_ALLGATHER_MATMUL_ALLTOALL_HPP

// Include dependent headers
#include "catcoc/catcoc.hpp"
// #include "device/shmem_device_rma.h"
#include "catlass/gemm/block/block_mmad_pingpong.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include <type_traits> // For std::is_same_v

namespace Catcoc::DGemm::Kernel {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

// Based on the new design (Matmul with Fused Scatter)
template <
    class BlockMmad_,
    class BlockEpilogueAllGather_,
    class BlockSchedulerForMatmul_,
    class CommScheduler_
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
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using AllGather = BlockEpilogueAllGather_;
    using AllGatherParams = typename AllGather::Params;
    

    using BlockSchedulerForMatmul = BlockSchedulerForMatmul_;
    using CommScheduler = CommScheduler_;
    static constexpr uint32_t WORKSPACE_STAGES = 2;

    struct Params {
        GemmCoord problemShape;
        uint32_t rankIdx;
        uint32_t rankSize;
        int32_t teamIdx;

        GM_ADDR ptrA; LayoutA layoutA;
        GM_ADDR ptrB; LayoutB layoutB;
        GM_ADDR ptrC; LayoutC layoutC;
        GM_ADDR ptrSymmetric;

        AllGatherParams allGatherParams;

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
            uint32_t commInterval_
        ) : problemShape(problemShape_),
            rankIdx(rank_), rankSize(rankSize_), teamIdx(teamIdx_),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrB(ptrB_), layoutB(layoutB_),
            ptrC(ptrC_), layoutC(layoutC_),
            ptrSymmetric(ptrSymmetric_),
            allGatherParams(allGatherParams_),
            commInterval(commInterval_) {}
    };

    struct Workspace {
        GM_ADDR ptr_ag_out;
        GM_ADDR ptr_scatter_out;
        GM_ADDR ptr_tmp_buffer;
        
        uint32_t ag_out_size_per_stage;
        // Size of the buffer for one (dest_rank, src_rank) pair
        uint32_t scatter_chunk_size_per_stage;

        CATLASS_DEVICE
        Workspace(Params const& params) {
            uint32_t K = params.problemShape.k();
            uint32_t N = params.problemShape.n();
            uint32_t N_per_rank = N / params.rankSize;
            uint32_t commSizeM = params.commInterval * L1TileShape::M;

            ag_out_size_per_stage = params.rankSize * commSizeM * K;
            scatter_chunk_size_per_stage = commSizeM * N_per_rank;

            ptr_ag_out = params.ptrSymmetric;
            
            uint32_t scatter_out_offset = WORKSPACE_STAGES * ag_out_size_per_stage * sizeof(ElementA);
            ptr_scatter_out = params.ptrSymmetric + scatter_out_offset;

            uint32_t tmp_buffer_offset = scatter_out_offset + (WORKSPACE_STAGES * params.rankSize * params.rankSize * scatter_chunk_size_per_stage * sizeof(ElementC));
            ptr_tmp_buffer = params.ptrSymmetric + tmp_buffer_offset;
        }

        CATLASS_DEVICE GM_ADDR GetAgOut(uint32_t stageId) {
            return ptr_ag_out + stageId * ag_out_size_per_stage * sizeof(ElementA);
        }

        // Gets the pointer to the buffer for (dest_rank, src_rank)
        CATLASS_DEVICE GM_ADDR GetScatterChunk(uint32_t stageId, uint32_t dest_rank, uint32_t src_rank, uint32_t rankSize) {
            uint32_t stage_offset = stageId * (rankSize * rankSize * scatter_chunk_size_per_stage);
            uint32_t dest_rank_offset = dest_rank * (rankSize * scatter_chunk_size_per_stage);
            uint32_t src_rank_offset = src_rank * scatter_chunk_size_per_stage;
            return ptr_scatter_out + (stage_offset + dest_rank_offset + src_rank_offset) * sizeof(ElementC);
        }
    };

    CATLASS_DEVICE AllGatherMatmulAlltoall() {
        for (uint32_t i = 0; i < WORKSPACE_STAGES; ++i) {
            flagAivFinishAllGather[i] = Catlass::Arch::CrossCoreFlag(i);
            flagAicFinishMatmulScatter[i] = Catlass::Arch::CrossCoreFlag(i);
        }
        flagAivFinish = Catlass::Arch::CrossCoreFlag(WORKSPACE_STAGES);
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params &params) {
        Workspace workspace(params);
        uint32_t M = params.problemShape.m();
        uint32_t K = params.problemShape.k();
        uint32_t N = params.problemShape.n();
        uint32_t N_per_rank = N / params.rankSize;
        uint32_t my_rank_i = params.rankIdx;

        uint32_t commSizeM = params.commInterval * L1TileShape::M;
        uint32_t commLoops = CeilDiv(M, commSizeM);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;
            uint32_t actualCommSizeM = Min(commSizeM, M - commIdx * commSizeM);
            
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishAllGather[stageId]);
            
            for (uint32_t dest_rank_j = 0; dest_rank_j < params.rankSize; ++dest_rank_j) {
                GM_ADDR ptr_A_j_base = workspace.GetAgOut(stageId);
                GM_ADDR ptr_A_j = ptr_A_j_base + dest_rank_j * actualCommSizeM * K * sizeof(ElementA);
                auto layout_A_j = Catlass::layout::RowMajor(actualCommSizeM, K);
                AscendC::GlobalTensor<ElementA> smem_a_j;
                smem_a_j.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(ptr_A_j));

                AscendC::GlobalTensor<ElementB> gmB;
                gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));

                GM_ADDR ptr_scatter_dest = workspace.GetScatterChunk(stageId, dest_rank_j, my_rank_i, params.rankSize);
                auto layout_scatter_dest = Catlass::layout::RowMajor(actualCommSizeM, N_per_rank);
                AscendC::GlobalTensor<ElementC> smem_scatter_dest;
                smem_scatter_dest.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(ptr_scatter_dest));

                BlockMmad blockMmad(resource);
                GemmCoord problem_shape_ji = {actualCommSizeM, N_per_rank, K};
                Catlass::Gemm::Block::GemmIdentityBlockSwizzle<> scheduler(problem_shape_ji, L1TileShape::ToCoordMN());
                
                uint32_t aicoreIdx = AscendC::GetBlockIdx();
                uint32_t aicoreNum = AscendC::GetBlockNum();
                uint32_t matmul_loops = scheduler.GetCoreLoops();

                for (uint32_t i = aicoreIdx; i < matmul_loops; i += aicoreNum) {
                    GemmCoord block_coord = scheduler.GetBlockCoord(i);
                    GemmCoord actual_block_shape = scheduler.GetActualBlockShape(block_coord);
                    GemmCoord offset_coord = block_coord * L1TileShape::ToCoord();

                    int64_t offsetA = layout_A_j.GetOffset(offset_coord.GetCoordMK());
                    int64_t offsetB = params.layoutB.GetOffset(offset_coord.GetCoordKN());
                    int64_t offsetC = layout_scatter_dest.GetOffset(offset_coord.GetCoordMN());

                    blockMmad(
                        smem_a_j[offsetA], layout_A_j,
                        gmB[offsetB], params.layoutB,
                        smem_scatter_dest[offsetC], layout_scatter_dest,
                        actual_block_shape
                    );
                }
            }
            
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishMatmulScatter[stageId]);
        }
        
        Catlass::Arch::CrossCoreBarrier<0, PIPE_FIX>();
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAivFinish);
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params &params) {
        Workspace workspace(params);
        uint32_t M = params.problemShape.m();
        uint32_t K = params.problemShape.k();
        uint32_t N = params.problemShape.n();
        uint32_t N_per_rank = N / params.rankSize;

        uint32_t commSizeM = params.commInterval * L1TileShape::M;
        uint32_t commLoops = CeilDiv(M, commSizeM);

        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;

            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishMatmulScatter[stageId]);
            }
            
            shmemx_barrier_all_vec();

            uint32_t actualCommSizeM = Min(commSizeM, M - commIdx * commSizeM);
            auto actualCommShape = DistMatrixCoord(actualCommSizeM, K, params.rankSize);
            // --- 1. AllGather ---
            {
                AllGather allGather(resource, params.allGatherParams);
                MatrixCoord commBlockShape = params.allGatherParams.BlockShape();
                MatrixCoord commCoreSplit = params.allGatherParams.CoreSplit();
                CommScheduler commScheduler(commBlockShape, commCoreSplit);
                MatrixCoord loopsInRank = CeilDiv(MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);
                commScheduler.UpdateProblem(actualCommShape, loopsInRank);
                auto commAicoreNum = commScheduler.GetRealCore();
                auto commCoreLoops = commScheduler.GetCoreLoop();
                MatrixCoord commSrcOffset{commIdx * commSizeM, 0};
                
                AscendC::GlobalTensor<ElementA> gmSymmetric;
                gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(workspace.GetAgOut(stageId)));
                auto layoutSymmetric = Catlass::layout::RowMajor(params.rankSize * actualCommSizeM, K);

                allGather.InitBlockLoop();
                uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
                uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
                if (subcoreIdx == 0 && aicoreIdx < commAicoreNum) {
                    for (uint32_t loopIdx = aicoreIdx; loopIdx < commCoreLoops; loopIdx += commAicoreNum) {
                        DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(loopIdx);
                        MatrixCoord blockOffsetInRank = commScheduler.GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                        MatrixCoord actualCommBlockShape = commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);
                        uint32_t remoteRankIdx = commBlockCoord.rank();
                        auto offsetSrc = commSrcOffset + blockOffsetInRank;
                        MatrixCoord commDstOffset{remoteRankIdx * actualCommSizeM, 0};
                        auto offsetDst = commDstOffset + blockOffsetInRank;
                        AscendC::GlobalTensor<ElementA> gmA;
                        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
                        auto gmBlockSrc = gmA[params.layoutA.GetOffset(offsetSrc)];
                        auto layoutBlockSrc = params.layoutA.GetTileLayout(actualCommBlockShape);
                        auto gmBlockDst = gmSymmetric[layoutSymmetric.GetOffset(offsetDst)];
                        auto layoutBlockDst = layoutSymmetric.GetTileLayout(actualCommBlockShape);
                        allGather(gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst, actualCommBlockShape, remoteRankIdx);
                    }
                }
                allGather.FinalizeBlockLoop();
            }

            shmemx_barrier_all_vec();
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishAllGather[stageId]);

            // --- Wait for AIC Matmul-Scatter ---
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishMatmulScatter[stageId]);
            shmemx_barrier_all_vec();

            // --- Final Assembly ---
            {
                uint32_t my_rank_j = params.rankIdx;
                uint32_t actualCommSizeM = Min(commSizeM, M - commIdx * commSizeM);
                AscendC::GlobalTensor<ElementC> gmC;
                gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
                uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
                uint32_t aicoreNum = AscendC::GetBlockNum();

                for (uint32_t src_rank_i = 0; src_rank_i < params.rankSize; ++src_rank_i) {
                    GM_ADDR src_ptr_base = workspace.GetScatterChunk(stageId, my_rank_j, src_rank_i, params.rankSize);
                    auto layout_src = Catlass::layout::RowMajor(actualCommSizeM, N_per_rank);
                    MatrixCoord dst_chunk_offset = {commIdx * commSizeM, src_rank_i * N_per_rank};

                    if (src_rank_i == my_rank_j) {
                        // Local data, copy directly from shmem to gmem.
                        __gm__ ElementC* src_ptr = reinterpret_cast<__gm__ ElementC*>(src_ptr_base);
                        for(uint32_t m = aicoreIdx; m < actualCommSizeM; m += aicoreNum) {
                            for(uint32_t n = 0; n < N_per_rank; ++n) {
                                MatrixCoord src_coord = {m, n};
                                MatrixCoord dst_coord = dst_chunk_offset + src_coord;
                                gmC.SetValue(params.layoutC.GetOffset(dst_coord), src_ptr[layout_src.GetOffset(src_coord)]);
                            }
                        }
                    } else {
                        // Remote data, pull from remote rank's shmem into a temporary local shmem buffer, then copy to gmem.
                        __gm__ ElementC* tmp_buffer_ptr = reinterpret_cast<__gm__ ElementC*>(workspace.ptr_tmp_buffer);
                        
                        if constexpr (std::is_same_v<ElementC, half>) {
                             shmem_get_half_mem_nbi(
                                tmp_buffer_ptr,
                                reinterpret_cast<__gm__ half*>(src_ptr_base),
                                actualCommSizeM * N_per_rank,
                                src_rank_i
                            );
                        }
                        
                        shmemx_barrier_all_vec();

                        for(uint32_t m = aicoreIdx; m < actualCommSizeM; m += aicoreNum) {
                            for(uint32_t n = 0; n < N_per_rank; ++n) {
                                MatrixCoord src_coord = {m, n};
                                MatrixCoord dst_coord = dst_chunk_offset + src_coord;
                                gmC.SetValue(params.layoutC.GetOffset(dst_coord), tmp_buffer_ptr[layout_src.GetOffset(src_coord)]);
                            }
                        }
                    }
                }
            }
        }

        Catlass::Arch::CrossCoreWaitFlag(flagAivFinish);
        Catlass::Arch::CrossCoreBarrier<0, PIPE_MTE3>();

        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // --- Member Variables ---
    Catlass::Arch::CrossCoreFlag flagAivFinishAllGather[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAicFinishMatmulScatter[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinish;
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catcoc::DGemm::Kernel

#endif // CATCOC_DGEMM_KERNEL_ALLGATHER_MATMUL_ALLTOALL_HPP
