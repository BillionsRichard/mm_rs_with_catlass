// Prevent multiple inclusions of the header file
#ifndef CATCOC_DGEMM_KERNEL_QUANT_MATMUL_REDUCE_SCATTER_HPP
#define CATCOC_DGEMM_KERNEL_QUANT_MATMUL_REDUCE_SCATTER_HPP

// Include dependent headers
#include "catcoc/catcoc.hpp"                     // Core header file for the catcoc library, may contain communication primitives, etc.
#include "catlass/arch/resource.hpp"             // Definitions for hardware resource management in the catlass library
#include "catlass/arch/cross_core_sync.hpp"      // Tools for inter-core synchronization in the catlass library, such as Flag
#include "catlass/gemm_coord.hpp"                // Structures for representing GEMM (General Matrix Multiplication) related coordinates in the catlass library
#include "catlass/matrix_coord.hpp"              // Structures for representing matrix coordinates in the catlass library

namespace Catcoc::DGemm::Kernel {

// Use type aliases to simplify code
using Catlass::MatrixCoord;
using Catlass::GemmCoord;

//
// QuantMatmulReduceScatter is a kernel implementation of a fused operator.
// It fuses a quantized matrix multiplication (QuantMatmul) with a communication operation (Reduce-Scatter).
template <
    class BlockMmad_,                // The basic computation unit for matrix multiplication (MMAD)
    class BlockEpilogueReduceScatter_, // Epilogue implementation for the Reduce-Scatter operation
    class BlockEpilogueDequant_,       // Epilogue implementation for dequantization
    class BlockScheduler_,           // Scheduler for computation tasks (Blocks)
    class BlockEpilogueScheduler_,     // Scheduler for communication tasks
    uint32_t WORKSPACE_STAGES_      // Number of stages in the pipeline, used to hide data transfer latency
>
class QuantMatmulReduceScatter {
public:
    // --- Type Alias Definitions ---
    // Define template parameters and their internal types as more readable aliases
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape; // The shape of the L1 Tile, which is the basic unit size for hardware computation
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;       // Element type of accumulator C, usually int32
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementBias = typename BlockMmad::ElementBias; // Get Bias type from BlockMmad
    using LayoutBias = typename BlockMmad::LayoutBias;
    using ReduceScatter = BlockEpilogueReduceScatter_;
    using ReduceScatterParams = typename ReduceScatter::Params;
    using Dequant = BlockEpilogueDequant_;
    using DequantParams = typename Dequant::Params;
    using ElementD = bfloat16_t;                          // Element type of the final output D
    using LayoutD = Catlass::layout::RowMajor;
    using BlockScheduler = BlockScheduler_;
    using CommScheduler = BlockEpilogueScheduler_;
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_; // Number of pipeline stages

    //
    // Params struct: used to pass all the necessary parameters for the operator from the host side
    //
    struct Params {
        GemmCoord problemShape; // The shape of the entire problem (M, N, K)
        uint32_t rankIdx;       // The ID of the current compute card (Rank)
        uint32_t rankSize;      // The total number of compute cards
        int32_t teamIdx;

        GM_ADDR ptrA; LayoutA layoutA; // Global memory address and layout of matrix A
        GM_ADDR ptrB; LayoutB layoutB; // Global memory address and layout of matrix B
        GM_ADDR ptrBias; LayoutBias layoutBias; // Global memory address and layout of Bias
        GM_ADDR ptrSymmetric;          // Workspace address of symmetric memory for inter-card communication
        ReduceScatterParams reduceScatterParams; // Parameters for the Reduce-Scatter Epilogue
        DequantParams dequantParams;             // Parameters for the Dequantization Epilogue
        GM_ADDR ptrC_accum; LayoutC layoutC_accum; // Address and layout of the C accumulator result
        GM_ADDR ptrD_out; LayoutD layoutD_out;     // Address and layout of the final output D
        uint32_t commInterval;                     // Communication interval, controls the ratio of computation to communication

        CATLASS_DEVICE Params() {} // Default constructor
        CATLASS_DEVICE Params(     // Parameterized constructor, for host-side initialization
            GemmCoord const &problemShape_,
            uint32_t rank_, uint32_t rankSize_, int32_t teamIdx_,
            GM_ADDR ptrA_, LayoutA const &layoutA_, 
            GM_ADDR ptrB_, LayoutB const &layoutB_, 
            GM_ADDR ptrBias_, LayoutBias const &layoutBias_, 
            GM_ADDR ptrSymmetric_,
            ReduceScatterParams const &reduceScatterParams_, DequantParams const &dequantParams_,
            GM_ADDR ptrC_accum_, LayoutC const &layoutC_accum_, 
            GM_ADDR ptrD_out_, LayoutD const &layoutD_out_, 
            uint32_t commInterval_
        ) : problemShape(problemShape_), 
            rankIdx(rank_), rankSize(rankSize_), teamIdx(teamIdx_),
            ptrA(ptrA_), layoutA(layoutA_), 
            ptrB(ptrB_), layoutB(layoutB_), 
            ptrBias(ptrBias_), layoutBias(layoutBias_), 
            ptrSymmetric(ptrSymmetric_),
            reduceScatterParams(reduceScatterParams_), 
            dequantParams(dequantParams_),
            ptrC_accum(ptrC_accum_), layoutC_accum(layoutC_accum_), 
            ptrD_out(ptrD_out_), layoutD_out(layoutD_out_),
            commInterval(commInterval_) {}
    };

    //
    // Kernel constructor: initializes Flags for inter-core synchronization
    //
    CATLASS_DEVICE QuantMatmulReduceScatter() {
        // Create multiple sets of Flags based on the number of pipeline stages for synchronization between AIC (AI Core) and AIV (AI Vector)
        // AIC is responsible for computation, AIV is responsible for communication and post-processing. They use pipelining to hide latency.
        for (uint32_t i = 0; i < WORKSPACE_STAGES; ++i) {
            flagAicFinishStore[i] = Catlass::Arch::CrossCoreFlag(i);   // Flag indicating that AIC has finished computation and stored the result
            flagAivFinishCompute[i] = Catlass::Arch::CrossCoreFlag(i); // Flag indicating that AIV has finished communication and post-processing
        }
    }

    //
    // Kernel execution entry point: provides different implementations for different types of hardware cores (AIC/AIV) through template specialization
    //
    template <int32_t CORE_TYPE = g_coreType> CATLASS_DEVICE void operator()(Params &params);

    //
    // Kernel implementation for AIC (AI Core): mainly responsible for high-density matrix multiplication computation
    //
    template <> CATLASS_DEVICE void operator()<AscendC::AIC>(Params &params) {
        // Get the ID of the current AI Core and the total number of AI Cores
        uint32_t aicoreIndex = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        // Calculate size parameters related to computation and communication
        uint32_t blockPerComm = aicoreNum * params.commInterval;
        uint32_t blockPerCommInRank = blockPerComm / params.rankSize;

        // Get the shape of the basic computation unit (Block)
        GemmCoord blockShape = L1TileShape::ToCoord();
        // Calculate the problem shape handled by the current Rank (M dimension is partitioned)
        GemmCoord problemShapeInRank = params.problemShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1, 1);
        // Initialize the computation task scheduler
        BlockScheduler matmulBlockScheduler(problemShapeInRank, blockShape.GetCoordMN());
        // Calculate the total number of loops
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops() * params.rankSize;
        uint32_t commLoops = CeilDiv(coreLoops, blockPerComm);

        // Create a matrix multiplication computation object
        BlockMmad blockMmad(resource);

        // Create tensor objects pointing to global memory
        AscendC::GlobalTensor<ElementA> gmA; 
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));

        AscendC::GlobalTensor<ElementB> gmB; 
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));

        AscendC::GlobalTensor<ElementBias> gmBias; 
        gmBias.SetGlobalBuffer(reinterpret_cast<__gm__ ElementBias *>(params.ptrBias));

        AscendC::GlobalTensor<ElementC> gmC_workspace; 
        gmC_workspace.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));

        AscendC::GlobalTensor<ElementC> gmC_accum; 
        gmC_accum.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC_accum));

        // Define the layout of the workspace for storing cross-card computation results
        auto layoutC = Catlass::layout::RowMajor{ WORKSPACE_STAGES * blockPerComm * L1TileShape::M, L1TileShape::N, L1TileShape::N };
        auto layoutCRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, blockPerComm, L1TileShape::M);
        auto layoutCRow = layout::AffineRankN<3>::Packed(layoutCRowLogicShape);

        // --- Main loop: execute computation in a pipelined manner ---
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES; // Calculate the current pipeline stage being used
            // If not in the initial stage, wait for AIV to finish processing data from the previous buffer of the same stage
            if (commIdx >= WORKSPACE_STAGES) { 
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]); 
            }

            // Calculate the actual number of blocks to be processed in the current communication cycle
            uint32_t actualBlockPerComm = (commIdx == commLoops - 1) ? (coreLoops - blockPerComm * commIdx) : blockPerComm;
            uint32_t actualBlockPerCommInRank = actualBlockPerComm / params.rankSize;
            uint32_t commBlockOffsetInRank = commIdx * blockPerCommInRank;

            // Each AI Core processes different computation blocks in a strided manner
            for (uint32_t blockIdxInComm = aicoreIndex; blockIdxInComm < actualBlockPerComm; blockIdxInComm += aicoreNum) {
                // Calculate the logical ID of the current block in the entire problem and its target Rank
                uint32_t loopIdxInRank = commBlockOffsetInRank + blockIdxInComm % actualBlockPerCommInRank;
                uint32_t targetRankIdx = blockIdxInComm / actualBlockPerCommInRank;
                
                // Get the coordinates and actual shape of the current block from the scheduler
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdxInRank);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
                GemmCoord offsetCoord = blockCoord * blockShape;
                
                // Calculate the offset addresses for matrices A and B
                auto rankOffsetA = problemShapeInRank.GetCoordMK() * Catlass::MakeCoord<uint32_t>(targetRankIdx, 0);
                auto blockOffsetA = offsetCoord.GetCoordMK() + rankOffsetA;
                auto blockOffsetB = offsetCoord.GetCoordKN();
                
                // Determine the storage location for the computation result
                MatrixCoord blockOffsetStore;
                AscendC::GlobalTensor<ElementC> gmStore;
                Catlass::layout::RowMajor layoutStore;
                if (targetRankIdx == params.rankIdx) {
                    // If computing for the current Rank, store the result directly into the final accumulator gmC_accum
                    blockOffsetStore = offsetCoord.GetCoordMN();
                    gmStore = gmC_accum;
                    layoutStore = params.layoutC_accum;
                } else {
                    // If computing for another Rank, store the result in the workspace of the symmetric memory gmC_workspace
                    blockOffsetStore = MatrixCoord{layoutCRow(Catlass::MakeCoord<int>(stageId, blockIdxInComm, 0)), 0};
                    gmStore = gmC_workspace;
                    layoutStore = layoutC;
                }
                
                // Get the final linear address offset
                int64_t offsetA = params.layoutA.GetOffset(blockOffsetA);// m*k
                int64_t offsetB = params.layoutB.GetOffset(blockOffsetB);//k*n
                int64_t offsetBias = params.layoutBias.GetOffset(Catlass::MakeCoord(blockOffsetB[1]));
                int64_t offsetStore = layoutStore.GetOffset(blockOffsetStore);
                
                
                // Execute matrix multiplication computation (fused with bias addition)
                // Note: It is assumed here that the BlockMmad used always supports the bias addition interface.
                // The host-side code is responsible for instantiating the correct version of BlockMmad.
                blockMmad( gmA[offsetA], params.layoutA, 
                           gmB[offsetB], params.layoutB, 
                           gmStore[offsetStore], layoutStore, 
                           gmBias[offsetBias], actualBlockShape );
            }
            // Computation is complete, set a flag to notify AIV to start processing
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
        }
        // Final synchronization to ensure all computations are completed
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    //
    // Kernel implementation for AIV (AI Vector): mainly responsible for communication, bias addition, dequantization, and other post-processing operations
    //
    template <> CATLASS_DEVICE void operator()<AscendC::AIV>(Params &params) {
        // ... Initialization code similar to that in AIC is omitted ...
        uint32_t aicoreIndex = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t aivIndex = AscendC::GetSubBlockIdx();
        uint32_t blockPerComm = aicoreNum * params.commInterval;
        uint32_t blockPerCommInRank = blockPerComm / params.rankSize;

        MatrixCoord blockShapeMN = L1TileShape::ToCoordMN();
        GemmCoord problemShapeInRank = params.problemShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1, 1);
        BlockScheduler matmulBlockScheduler(problemShapeInRank, blockShapeMN);
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops() * params.rankSize;
        auto commLoops = CeilDiv(coreLoops, blockPerComm);

        // Initialize the Reduce-Scatter communication object
        ReduceScatter reduceScatter(resource, params.reduceScatterParams);

        AscendC::GlobalTensor<ElementC> gmC_workspace; 
        gmC_workspace.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementC> gmC_accum; 
        gmC_accum.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC_accum));

        MatrixCoord commBlockShape = params.reduceScatterParams.BlockShape();
        MatrixCoord commCoreSplit = params.reduceScatterParams.CoreSplit();
        MatrixCoord commShape = MatrixCoord{blockPerComm, 1} * blockShapeMN;
        MatrixCoord dataLoopsMx = CeilDiv(commShape, commBlockShape);
        uint32_t dLoopsInRank = CeilDiv(dataLoopsMx.row() * dataLoopsMx.column(), params.rankSize);
        // CommScheduler commScheduler(params.rankIdx, params.rankSize, commCoreSplit, commShape, commBlockShape, dLoopsInRank);
        CommScheduler commScheduler(params.rankIdx, params.rankSize, commBlockShape, commCoreSplit);
        MatrixCoord actualCommShapeInRank = commShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1);

        auto layoutCommLogicShape = Catlass::MakeCoord<int>(1, dLoopsInRank, commBlockShape.row());
        auto layoutComm = layout::AffineRankN<3>::Packed(layoutCommLogicShape);

        // --- Main loop: corresponds to the AIC pipeline ---
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;
            // ... Code for handling the boundary case of the last loop is omitted ...
            // if (commIdx == commLoops - 1) {
            //     uint32_t actualBlockInComm = coreLoops - commIdx * blockPerComm;
            //     commShape = MatrixCoord{actualBlockInComm, 1} * blockShapeMN;
            //     dataLoopsMx = CeilDiv(commShape, commBlockShape);
            //     dLoopsInRank = CeilDiv(dataLoopsMx.row() * dataLoopsMx.column(), params.rankSize);
            //     commScheduler.Update(commShape, commBlockShape, dLoopsInRank);
            //     layoutCommLogicShape = Catlass::MakeCoord<int>(1, dLoopsInRank, commBlockShape.row());
            //     layoutComm = layout::AffineRankN<3>::Packed(layoutCommLogicShape);
            //     actualCommShapeInRank = commShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1);
            // }
            //replace new start
            uint32_t actualBlockInComm = Min(blockPerComm, coreLoops - commIdx * blockPerComm);
            MatrixCoord actualCommShape = MatrixCoord{actualBlockInComm, 1} * blockShapeMN;
            MatrixCoord actualCommShapeInRank = actualCommShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1);
            commScheduler.template SetProblemSize<ReduceScatter::RemoteCopyMode, true>(actualCommShape);
            //replace new end
            auto commAicoreNum = commScheduler.GetRealCore();
            auto commCoreLoops = commScheduler.GetCoreLoop();

            MatrixCoord stageOffset = MatrixCoord{stageId * blockPerComm, 0} * blockShapeMN;
            MatrixCoord commOffsetInRank = MatrixCoord{commIdx * blockPerCommInRank, 0} * blockShapeMN;

            // Wait for AIC to complete computation
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
            // --- Reduce-Scatter Communication Operation ---
            shmemx_barrier_all_vec(); // All Ranks synchronize here
            AscendC::SetAtomicAdd<ElementC>(); // Set the hardware atomic add function
            AscendC::PipeBarrier<PIPE_ALL>();

            // ... It will sum the results in gmC_accum from all Ranks and write the sliced results back to each Rank's own gmC_accum ...
            reduceScatter.AllocEventID();
            if (aivIndex == 0 && aicoreIndex < commAicoreNum) {
                for (uint32_t commLoopIdx = aicoreIndex; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                    MatrixCoord commBlockCoord = commScheduler.GetBlockIdx(commLoopIdx);
                    MatrixCoord blockOffset = commScheduler.template GetBlockOffset<ReduceScatter::RemoteCopyMode, 
                                                ReduceScatter::RemoteCopyDirect>(commBlockCoord);
                    MatrixCoord actualCommBlockShape = commScheduler.template GetActualBlockShape<ReduceScatter::RemoteCopyMode, 
                                                ReduceScatter::RemoteCopyDirect>(commBlockCoord);
                    MatrixCoord blockOffsetInRank = blockOffset % actualCommShapeInRank;

                    uint32_t remoteRankIdx = commBlockCoord.column();
                    if (remoteRankIdx == params.rankIdx) { continue; }

                    auto offsetIn = stageOffset + blockOffset;
                    auto offsetOut = commOffsetInRank + blockOffsetInRank;
                    auto globalLoopIdx = offsetOut.row() / blockShapeMN.row();

                    reduceScatter(blockShapeMN, offsetOut, offsetIn, actualCommBlockShape, 
                                 gmC_accum, params.layoutC_accum, globalLoopIdx, 
                                 remoteRankIdx % params.rankSize, params.teamIdx);
                }       
            }
            reduceScatter.ReleaseEventID();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetAtomicNone();
            AscendC::PipeBarrier<PIPE_ALL>();

            shmemx_barrier_all_vec(); // Synchronize again
            // Notify AIC that AIV has finished processing the current stage's buffer, and AIC can start writing new data
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
        }
        
        // --- Dequantization Operation ---
        // After all computation and communication are complete, dequantize the final int32 result to get a bfloat16 output
        uint32_t M_per_rank = params.problemShape.m() / params.rankSize;
        uint32_t N = params.problemShape.n();
        GemmCoord problemShapeEpilogue{M_per_rank, N, 1};
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

        // Initialize the dequantization Epilogue object
        Dequant dequantEpilogue(resource, params.dequantParams);

        auto cord = Dequant::TileShape::ToCoord();
        typename Dequant::EpilogueTileSwizzle tileScheduler(problemShapeEpilogue.GetCoordMN(), cord);
        uint32_t tileLoops = tileScheduler.GetLoops();

        // AIVs divide the work to process different blocks
        for(uint32_t i = coreIdx; i < tileLoops; i += coreNum) {
            auto tileCoord = tileScheduler.GetTileCoord(i);
            auto actualTileShape = tileScheduler.GetActualTileShape(tileCoord);
            auto acc_offset = tileCoord * cord;
            // Call the dequantization Epilogue
            dequantEpilogue(
                GemmCoord(cord[0], cord[1], 1),
                GemmCoord(tileCoord.row(), tileCoord.column(), 0), 
                GemmCoord(actualTileShape.row(), actualTileShape.column(), 1), 
                gmC_accum[params.layoutC_accum.GetOffset(acc_offset)], 
                params.layoutC_accum.GetTileLayout(actualTileShape)
            );
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    // --- Member Variables ---
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES]; // Flag for AIC->AIV synchronization
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES]; // Flag for AIV->AIC synchronization
    Catlass::Arch::Resource<ArchTag> resource; // Hardware resource object
};

} // namespace Catcoc::DGemm::Kernel

#endif // CATCOC_DGEMM_KERNEL_QUANT_MATMUL_REDUCE_SCATTER_HPP
