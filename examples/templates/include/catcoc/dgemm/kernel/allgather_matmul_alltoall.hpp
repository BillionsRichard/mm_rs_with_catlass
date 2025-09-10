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

void shmem_put_half_mem_nbi(__gm__ half* dest, const __gm__ half* source, size_t nelems, int pe);
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
    

    using BlockSchedulerForMatmul = BlockSchedulerForMatmul_;   // Matmul调度器
    using CommScheduler = CommScheduler_;                       // 通信调度器
    static constexpr uint32_t WORKSPACE_STAGES = 2;             // 定义工作空间（Workspace）使用的流水线级数，通常为2以实现双缓冲

    // 内核启动参数结构体，由Host侧填充并传递给Device侧
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

    // 工作空间结构体，用于在对称内存（SMEM）中规划不同的缓冲区
    struct Workspace {
        GM_ADDR ptr_ag_out;        // AllGather操作的输出缓冲区指针
        GM_ADDR ptr_scatter_out;   // Matmul-Scatter操作的输出缓冲区指针
        GM_ADDR ptr_tmp_buffer;    // 一个临时的中转缓冲区
        
        uint32_t ag_out_size_per_stage;       // AllGather缓冲区每一级流水线的大小（字节）
        // Size of the buffer for one (dest_rank, src_rank) pair
        uint32_t scatter_chunk_size_per_stage; // Scatter缓冲区中每个数据块的大小（字节）

        // 设备端构造函数，根据传入的参数计算和规划工作空间
        CATLASS_DEVICE
        Workspace(Params const& params) {
            uint32_t K = params.problemShape.k();
            uint32_t N = params.problemShape.n();
            uint32_t N_per_rank = N / params.rankSize;
            uint32_t commSizeM = params.commInterval * L1TileShape::M;

            // 计算AllGather输出缓冲区大小：(rank数量 * M方向通信步长 * K维度), TODO: 自己那份没必要写入到共享内存吧？
            ag_out_size_per_stage = params.rankSize * commSizeM * K; // --->（params.rankSize-1） * commSizeM * K ？
            // 计算Scatter数据块大小：(M方向通信步长 * N维度分片大小)
            scatter_chunk_size_per_stage = commSizeM * N_per_rank;

            // AllGather的输出缓冲区从对称内存的起始位置开始
            ptr_ag_out = params.ptrSymmetric;
            
            // Scatter的输出缓冲区紧跟在AllGather缓冲区之后
            uint32_t scatter_out_offset = WORKSPACE_STAGES * ag_out_size_per_stage * sizeof(ElementA);
            ptr_scatter_out = params.ptrSymmetric + scatter_out_offset;

            // 临时缓冲区紧跟在Scatter缓冲区之后
            uint32_t tmp_buffer_offset = scatter_out_offset + (WORKSPACE_STAGES * params.rankSize * params.rankSize * scatter_chunk_size_per_stage * sizeof(ElementC));
            ptr_tmp_buffer = params.ptrSymmetric + tmp_buffer_offset;
        }

        // 获取指定流水线阶段的AllGather输出缓冲区地址
        CATLASS_DEVICE GM_ADDR GetAgOut(uint32_t stageId) {
            return ptr_ag_out + stageId * ag_out_size_per_stage * sizeof(ElementA);
        }

        // 获取Scatter缓冲区中特定数据块的地址
        // 布局为 [stageId][dest_rank][src_rank]
        CATLASS_DEVICE GM_ADDR GetScatterChunk(uint32_t stageId, uint32_t dest_rank, uint32_t src_rank, uint32_t rankSize) {
            uint32_t stage_offset = stageId * (rankSize * rankSize * scatter_chunk_size_per_stage);
            uint32_t dest_rank_offset = dest_rank * (rankSize * scatter_chunk_size_per_stage);
            uint32_t src_rank_offset = src_rank * scatter_chunk_size_per_stage;
            return ptr_scatter_out + (stage_offset + dest_rank_offset + src_rank_offset) * sizeof(ElementC);
        }
    };

    // 内核类构造函数
    CATLASS_DEVICE AllGatherMatmulAlltoall() {
        // 初始化AIC和AIV之间、以及流水线各阶段之间用于同步的标志位
        for (uint32_t i = 0; i < WORKSPACE_STAGES; ++i) {
            flagAivFinishAllGather[i] = Catlass::Arch::CrossCoreFlag(i);
            flagAicFinishMatmulScatter[i] = Catlass::Arch::CrossCoreFlag(i);
        }
        flagAivFinish = Catlass::Arch::CrossCoreFlag(WORKSPACE_STAGES);
    }

    // 内核主执行函数，通过模板特化区分AIC和AIV的逻辑
    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params &params);

    // AIC（AI Core，计算核心）的模板特化实现
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params &params) {
        // 创建工作空间对象
        Workspace workspace(params);
        // 获取问题尺寸
        uint32_t M = params.problemShape.m();
        uint32_t K = params.problemShape.k();
        uint32_t N = params.problemShape.n();
        uint32_t N_per_rank = N / params.rankSize; // 每个rank负责的N维度大小
        uint32_t my_rank_i = params.rankIdx;

        // 计算M维度上的通信步长和循环次数
        uint32_t commSizeM = params.commInterval * L1TileShape::M;
        uint32_t commLoops = CeilDiv(M, commSizeM);

        // 按M维度切分，进行流水线处理
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES; // 计算当前使用的流水线阶段 (0或1)
            uint32_t actualCommSizeM = Min(commSizeM, M - commIdx * commSizeM); // 计算当前步长实际处理的M大小
            
            // 等待AIV核完成当前阶段的AllGather操作
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishAllGather[stageId]);
            
            // 为每一个目标Rank j，计算 A_j @ B_i 的结果
            for (uint32_t dest_rank_j = 0; dest_rank_j < params.rankSize; ++dest_rank_j) {
                // 从SMEM中获取已经AllGather好的矩阵A的第j片 (A_j)
                GM_ADDR ptr_A_j_base = workspace.GetAgOut(stageId);
                GM_ADDR ptr_A_j = ptr_A_j_base + dest_rank_j * actualCommSizeM * K * sizeof(ElementA);
                auto layout_A_j = Catlass::layout::RowMajor(actualCommSizeM, K);
                AscendC::GlobalTensor<ElementA> smem_a_j;
                smem_a_j.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(ptr_A_j));

                // 获取当前Rank i本地的矩阵B (B_i)
                AscendC::GlobalTensor<ElementB> gmB;
                gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));

                // 计算结果要写入的目标地址，位于SMEM的Scatter缓冲区
                GM_ADDR ptr_scatter_dest = workspace.GetScatterChunk(stageId, my_rank_i, dest_rank_j, params.rankSize);
                if (my_rank_i == 1 && dest_rank_j == 0 && commIdx == 0 && AscendC::GetBlockIdx() == 0) {
                    cce::printf("AIC[rank=%u, block=%u] writing for dest_rank=%u. commIdx=%u, stageId=%u\n", 
                                my_rank_i, AscendC::GetBlockIdx(), dest_rank_j, commIdx, stageId);
                }
                auto layout_scatter_dest = Catlass::layout::RowMajor(actualCommSizeM, N_per_rank);
                AscendC::GlobalTensor<ElementC> smem_scatter_dest;
                smem_scatter_dest.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(ptr_scatter_dest));

                // 实例化块级MMAD计算模块
                BlockMmad blockMmad(resource);
                // 定义当前要计算的子问题 A_j @ B_i 的形状
                GemmCoord problem_shape_ji = {actualCommSizeM, N_per_rank, K};
                // 创建任务调度器，将子问题划分到不同的计算核上
                Catlass::Gemm::Block::GemmIdentityBlockSwizzle<> scheduler(problem_shape_ji, L1TileShape::ToCoordMN());
                
                uint32_t aicoreIdx = AscendC::GetBlockIdx(); // 获取当前AICore的ID
                uint32_t aicoreNum = AscendC::GetBlockNum(); // 获取总AICore数量
                uint32_t matmul_loops = scheduler.GetCoreLoops(); // 获取需要循环的次数

                // 每个AICore处理一部分计算任务
                for (uint32_t i = aicoreIdx; i < matmul_loops; i += aicoreNum) {
                    GemmCoord block_coord = scheduler.GetBlockCoord(i); // 获取当前处理块的坐标
                    GemmCoord actual_block_shape = scheduler.GetActualBlockShape(block_coord); // 获取实际块形状（处理边界情况）
                    GemmCoord offset_coord = block_coord * L1TileShape::ToCoord(); // 计算偏移坐标

                    // 根据偏移坐标计算A, B, C各自的偏移量
                    int64_t offsetA = layout_A_j.GetOffset(offset_coord.GetCoordMK());
                    int64_t offsetB = params.layoutB.GetOffset(offset_coord.GetCoordKN());
                    int64_t offsetC = layout_scatter_dest.GetOffset(offset_coord.GetCoordMN());

                    // 调用MMAD模块执行计算，并将结果直接写入smem_scatter_dest指向的SMEM地址
                    blockMmad(
                        smem_a_j[offsetA], layout_A_j,
                        gmB[offsetB], params.layoutB,
                        smem_scatter_dest[offsetC], layout_scatter_dest,
                        actual_block_shape
                    );
                    cce::printf("after mm, AIC[rank=%u, block=%u, aicoreIdx=%u] a[0]=%f, b[0]=%f, C[0]=%f, \n", 
                                                my_rank_i, AscendC::GetBlockIdx(), aicoreIdx,
                                                smem_a_j.GetValue(offsetA),
                                                gmB.GetValue(offsetB),
                                                smem_scatter_dest.GetValue(offsetC));
                    
                }
            }
            
            // 设置标志位，通知AIV核当前阶段的Matmul-Scatter已完成
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishMatmulScatter[stageId]);
        }
        
        // 等待所有AICore都完成工作
        Catlass::Arch::CrossCoreBarrier<0, PIPE_FIX>();
        // 设置最终完成标志
        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAivFinish);
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params &params) {
        // 创建工作空间对象
        Workspace workspace(params);
        // 获取问题尺寸
        uint32_t M = params.problemShape.m();
        uint32_t K = params.problemShape.k();
        uint32_t N = params.problemShape.n();
        uint32_t N_per_rank = N / params.rankSize;

        // 计算M维度上的通信步长和循环次数
        uint32_t commSizeM = params.commInterval * L1TileShape::M;
        uint32_t commLoops = CeilDiv(M, commSizeM);

        // 按M维度切分，进行流水线处理
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES; // 计算当前流水线阶段

            // 如果是第二轮或之后的流水线，需要等待上一轮的Scatter操作完成才能复用缓冲区
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAicFinishMatmulScatter[stageId]);
            }
            
            // 向量核间的同步栅栏
            shmemx_barrier_all_vec();

            uint32_t actualCommSizeM = Min(commSizeM, M - commIdx * commSizeM); // 计算当前步长实际处理的M大小
            auto actualCommShape = DistMatrixCoord(actualCommSizeM, K, params.rankSize);
            // --- 阶段一: AllGather ---
            // AIV核负责将各自本地的矩阵A分片，通过通信汇聚到SMEM中
            {
                // 实例化AllGather模块
                AllGather allGather(resource, params.allGatherParams);
                // 获取通信块形状和核划分信息
                MatrixCoord commBlockShape = params.allGatherParams.BlockShape();
                MatrixCoord commCoreSplit = params.allGatherParams.CoreSplit();
                // 创建通信调度器
                CommScheduler commScheduler(commBlockShape, commCoreSplit);
                MatrixCoord loopsInRank = CeilDiv(MatrixCoord(actualCommShape.GetCoordInRank()), commBlockShape);
                commScheduler.UpdateProblem(actualCommShape, loopsInRank);
                auto commAicoreNum = commScheduler.GetRealCore();
                auto commCoreLoops = commScheduler.GetCoreLoop();
                MatrixCoord commSrcOffset{commIdx * commSizeM, 0}; // 计算源数据在M维度上的偏移
                
                // 获取SMEM中用于AllGather输出的Tensor
                AscendC::GlobalTensor<ElementA> gmSymmetric;
                gmSymmetric.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(workspace.GetAgOut(stageId)));
                auto layoutSymmetric = Catlass::layout::RowMajor(params.rankSize * actualCommSizeM, K);

                allGather.InitBlockLoop(); // 初始化块循环
                uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(); // 获取AIV核ID
                uint32_t subcoreIdx = AscendC::GetSubBlockIdx();
                if (subcoreIdx == 0 && aicoreIdx < commAicoreNum) {
                    // 每个AIV核处理一部分AllGather任务
                    for (uint32_t loopIdx = aicoreIdx; loopIdx < commCoreLoops; loopIdx += commAicoreNum) {
                        DistMatrixCoord commBlockCoord = commScheduler.GetBlockCoord(loopIdx);
                        MatrixCoord blockOffsetInRank = commScheduler.GetBlockOffsetInRank(commBlockCoord.GetCoordInRank());
                        MatrixCoord actualCommBlockShape = commScheduler.GetActualBlockShapeByOffset(blockOffsetInRank);
                        uint32_t remoteRankIdx = commBlockCoord.rank();
                        auto offsetSrc = commSrcOffset + blockOffsetInRank; // 源偏移
                        MatrixCoord commDstOffset{remoteRankIdx * actualCommSizeM, 0}; // 目标偏移
                        auto offsetDst = commDstOffset + blockOffsetInRank;
                        AscendC::GlobalTensor<ElementA> gmA;
                        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
                        // 获取源数据块和目标数据块
                        auto gmBlockSrc = gmA[params.layoutA.GetOffset(offsetSrc)];
                        auto layoutBlockSrc = params.layoutA.GetTileLayout(actualCommBlockShape);
                        auto gmBlockDst = gmSymmetric[layoutSymmetric.GetOffset(offsetDst)];
                        auto layoutBlockDst = layoutSymmetric.GetTileLayout(actualCommBlockShape);
                        // 调用AllGather模块执行数据拷贝/通信
                        allGather(gmBlockSrc, layoutBlockSrc, gmBlockDst, layoutBlockDst, actualCommBlockShape, remoteRankIdx);
                    }
                }
                allGather.FinalizeBlockLoop(); // 结束块循环
            }

            // 向量核间同步，确保AllGather完成
            shmemx_barrier_all_vec();
            // 设置标志位，通知AIC核当前阶段的AllGather已完成
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishAllGather[stageId]);

            // --- 等待AIC完成Matmul-Scatter ---
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishMatmulScatter[stageId]);
            shmemx_barrier_all_vec();

            // --- Final Assembly ---
			// AIV核从SMEM中读取计算结果，并写入最终的GMEM输出地址
            {
                uint32_t my_rank_i = params.rankIdx; // This AIV is on the source rank
                uint32_t aiv_core_idx_in_rank = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
                uint32_t aiv_core_num_in_rank = AscendC::GetBlockNum();
                uint32_t data_len_per_chunk = actualCommSizeM * N_per_rank;
                for (uint32_t dest_rank_j = aiv_core_idx_in_rank; dest_rank_j < params.rankSize; dest_rank_j += aiv_core_num_in_rank) {
                    if (dest_rank_j == my_rank_i) {
                        continue; // No need to send data to myself
                    }
                    GM_ADDR local_src_ptr = workspace.GetScatterChunk(stageId, my_rank_i, dest_rank_j, params.rankSize);
                    GM_ADDR remote_dest_ptr = workspace.GetScatterChunk(stageId, dest_rank_j, my_rank_i, params.rankSize);
                    if constexpr (std::is_same_v<ElementC, half>) {
                        shmem_put_half_mem_nbi(
                            reinterpret_cast<__gm__ half*>(remote_dest_ptr), // remote destination
                            reinterpret_cast<__gm__ half*>(local_src_ptr),      // local source
                            data_len_per_chunk,
                            dest_rank_j                                         // destination rank
                        );
                    }
                }
            }
            shmemx_barrier_all_vec();
            {
                uint32_t my_rank_j = params.rankIdx;
                AscendC::GlobalTensor<ElementC> gmC;
                gmC.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC));
                uint32_t aicoreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
                uint32_t aicoreNum = AscendC::GetBlockNum();

                // 遍历所有源Rank i，收集它们为我（Rank j）计算的数据
                for (uint32_t src_rank_i = 0; src_rank_i < params.rankSize; ++src_rank_i) {
                    // 计算数据源在SMEM中的地址
                    GM_ADDR src_ptr_base = workspace.GetScatterChunk(stageId, my_rank_j, src_rank_i, params.rankSize);
                    auto layout_src = Catlass::layout::RowMajor(actualCommSizeM, N_per_rank);
                    // 计算数据在最终输出矩阵C中的偏移
                    MatrixCoord dst_chunk_offset = {commIdx * commSizeM, src_rank_i * N_per_rank};

                        // Local data, copy directly from shmem to gmem.
                        __gm__ ElementC* src_ptr = reinterpret_cast<__gm__ ElementC*>(src_ptr_base);
                        for(uint32_t m = aicoreIdx; m < actualCommSizeM; m += aicoreNum) {
                            for(uint32_t n = 0; n < N_per_rank; ++n) {
                                MatrixCoord src_coord = {m, n};
                                MatrixCoord dst_coord = dst_chunk_offset + src_coord;
                                gmC.SetValue(params.layoutC.GetOffset(dst_coord), src_ptr[layout_src.GetOffset(src_coord)]);
                        // Remote data, pull from remote rank's shmem into a temporary local shmem buffer, then copy to gmem.
                        
                        

                            // Cast to half for printing
                            // half* tmp_half_ptr = reinterpret_cast<half*>(tmp_buffer_ptr);

                        }
                    }
                }
            }
        }

        // 等待所有流水线步骤完成
        Catlass::Arch::CrossCoreWaitFlag(flagAivFinish);
        Catlass::Arch::CrossCoreBarrier<0, PIPE_MTE3>();

        // 最终的流水线屏障
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
