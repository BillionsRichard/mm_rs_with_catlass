#ifndef CATCOC_DGEMM_KERNEL_QUANT_MATMUL_REDUCE_SCATTER_HPP
#define CATCOC_DGEMM_KERNEL_QUANT_MATMUL_REDUCE_SCATTER_HPP

// 该文件定义了量化矩阵乘+ReduceScatter融合算子的核心Kernel。
// Kernel融合了四个主要阶段：
// 1. Quantized Matrix Multiplication (在AIC上执行)
// 2. Reduce-Scatter Communication (在AIV上执行)
// 3. Bias Addition Epilogue (在AIV上执行)
// 4. Dequantization Epilogue (在AIV上执行)

#include "catcoc/catcoc.hpp"

// from catlass
#include "catlass/arch/resource.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"

namespace Catcoc::DGemm::Kernel {

using Catlass::MatrixCoord;
using Catlass::GemmCoord;

// 量化矩阵乘法 + Reduce-Scatter 的核心 Kernel 模板类
template <
    class BlockMmad_,                 // GEMM 计算的基本单元 (e.g., BlockMmad)
    class BlockEpilogueReduceScatter_,// Reduce-Scatter Epilogue
    class BlockEpilogueBias_,         // Bias Addition Epilogue
    class BlockEpilogueDequant_,      // Dequantization Epilogue
    class BlockScheduler_,            // GEMM 计算块的调度器
    class BlockEpilogueScheduler_,    // 通信Epilogue块的调度器
    uint32_t WORKSPACE_STAGES_        // 流水线 stage 数量
>
class QuantMatmulReduceScatter {
public:
    //
    // ------------------- 类型别名和模板参数定义 -------------------
    //
    // 从 GEMM 计算单元中提取类型
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC; // 累加器类型，通常是 int32
    using LayoutC = typename BlockMmad::LayoutC;

    // Epilogue 相关的类型定义
    using ReduceScatter = BlockEpilogueReduceScatter_;
    using ReduceScatterParams = typename ReduceScatter::Params;
    using BiasAdd = BlockEpilogueBias_;
    using BiasParams = typename BiasAdd::Params;
    using Dequant = BlockEpilogueDequant_;
    using DequantParams = typename Dequant::Params;

    using ElementD = bfloat16_t; // 最终输出类型
    using LayoutD = Catlass::layout::RowMajor;

    // 调度器类型
    using BlockScheduler = BlockScheduler_;
    using CommScheduler = BlockEpilogueScheduler_;

    // 流水线 stage 数量
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    /// @brief Kernel 的参数结构体，用于从 Host 传递参数到 Device
    struct Params {
        //
        // --- 数据成员 ---
        //
        GemmCoord problemShape; // 整个问题的尺寸 (m, n, k)

        uint32_t rankIdx;       // 当前 Rank 的 ID
        uint32_t rankSize;      // 总 Rank 数量

        GM_ADDR ptrA;           // 输入矩阵 A 的 GMEM 地址 (int8)
        LayoutA layoutA;        // 矩阵 A 的布局
        GM_ADDR ptrB;           // 输入矩阵 B 的 GMEM 地址 (int8)
        LayoutB layoutB;        // 矩阵 B 的布局
        GM_ADDR ptrSymmetric;   // 用于Rank间通信的对称内存工作空间指针

        // 各个 Epilogue 的参数
        ReduceScatterParams reduceScatterParams;
        BiasParams biasParams;
        DequantParams dequantParams;

        GM_ADDR ptrC_accum;     // 累加结果 C 的 GMEM 地址 (int32)
        LayoutC layoutC_accum;  // 累加结果 C 的布局
        
        GM_ADDR ptrD_out;       // 最终输出 D 的 GMEM 地址 (bfloat16)
        LayoutD layoutD_out;    // 输出 D 的布局

        uint32_t commInterval;  // 通信间隔

        //
        // --- 方法 ---
        //
        CATLASS_DEVICE
        Params() {}

        CATLASS_DEVICE
        Params(
            GemmCoord const &problemShape_,
            uint32_t rank_, uint32_t rankSize_,
            GM_ADDR ptrA_, LayoutA const &layoutA_,
            GM_ADDR ptrB_, LayoutB const &layoutB_,
            GM_ADDR ptrSymmetric_,
            ReduceScatterParams const &reduceScatterParams_,
            BiasParams const &biasParams_,
            DequantParams const &dequantParams_,
            GM_ADDR ptrC_accum_, LayoutC const &layoutC_accum_,
            GM_ADDR ptrD_out_, LayoutD const &layoutD_out_,
            uint32_t commInterval_
        ) : problemShape(problemShape_),
            rankIdx(rank_), rankSize(rankSize_),
            ptrA(ptrA_), layoutA(layoutA_),
            ptrB(ptrB_), layoutB(layoutB_),
            ptrSymmetric(ptrSymmetric_),
            reduceScatterParams(reduceScatterParams_),
            biasParams(biasParams_),
            dequantParams(dequantParams_),
            ptrC_accum(ptrC_accum_), layoutC_accum(layoutC_accum_),
            ptrD_out(ptrD_out_), layoutD_out(layoutD_out_),
            commInterval(commInterval_) {}
    };

    //
    // ------------------- Kernel 方法 -------------------
    //
    // 构造函数
    CATLASS_DEVICE
    QuantMatmulReduceScatter()
    {
        // 初始化用于 AIC 和 AIV 之间进行乒乓操作同步的 Flag
        for (uint32_t i = 0; i < WORKSPACE_STAGES; ++i) {
            flagAicFinishStore[i] = Catlass::Arch::CrossCoreFlag(i);
            flagAivFinishCompute[i] = Catlass::Arch::CrossCoreFlag(i);
        }
    }

    // Kernel 主函数，通过模板特化区分 AIC 和 AIV 的不同逻辑
    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE
    void operator()(Params &params);

    // AIC (AI Core) 上的 Kernel 实现，主要负责 GEMM 计算
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIC>(Params &params)
    {
        // 获取 AI Core 的索引和总数
        uint32_t aicoreIndex = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        uint32_t blockPerComm = aicoreNum * params.commInterval; // 每次通信处理的 block 总数
        uint32_t blockPerCommInRank = blockPerComm / params.rankSize; // 每个 Rank 每次通信处理的 block 数

        GemmCoord blockShape = L1TileShape::ToCoord(); // L1 Tile 的形状
        GemmCoord problemShapeInRank = params.problemShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1, 1); // 每个 Rank 处理的问题尺寸
        BlockScheduler matmulBlockScheduler(problemShapeInRank, blockShape.GetCoordMN()); // 初始化 GEMM 计算块调度器
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops() * params.rankSize; // 总的循环次数
        uint32_t commLoops = CeilDiv(coreLoops, blockPerComm); // 通信循环的次数

        BlockMmad blockMmad(resource); // 实例化 GEMM 计算单元

        // 创建指向 GMEM 的 Tensor 对象
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.ptrA));
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.ptrB));
        AscendC::GlobalTensor<ElementC> gmC_workspace;
        gmC_workspace.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementC> gmC_accum;
        gmC_accum.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC_accum));

        // 定义用于通信的 Workspace 的布局
        auto layoutC = Catlass::layout::RowMajor{
            WORKSPACE_STAGES * blockPerComm * L1TileShape::M, L1TileShape::N,
            L1TileShape::N
        };

        auto layoutCRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, blockPerComm, L1TileShape::M);
        auto layoutCRow = layout::AffineRankN<3>::Packed(layoutCRowLogicShape);

        // 主循环，按通信粒度进行
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES; // 当前使用的 pipeline stage

            // 如果 pipeline 已满, 等待 AIV 完成上一轮的计算
            if (commIdx >= WORKSPACE_STAGES) {
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]);
            }

            // 计算当前通信轮次实际需要处理的 block 数量 (处理末尾剩余部分)
            uint32_t actualBlockPerComm = (commIdx == commLoops - 1) ?
                (coreLoops - blockPerComm * commIdx) : blockPerComm;
            uint32_t actualBlockPerCommInRank = actualBlockPerComm / params.rankSize;

            uint32_t commBlockOffsetInRank = commIdx * blockPerCommInRank;
            // 在所有 AICore 间分发计算任务
            for (
                uint32_t blockIdxInComm = aicoreIndex;
                blockIdxInComm < actualBlockPerComm;
                blockIdxInComm += aicoreNum
            ) {
                uint32_t loopIdxInRank = commBlockOffsetInRank + blockIdxInComm % actualBlockPerCommInRank;
                uint32_t targetRankIdx = blockIdxInComm / actualBlockPerCommInRank;
                
                // 计算当前 block 的坐标和实际形状
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdxInRank);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);

                GemmCoord offsetCoord = blockCoord * blockShape;
                
                // 计算输入矩阵 A 和 B 的偏移
                auto rankOffsetA = problemShapeInRank.GetCoordMK() * Catlass::MakeCoord<uint32_t>(targetRankIdx, 0);
                auto blockOffsetA = offsetCoord.GetCoordMK() + rankOffsetA;
                auto blockOffsetB = offsetCoord.GetCoordKN();
                
                MatrixCoord blockOffsetStore;
                AscendC::GlobalTensor<ElementC> gmStore;
                Catlass::layout::RowMajor layoutStore;
                // 判断结果是写回本地累加区还是跨 Rank 的工作区
                if (targetRankIdx == params.rankIdx) {
                    blockOffsetStore = offsetCoord.GetCoordMN();
                    gmStore = gmC_accum;
                    layoutStore = params.layoutC_accum;
                }
                else {
                    blockOffsetStore = MatrixCoord{layoutCRow(Catlass::MakeCoord<int>(stageId, blockIdxInComm, 0)), 0};
                    gmStore = gmC_workspace;
                    layoutStore = layoutC;
                }
                
                int64_t offsetA = params.layoutA.GetOffset(blockOffsetA);
                int64_t offsetB = params.layoutB.GetOffset(blockOffsetB);
                int64_t offsetStore = layoutStore.GetOffset(blockOffsetStore);
                
                // 执行块内矩阵乘加
                blockMmad(
                    gmA[offsetA], params.layoutA,
                    gmB[offsetB], params.layoutB,
                    gmStore[offsetStore], layoutStore,
                    actualBlockShape
                );
            }

            // 通知 AIV 当前 stage 的计算已完成
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
        }
        AscendC::PipeBarrier<PIPE_ALL>(); // 确保所有 AICore 任务完成
    }


    // AIV (AI Vector Core) 上的 Kernel 实现，主要负责通信和后处理
    template <>
    CATLASS_DEVICE
    void operator()<AscendC::AIV>(Params &params)
    {
        // 获取 AIV Core 的索引和总数
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

        ReduceScatter reduceScatter(resource, params.reduceScatterParams); // 实例化 Reduce-Scatter Epilogue

        // 创建指向 GMEM 的 Tensor 对象
        AscendC::GlobalTensor<ElementC> gmC_workspace;
        gmC_workspace.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrSymmetric));
        AscendC::GlobalTensor<ElementC> gmC_accum;
        gmC_accum.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.ptrC_accum));

        // 初始化通信调度器
        MatrixCoord commBlockShape = params.reduceScatterParams.BlockShape();
        MatrixCoord commCoreSplit = params.reduceScatterParams.CoreSplit();
        MatrixCoord commShape = MatrixCoord{blockPerComm, 1} * blockShapeMN;
        MatrixCoord dataLoopsMx = CeilDiv(commShape, commBlockShape);
        uint32_t dLoopsInRank = CeilDiv(dataLoopsMx.row() * dataLoopsMx.column(), params.rankSize);
        CommScheduler commScheduler(params.rankIdx, params.rankSize, commCoreSplit, 
                                    commShape, commBlockShape, dLoopsInRank);
        MatrixCoord actualCommShapeInRank = commShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1);

        auto layoutCommLogicShape = Catlass::MakeCoord<int>(1, dLoopsInRank, commBlockShape.row());
        auto layoutComm = layout::AffineRankN<3>::Packed(layoutCommLogicShape);

        // 主循环，按通信粒度进行
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;
            // 处理最后一个可能不足一个 full block 的循环
            if (commIdx == commLoops - 1) {
                uint32_t actualBlockInComm = coreLoops - commIdx * blockPerComm;
                commShape = MatrixCoord{actualBlockInComm, 1} * blockShapeMN;
                dataLoopsMx = CeilDiv(commShape, commBlockShape);
                dLoopsInRank = CeilDiv(dataLoopsMx.row() * dataLoopsMx.column(), params.rankSize);
                commScheduler.Update(commShape, commBlockShape, dLoopsInRank);
                layoutCommLogicShape = Catlass::MakeCoord<int>(1, dLoopsInRank, commBlockShape.row());
                layoutComm = layout::AffineRankN<3>::Packed(layoutCommLogicShape);
                actualCommShapeInRank = commShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1);
            }
            auto commAicoreNum = commScheduler.GetRealCore();
            auto commCoreLoops = commScheduler.GetCoreLoop();

            MatrixCoord stageOffset = MatrixCoord{stageId * blockPerComm, 0} * blockShapeMN;
            MatrixCoord commOffsetInRank = MatrixCoord{commIdx * blockPerCommInRank, 0} * blockShapeMN;

            // 等待 AIC 完成计算
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
            shmemx_barrier_all_vec(); // 等待所有 Rank 的 AIC 计算完成

            // 执行 Reduce-Scatter
            AscendC::SetAtomicAdd<ElementC>(); // 设置硬件原子加
            AscendC::PipeBarrier<PIPE_ALL>();
            reduceScatter.AllocEventID();
            if (aivIndex == 0 && aicoreIndex < commAicoreNum) {
                for (uint32_t commLoopIdx = aicoreIndex; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                    MatrixCoord commBlockCoord = commScheduler.GetBlockIdx(commLoopIdx);
                    MatrixCoord blockOffset = commScheduler.template GetBlockOffset<ReduceScatter::RemoteCopyMode,
                        ReduceScatter::RemoteCopyDirect>(commBlockCoord, layoutComm);
                    MatrixCoord actualCommBlockShape = commScheduler.template GetActualBlockShape<
                        ReduceScatter::RemoteCopyMode, ReduceScatter::RemoteCopyDirect>(commBlockCoord, layoutComm);
                    MatrixCoord blockOffsetInRank = blockOffset % actualCommShapeInRank;

                    uint32_t remoteRankIdx = commBlockCoord.column();
                    if (remoteRankIdx == params.rankIdx) {
                        continue;
                    }

                    auto offsetIn = stageOffset + blockOffset;
                    auto offsetOut = commOffsetInRank + blockOffsetInRank;

                    auto globalLoopIdx = offsetOut.row() / blockShapeMN.row();

                    // 调用 Reduce-Scatter epilogue
                    reduceScatter(blockShapeMN, offsetOut, offsetIn, actualCommBlockShape,
                        gmC_accum, params.layoutC_accum, globalLoopIdx, remoteRankIdx % params.rankSize);
                }
            }
            reduceScatter.ReleaseEventID();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetAtomicNone(); // 关闭原子加
            AscendC::PipeBarrier<PIPE_ALL>();

            // 等待所有 Rank 的 Reduce-Scatter 完成
            shmemx_barrier_all_vec();

            // 通知 AIC 当前 stage 的 AIV 任务已完成
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
        }//通信for循环结束

        // --- 后处理阶段 ---
        uint32_t M_per_rank = params.problemShape.m() / params.rankSize; // 每个Rank处理的M维度大小
        uint32_t N = params.problemShape.n();                           // N维度大小
        GemmCoord problemShapeEpilogue{M_per_rank, N, 1};               // Epilogue 阶段的问题尺寸
        uint32_t coreNum = AscendC::GetBlockNum();                      // 核总数
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum(); // 当前核ID

        // /*
        // --- 偏置加法步骤 ---
        if (params.biasParams.ptr_bias != 0) {
            {
                // 实例化 Bias Epilogue
                BiasAdd biasAddEpilogue(resource, params.biasParams);
                // 调用 Bias Epilogue，处理整个 epilogue 问题
                biasAddEpilogue(
                    problemShapeEpilogue,
                    GemmCoord{0, 0, 0},
                    problemShapeEpilogue,
                    gmC_accum,
                    params.layoutC_accum
                );
                AscendC::PipeBarrier<PIPE_ALL>(); // 确保所有 core 的 bias add 完成
            }
        }
        // */
        // --- 最终的反量化步骤 ---
        Dequant dequantEpilogue(resource, params.dequantParams); // 实例化 Dequant Epilogue

        // 使用 Dequant Epilogue 自己的 tile 调度器来遍历输出矩阵
        auto cord = Dequant::TileShape::ToCoord();
        typename Dequant::EpilogueTileSwizzle tileScheduler(problemShapeEpilogue.GetCoordMN(), cord);
        uint32_t tileLoops = tileScheduler.GetLoops();

        // 在所有 AIV core 间分发 tile 任务
        for(uint32_t i = coreIdx; i < tileLoops; i += coreNum) {
            auto tileCoord = tileScheduler.GetTileCoord(i);
            auto actualTileShape = tileScheduler.GetActualTileShape(tileCoord);
            auto acc_offset = tileCoord * cord;
            // 对每个 tile 调用 dequant epilogue
            dequantEpilogue(
                GemmCoord(cord[0], cord[1], 1),
                GemmCoord(tileCoord.row(), tileCoord.column(), 0), 
                GemmCoord(actualTileShape.row(), actualTileShape.column(), 1), 
                gmC_accum[params.layoutC_accum.GetOffset(acc_offset)], 
                params.layoutC_accum.GetTileLayout(actualTileShape)
            );
        }
        AscendC::PipeBarrier<PIPE_ALL>(); // 确保所有 dequant 任务完成
    }

private:
    // 用于 AIC 和 AIV 之间进行乒乓操作同步的 Flag
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES];
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES];
    // 每个核上的硬件资源 (例如 UBUF)
    Catlass::Arch::Resource<ArchTag> resource;
};

} // namespace Catcoc::DGemm::Kernel

#endif // CATCOC_DGEMM_KERNEL_QUANT_MATMUL_REDUCE_SCATTER_HPP
