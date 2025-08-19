// 防止头文件被重复包含
#ifndef CATCOC_DGEMM_KERNEL_QUANT_MATMUL_REDUCE_SCATTER_HPP
#define CATCOC_DGEMM_KERNEL_QUANT_MATMUL_REDUCE_SCATTER_HPP

// 包含依赖的头文件
#include "catcoc/catcoc.hpp"                     // catcoc库的核心头文件，可能包含通信原语等
#include "catlass/arch/resource.hpp"             // catlass库中关于硬件资源管理的定义
#include "catlass/arch/cross_core_sync.hpp"      // catlass库中用于核间同步的工具，如Flag
#include "catlass/gemm_coord.hpp"                // catlass库中用于表示GEMM（通用矩阵乘法）相关坐标的结构
#include "catlass/matrix_coord.hpp"              // catlass库中用于表示矩阵坐标的结构

namespace Catcoc::DGemm::Kernel {

// 使用类型别名简化代码
using Catlass::MatrixCoord;
using Catlass::GemmCoord;

//
// QuantMatmulReduceScatter 是一个融合算子的内核实现。它将量化矩阵乘法（QuantMatmul）与一个通信操作（Reduce-Scatter）融合在一起。
template <
    class BlockMmad_,                // 矩阵乘法（MMAD）的基本计算单元
    class BlockEpilogueReduceScatter_, // Reduce-Scatter操作的Epilogue实现
    class BlockEpilogueDequant_,       // 反量化的Epilogue实现
    class BlockScheduler_,           // 计算任务（Block）的调度器
    class BlockEpilogueScheduler_,     // 通信任务的调度器
    uint32_t WORKSPACE_STAGES_      // 流水线（Pipeline）的级数，用于隐藏数据传输延迟
>
class QuantMatmulReduceScatter {
public:
    // --- 类型别名定义 ---
    // 将模板参数和其内部类型定义为更易读的别名
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape; // L1 Tile的形状，这是硬件执行计算的基本单元大小
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;       // 累加器C的元素类型，通常是int32
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementBias = typename BlockMmad::ElementBias; // 从BlockMmad获取Bias类型
    using LayoutBias = typename BlockMmad::LayoutBias;
    using ReduceScatter = BlockEpilogueReduceScatter_;
    using ReduceScatterParams = typename ReduceScatter::Params;
    using Dequant = BlockEpilogueDequant_;
    using DequantParams = typename Dequant::Params;
    using ElementD = bfloat16_t;                          // 最终输出D的元素类型
    using LayoutD = Catlass::layout::RowMajor;
    using BlockScheduler = BlockScheduler_;
    using CommScheduler = BlockEpilogueScheduler_;
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_; // 流水线级数

    //
    // Params 结构体：用于从主机侧传递算子所需的所有参数
    //
    struct Params {
        GemmCoord problemShape; // 整个问题的形状 (M, N, K)
        uint32_t rankIdx;       // 当前计算卡（Rank）的ID
        uint32_t rankSize;      // 总共的计算卡数量
        GM_ADDR ptrA; LayoutA layoutA; // A矩阵的全局内存地址和布局
        GM_ADDR ptrB; LayoutB layoutB; // B矩阵的全局内存地址和布局
        GM_ADDR ptrBias; LayoutBias layoutBias; // Bias的全局内存地址和布局
        GM_ADDR ptrSymmetric;          // 用于卡间通信的对称内存（Symmetric Memory）工作空间地址
        ReduceScatterParams reduceScatterParams; // Reduce-Scatter Epilogue的参数
        DequantParams dequantParams;             // 反量化 Epilogue的参数
        GM_ADDR ptrC_accum; LayoutC layoutC_accum; // C累加器结果的地址和布局
        GM_ADDR ptrD_out; LayoutD layoutD_out;     // 最终输出D的地址和布局
        uint32_t commInterval;                     // 通信间隔，控制计算和通信的比例

        CATLASS_DEVICE Params() {} // 默认构造
        CATLASS_DEVICE Params(     // 带参构造，用于主机侧初始化
            GemmCoord const &problemShape_, 
            uint32_t rank_, uint32_t rankSize_,
            GM_ADDR ptrA_, LayoutA const &layoutA_, 
            GM_ADDR ptrB_, LayoutB const &layoutB_, 
            GM_ADDR ptrBias_, LayoutBias const &layoutBias_, 
            GM_ADDR ptrSymmetric_,
            ReduceScatterParams const &reduceScatterParams_, DequantParams const &dequantParams_,
            GM_ADDR ptrC_accum_, LayoutC const &layoutC_accum_, 
            GM_ADDR ptrD_out_, LayoutD const &layoutD_out_, 
            uint32_t commInterval_
        ) : problemShape(problemShape_), 
            rankIdx(rank_), rankSize(rankSize_),
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
    // 内核构造函数：初始化核间同步用的Flag
    //
    CATLASS_DEVICE QuantMatmulReduceScatter() {
        // 通过流水线级数，创建多组Flag，用于AIC（AI Core）和AIV（AI Vector）之间的同步
        // AIC负责计算，AIV负责通信和后处理，它们之间通过乒乓操作（Pipelining）来隐藏延迟
        for (uint32_t i = 0; i < WORKSPACE_STAGES; ++i) {
            flagAicFinishStore[i] = Catlass::Arch::CrossCoreFlag(i);   // AIC完成计算并存储结果的标志
            flagAivFinishCompute[i] = Catlass::Arch::CrossCoreFlag(i); // AIV完成通信和后处理的标志
        }
    }

    //
    // 内核执行入口：通过模板特化，为不同类型的硬件核心（AIC/AIV）提供不同的实现
    //
    template <int32_t CORE_TYPE = g_coreType> CATLASS_DEVICE void operator()(Params &params);

    //
    // AIC (AI Core) 的内核实现：主要负责高密度的矩阵乘法计算
    //
    template <> CATLASS_DEVICE void operator()<AscendC::AIC>(Params &params) {
        // 获取当前AICore的ID和总AICore数量
        uint32_t aicoreIndex = AscendC::GetBlockIdx();
        uint32_t aicoreNum = AscendC::GetBlockNum();
        // 计算和通信相关的尺寸参数
        uint32_t blockPerComm = aicoreNum * params.commInterval;
        uint32_t blockPerCommInRank = blockPerComm / params.rankSize;

        // 获取计算的基本单元（Block）的形状
        GemmCoord blockShape = L1TileShape::ToCoord();
        // 计算当前Rank负责处理的问题形状（M维度被切分）
        GemmCoord problemShapeInRank = params.problemShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1, 1);
        // 初始化计算任务调度器
        BlockScheduler matmulBlockScheduler(problemShapeInRank, blockShape.GetCoordMN());
        // 计算总的循环次数
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops() * params.rankSize;
        uint32_t commLoops = CeilDiv(coreLoops, blockPerComm);

        // 创建矩阵乘法计算对象
        BlockMmad blockMmad(resource);

        // 创建指向全局内存的张量对象
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

        // 定义用于存储跨卡计算结果的工作空间（Workspace）的布局
        auto layoutC = Catlass::layout::RowMajor{ WORKSPACE_STAGES * blockPerComm * L1TileShape::M, L1TileShape::N, L1TileShape::N };
        auto layoutCRowLogicShape = Catlass::MakeCoord<int>(WORKSPACE_STAGES, blockPerComm, L1TileShape::M);
        auto layoutCRow = layout::AffineRankN<3>::Packed(layoutCRowLogicShape);

        // --- 主循环：流水线式地执行计算 ---
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES; // 计算当前使用的流水线阶段
            // 如果不是初始阶段，需要等待AIV完成对上一个同阶段缓冲区数据的处理
            if (commIdx >= WORKSPACE_STAGES) { 
                Catlass::Arch::CrossCoreWaitFlag(flagAivFinishCompute[stageId]); 
            }

            // 计算当前通信周期内实际要处理的块数
            uint32_t actualBlockPerComm = (commIdx == commLoops - 1) ? (coreLoops - blockPerComm * commIdx) : blockPerComm;
            uint32_t actualBlockPerCommInRank = actualBlockPerComm / params.rankSize;
            uint32_t commBlockOffsetInRank = commIdx * blockPerCommInRank;

            // 每个AICore通过步进方式，分工处理不同的计算块
            for (uint32_t blockIdxInComm = aicoreIndex; blockIdxInComm < actualBlockPerComm; blockIdxInComm += aicoreNum) {
                // 计算当前块在整个问题中的逻辑ID和目标Rank
                uint32_t loopIdxInRank = commBlockOffsetInRank + blockIdxInComm % actualBlockPerCommInRank;
                uint32_t targetRankIdx = blockIdxInComm / actualBlockPerCommInRank;
                
                // 从调度器获取当前块的坐标和实际形状
                GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdxInRank);
                GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
                GemmCoord offsetCoord = blockCoord * blockShape;
                
                // 计算A、B矩阵的偏移地址
                auto rankOffsetA = problemShapeInRank.GetCoordMK() * Catlass::MakeCoord<uint32_t>(targetRankIdx, 0);
                auto blockOffsetA = offsetCoord.GetCoordMK() + rankOffsetA;
                auto blockOffsetB = offsetCoord.GetCoordKN();
                
                // 决定计算结果的存储位置
                MatrixCoord blockOffsetStore;
                AscendC::GlobalTensor<ElementC> gmStore;
                Catlass::layout::RowMajor layoutStore;
                if (targetRankIdx == params.rankIdx) {
                    // 如果是为本Rank计算，结果直接存入最终的累加器gmC_accum
                    blockOffsetStore = offsetCoord.GetCoordMN();
                    gmStore = gmC_accum;
                    layoutStore = params.layoutC_accum;
                } else {
                    // 如果是为其他Rank计算，结果存入对称内存的工作空间gmC_workspace
                    blockOffsetStore = MatrixCoord{layoutCRow(Catlass::MakeCoord<int>(stageId, blockIdxInComm, 0)), 0};
                    gmStore = gmC_workspace;
                    layoutStore = layoutC;
                }
                
                // 获取最终的线性地址偏移
                int64_t offsetA = params.layoutA.GetOffset(blockOffsetA);// m*k
                int64_t offsetB = params.layoutB.GetOffset(blockOffsetB);//k*n
                int64_t offsetBias = params.layoutBias.GetOffset(Catlass::MakeCoord(blockOffsetB[1]));
                int64_t offsetStore = layoutStore.GetOffset(blockOffsetStore);
                
                
                // 执行矩阵乘法计算 (融合偏置加法)
                // 注意: 此处假定使用的BlockMmad总是支持偏置加法接口。
                // 主机端代码负责实例化正确的BlockMmad版本。
                blockMmad( gmA[offsetA], params.layoutA, 
                           gmB[offsetB], params.layoutB, 
                           gmStore[offsetStore], layoutStore, 
                           gmBias[offsetBias], actualBlockShape );
            }
            // 计算完成，设置标志通知AIV可以开始处理
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(flagAicFinishStore[stageId]);
        }
        // 最终同步，确保所有计算都已完成
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    //
    // AIV (AI Vector) 的内核实现：主要负责通信、加偏置、反量化等后处理操作
    //
    template <> CATLASS_DEVICE void operator()<AscendC::AIV>(Params &params) {
        // ... 省略与AIC中类似的初始化代码 ...
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

        // 初始化Reduce-Scatter通信对象
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
        CommScheduler commScheduler(params.rankIdx, params.rankSize, commCoreSplit, commShape, commBlockShape, dLoopsInRank);
        MatrixCoord actualCommShapeInRank = commShape / Catlass::MakeCoord<uint32_t>(params.rankSize, 1);

        auto layoutCommLogicShape = Catlass::MakeCoord<int>(1, dLoopsInRank, commBlockShape.row());
        auto layoutComm = layout::AffineRankN<3>::Packed(layoutCommLogicShape);

        // --- 主循环：与AIC流水线对应 ---
        for (uint32_t commIdx = 0; commIdx < commLoops; ++commIdx) {
            uint32_t stageId = commIdx % WORKSPACE_STAGES;
            // ... 省略处理最后一个循环的边界情况代码 ...
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

            // 等待AIC完成计算
            Catlass::Arch::CrossCoreWaitFlag(flagAicFinishStore[stageId]);
            // --- Reduce-Scatter 通信操作 ---
            shmemx_barrier_all_vec(); // 所有Rank在此同步
            AscendC::SetAtomicAdd<ElementC>(); // 设置硬件原子加功能
            AscendC::PipeBarrier<PIPE_ALL>();

            // ... 它会把所有Rank的gmC_accum中的结果进行求和，并将结果分片写回每个Rank各自的gmC_accum中 ...
            reduceScatter.AllocEventID();
            if (aivIndex == 0 && aicoreIndex < commAicoreNum) {
                for (uint32_t commLoopIdx = aicoreIndex; commLoopIdx < commCoreLoops; commLoopIdx += commAicoreNum) {
                    MatrixCoord commBlockCoord = commScheduler.GetBlockIdx(commLoopIdx);
                    MatrixCoord blockOffset = commScheduler.template GetBlockOffset<ReduceScatter::RemoteCopyMode, 
                                                ReduceScatter::RemoteCopyDirect>(commBlockCoord, layoutComm);
                    MatrixCoord actualCommBlockShape = commScheduler.template GetActualBlockShape<ReduceScatter::RemoteCopyMode, 
                                                ReduceScatter::RemoteCopyDirect>(commBlockCoord, layoutComm);
                    MatrixCoord blockOffsetInRank = blockOffset % actualCommShapeInRank;

                    uint32_t remoteRankIdx = commBlockCoord.column();
                    if (remoteRankIdx == params.rankIdx) { continue; }

                    auto offsetIn = stageOffset + blockOffset;
                    auto offsetOut = commOffsetInRank + blockOffsetInRank;
                    auto globalLoopIdx = offsetOut.row() / blockShapeMN.row();

                    reduceScatter(blockShapeMN, offsetOut, offsetIn, actualCommBlockShape, 
                                 gmC_accum, params.layoutC_accum, globalLoopIdx, 
                                 remoteRankIdx % params.rankSize);
                }       
            }
            reduceScatter.ReleaseEventID();
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetAtomicNone();
            AscendC::PipeBarrier<PIPE_ALL>();

            shmemx_barrier_all_vec(); // 再次同步
            // 通知AIC，AIV已经处理完当前阶段的缓冲区，AIC可以开始写入新数据
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishCompute[stageId]);
        }
        
        // --- 反量化操作 ---
        // 在所有计算和通信完成后，对最终的int32结果进行反量化，得到bfloat16输出
        uint32_t M_per_rank = params.problemShape.m() / params.rankSize;
        uint32_t N = params.problemShape.n();
        GemmCoord problemShapeEpilogue{M_per_rank, N, 1};
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

        // 初始化反量化Epilogue对象
        Dequant dequantEpilogue(resource, params.dequantParams);

        auto cord = Dequant::TileShape::ToCoord();
        typename Dequant::EpilogueTileSwizzle tileScheduler(problemShapeEpilogue.GetCoordMN(), cord);
        uint32_t tileLoops = tileScheduler.GetLoops();

        // AIV分工处理不同的块
        for(uint32_t i = coreIdx; i < tileLoops; i += coreNum) {
            auto tileCoord = tileScheduler.GetTileCoord(i);
            auto actualTileShape = tileScheduler.GetActualTileShape(tileCoord);
            auto acc_offset = tileCoord * cord;
            // 调用反量化Epilogue
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
    // --- 成员变量 ---
    Catlass::Arch::CrossCoreFlag flagAicFinishStore[WORKSPACE_STAGES]; // 用于AIC->AIV同步的标志
    Catlass::Arch::CrossCoreFlag flagAivFinishCompute[WORKSPACE_STAGES]; // 用于AIV->AIC同步的标志
    Catlass::Arch::Resource<ArchTag> resource; // 硬件资源对象
};

} // namespace Catcoc::DGemm::Kernel

#endif // CATCOC_DGEMM_KERNEL_QUANT_MATMUL_REDUCE_SCATTER_HPP
