#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_BIAS_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_BIAS_HPP

#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"

namespace Catlass {
namespace Epilogue {
namespace Block {

// 偏置加法 Epilogue (后处理) 的模板类
// 功能: 在 GEMM 计算完成后，为结果矩阵 C (累加器) 加上一个偏置向量 Bias
// 设计模式:
//   - 此 Epilogue 采用“内部循环”模式，即它被调用一次来处理一个大的矩阵块(Block)。
//   - 它内部使用 EpilogueTileSwizzle 来将这个大块切分成更小的计算瓦片(Tile)。
//   - 每个 AI Core 的 SubBlock 会处理一部分 Tile。
template <
    typename DispatchPolicy_,     // 分发策略，定义硬件架构、tile 形状等
    typename CType_,              // C 矩阵（累加结果）的类型信息
    typename BiasType_,           // Bias 向量的类型信息
    typename DType_,              // D 矩阵（最终输出）的类型信息
    typename TileCopy_,           // 用于在 GMEM 和 UBUF 之间拷贝数据的类
    typename EpilogueTileSwizzle_  // Epilogue tile 的切分和遍历策略
>
class BlockEpilogueBias {
public:
    //
    // 类型别名定义
    //
    using DispatchPolicy = DispatchPolicy_;                           // 分发策略
    using ArchTag = typename DispatchPolicy::ArchTag;                  // 硬件架构 (e.g., Ascend910B)
    using TileShape = typename DispatchPolicy::TileShape;              // Epilogue 的 Tile 形状
    using CType = CType_;                                              // C 矩阵类型
    using BiasType = BiasType_;                                        // Bias 向量类型
    using DType = DType_;                                              // D 矩阵（输出）类型, 在此实现中与CType相同
    using TileCopy = TileCopy_;                                        // Tile 拷贝算子
    using EpilogueTileSwizzle = EpilogueTileSwizzle_;                  // Tile 遍历逻辑

    using ElementC = typename CType::Element;                          // C 矩阵元素类型
    using LayoutC = typename CType::Layout;                            // C 矩阵布局
    using ElementBias = typename BiasType::Element;                    // Bias 向量元素类型
    using LayoutBias = typename BiasType::Layout;                      // Bias 向量布局
    using ElementD = typename DType::Element;                          // D 矩阵元素类型
    using LayoutD = typename DType::Layout;                            // D 矩阵布局
    using Resource = Arch::Resource<ArchTag>;                          // 硬件资源 (e.g. Unified Buffer)

    // 定义不同拷贝方向的拷贝类
    using CopyGmToUbC = typename TileCopy::CopyGmToUbC;                // GMEM -> UBUF (for C)
    using CopyGmToUbBias = typename TileCopy::CopyGmToUbX;             // GMEM -> UBUF (for Bias)
    using CopyUbToGmD = typename TileCopy::CopyUbToGmD;                // UBUF -> GMEM (for D)

    // 存储 Epilogue 参数的结构体
    struct Params {
        GM_ADDR ptr_c;           // 指向 GMEM 中 C/D 矩阵的指针 (因为是in-place操作，C和D是同一块内存)
        LayoutC layout_c;        // C/D 矩阵的布局
        GM_ADDR ptr_bias;        // 指向 GMEM 中 Bias 向量的指针
        LayoutBias layout_bias;  // Bias 向量的布局
        uint32_t rank;

        CATLASS_DEVICE
        Params() = default;

        CATLASS_DEVICE
        Params(
            GM_ADDR ptr_c_,
            LayoutC const &layout_c_,
            GM_ADDR ptr_bias_,
            LayoutBias const &layout_bias_,
            uint32_t rank_
        ) : ptr_c(ptr_c_),
            layout_c(layout_c_),
            ptr_bias(ptr_bias_),
            layout_bias(layout_bias_),
            rank(rank_) {}
    };

public:
    // 构造函数
    CATLASS_DEVICE
    BlockEpilogueBias(
        Resource &resource,      // 硬件资源，主要用于访问Unified Buffer
        Params const &params     // Epilogue 参数
    ) :
        params_(params)
    {
        uint32_t ub_offset = 0;
        // 为 C tile 在 UB 中分配空间
        ub_c_ = resource.ubBuf.template GetBufferByByte<ElementC>(ub_offset);
        uint32_t aivId = AscendC::GetBlockIdx();
        if (params.rank == 0)
        {
            cce::printf("acdebug BlockEpilogueBias constructor1 at blockIdx: %u, ub_offset: :%u\n", aivId, ub_offset);
        }
        // TileShape: [64, 128] -> count= 64*128
        ub_offset += TileShape::COUNT * sizeof(ElementC);
        // 为 Bias tile 在 UB 中分配空间uctor2 at blockIdx: %u, ub_offset: %u\n", reinterpret_cast<int>aivId, reinterpret_cast<int>ub_offset);
        if (params.rank == 0){
            cce::printf("acdebug BlockEpilogueBias constructor2 at blockIdx: %u, ub_offset:%u\n", aivId, ub_offset);
        }
        ub_bias_ = resource.ubBuf.template GetBufferByByte<ElementBias>(ub_offset);
    }

    // Epilogue 的主执行函数 (operator())
    CATLASS_DEVICE
    void operator()(
        GemmCoord const &problem_shape,       // 整个问题(或当前Block)的形状
        GemmCoord const &block_coord,         // 当前Block在整个问题中的坐标 (此实现中未使用)
        GemmCoord const &actual_block_shape,  // 当前Block的实际形状
        AscendC::GlobalTensor<ElementC> g_c_in, // 输入的累加结果 C (在 GMEM 中)
        LayoutC const &layout_c_in            // 输入 C 矩阵的布局
    ) {
        // 创建指向 GMEM 中输出矩阵 D 和偏置向量 Bias 的 Tensor 对象
        AscendC::GlobalTensor<ElementD> g_d_out;
        g_d_out.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD *>(params_.ptr_c));
        AscendC::GlobalTensor<ElementBias> g_bias;
        g_bias.SetGlobalBuffer(reinterpret_cast<__gm__ ElementBias *>(params_.ptr_bias));

        // 根据 Block 形状和 Tile 形状初始化 Tile 遍历器
        EpilogueTileSwizzle tile_swizzle(actual_block_shape.GetCoordMN(), TileShape::ToCoord());
        uint32_t tile_loops = tile_swizzle.GetLoops();             // 获取总的 Tile 数量
        uint32_t subblock_idx = AscendC::GetSubBlockIdx();         // 获取当前 sub-block 的 ID
        uint32_t subblock_num = AscendC::GetSubBlockNum();         // 获取总的 sub-block 数量

        // 多 aicore 并行处理，每个 aicore(subblock) 处理一部分 Tile
        for (uint32_t i = subblock_idx; i < tile_loops; i += subblock_num) {
            auto tile_coord = tile_swizzle.GetTileCoord(i);                   // 获取当前 Tile 的坐标 (在 Block 内)
            auto actual_tile_shape = tile_swizzle.GetActualTileShape(tile_coord); // 获取当前 Tile 的实际形状
            auto tile_offset = tile_coord * TileShape::ToCoord();             // 计算当前 Tile 在 Block 内的坐标偏移

            // 1. 将累加器 Tile (C) 从 GMEM 拷贝到 UBUF
            auto g_c_tile = g_c_in[layout_c_in.GetOffset(tile_offset)];       // 获取 GMEM 上 C tile 的指针
            auto layout_c_tile = layout_c_in.GetTileLayout(actual_tile_shape); // 获取 C tile 的布局
            auto ub_tile_stride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L); // UBUF 中 tile 的步长
            LayoutC layout_ub_c(actual_tile_shape, ub_tile_stride);            // 定义 UBUF 中 C tile 的布局 (紧凑型)
            CopyGmToUbC copy_c;
            copy_c(ub_c_, g_c_tile, layout_ub_c, layout_c_tile);

            // 2. 将 Bias Tile 从 GMEM 拷贝到 UBUF
            // 修正: bias的偏移量必须根据tile在整个问题中的列坐标来计算
            auto bias_tile_offset_n = tile_offset[1];                          // 获取tile的列坐标
            auto bias_tile_shape_n = actual_tile_shape.column();               // 获取tile的实际宽度
            auto g_bias_tile = g_bias[params_.layout_bias.GetOffset(MakeCoord(bias_tile_offset_n))]; // 获取GMEM上bias tile的指针
            auto layout_bias_tile = params_.layout_bias.GetTileLayout(MakeCoord(bias_tile_shape_n)); // 获取bias tile的布局
            LayoutBias layout_ub_bias(bias_tile_shape_n);                      // 定义UBUF中bias tile的布局
            CopyGmToUbBias copy_bias;
            copy_bias(ub_bias_, g_bias_tile, layout_ub_bias, layout_bias_tile);

            // 使用 barrier 确保 GMEM 到 UBUF 的数据拷贝完成
            AscendC::PipeBarrier<PIPE_ALL>();

            // 3. 执行 C = C + Bias 的操作 (in-place)
            constexpr uint32_t maxRepeatTimes = 255;                           // 向量指令单次执行的最大重复次数
            constexpr uint32_t eleNumPerBlk = BYTE_PER_BLK / sizeof(ElementC);   // 每个 block 的元素数量， 256/4
            constexpr uint32_t blkNumPerColumn = TileShape::COLUMN / eleNumPerBlk; // 每列的 block 数量
            AscendC::BinaryRepeatParams repeatParams;
            repeatParams.dstBlkStride = 1;                                     // 目标操作数 block 间 stride
            repeatParams.src0BlkStride = 1;                                    // 源操作数0 block 间 stride
            repeatParams.src1BlkStride = 1;                                    // 源操作数1 block 间 stride
            repeatParams.dstRepStride = blkNumPerColumn;                       // 目标操作数重复间 stride
            repeatParams.src0RepStride = blkNumPerColumn;                      // 源操作数0重复间 stride
            repeatParams.src1RepStride = 0;                                    // 源操作数1重复间 stride (0 表示广播)

            constexpr uint32_t rowNumPerCompute = maxRepeatTimes;              // 每次向量计算处理的行数 #255
            constexpr uint32_t colNumPerCompute = BYTE_PER_VECTOR_FRACTAL / sizeof(ElementC); // 每次向量计算处理的列数 #64

            // 遍历 Tile 内的每一行每一列，执行加法
            for (uint32_t rowOffset = 0; rowOffset < actual_tile_shape.row(); rowOffset += rowNumPerCompute) {
                uint32_t residueM = actual_tile_shape.row() - rowOffset;  //residueM:2
                uint8_t repeatTimes = static_cast<uint8_t>((residueM >= rowNumPerCompute) ? rowNumPerCompute : residueM);
                for (uint32_t colOffset = 0; colOffset < actual_tile_shape.column(); colOffset += colNumPerCompute) {
                    uint32_t residueN = actual_tile_shape.column() - colOffset; //residueN:128
                    // 修正: AscendC::Add 的 mask 参数在连续模式下应为元素数量，而非位掩码
                    uint64_t mask = (residueN >= colNumPerCompute) ? colNumPerCompute : residueN;//64,
                    
                    // 执行向量加法: ub_c_ + ub_bias_ -> ub_c_
                    AscendC::Add(
                        ub_c_[rowOffset * TileShape::COLUMN + colOffset],
                        ub_c_[rowOffset * TileShape::COLUMN + colOffset],
                        ub_bias_[colOffset],
                        mask, repeatTimes, repeatParams
                    );
                }
            }
            
            // 使用 barrier 确保计算完成
            AscendC::PipeBarrier<PIPE_ALL>();

            // 4. 将结果 Tile 从 UBUF 拷贝回 GMEM
            auto g_d_tile = g_d_out[params_.layout_c.GetOffset(tile_offset)];   // 获取GMEM上D tile的指针
            auto layout_d_tile = params_.layout_c.GetTileLayout(actual_tile_shape); // 获取D tile的布局
            LayoutD layout_ub_d(actual_tile_shape, ub_tile_stride);             // 定义UBUF中D tile的布局
            CopyUbToGmD copy_d;
            copy_d(g_d_tile, ub_c_, layout_ub_d, layout_d_tile);
        }
    }

private:
    Params params_;                                  // Epilogue 参数实例
    AscendC::LocalTensor<ElementC> ub_c_;           // UBUF 上的 C tile
    AscendC::LocalTensor<ElementBias> ub_bias_;     // UBUF 上的 Bias tile
};

} // namespace Block
} // namespace Epilogue
} // namespace Catlass

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_BIAS_HPP
