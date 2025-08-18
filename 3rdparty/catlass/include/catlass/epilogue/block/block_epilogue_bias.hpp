#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_BIAS_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_BIAS_HPP

#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "kernel_operator.h"

namespace Catlass::Epilogue::Block {

template <
    typename DispatchPolicy_,
    typename CType_,
    typename BiasType_,
    typename DType_,
    typename TileCopy_,
    typename EpilogueTileSwizzle_
>
class BlockEpilogueBias {
public:
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileShape = typename DispatchPolicy::TileShape;
    using CType = CType_;
    using BiasType = BiasType_;
    using DType = DType_;
    using TileCopy = TileCopy_;
    using EpilogueTileSwizzle = EpilogueTileSwizzle_;
    using ElementC = typename CType::Element;
    using LayoutC = typename CType::Layout;
    using ElementBias = typename BiasType::Element;
    using LayoutBias = typename BiasType::Layout;
    using ElementD = typename DType::Element;
    using LayoutD = typename DType::Layout;
    using Resource = Arch::Resource<ArchTag>;
    using CopyGmToUbC = typename TileCopy::CopyGmToUbC;
    using CopyGmToUbBias = typename TileCopy::CopyGmToUbX;
    using CopyUbToGmD = typename TileCopy::CopyUbToGmD;
    
    struct Params {
        GM_ADDR ptr_bias;
        LayoutBias layout_bias;
        uint32_t rank;
        CATLASS_DEVICE Params() = default;
        CATLASS_DEVICE Params(
            GM_ADDR ptr_c_, LayoutC const &layout_c_,
            GM_ADDR ptr_bias_, LayoutBias const &layout_bias_,
            uint32_t rank_
        ) : ptr_bias(ptr_bias_), layout_bias(layout_bias_), rank(rank_){}
    };

public:
    CATLASS_DEVICE
    BlockEpilogueBias(
        Resource &resource,
        Params const &params
    ) :
        params_(params)
    {
        uint32_t ub_offset = 0;
        ub_c_ = resource.ubBuf.template GetBufferByByte<ElementC>(ub_offset);
        ub_offset += TileShape::COUNT * sizeof(ElementC);
        ub_bias_ = resource.ubBuf.template GetBufferByByte<ElementBias>(ub_offset);
    }

    CATLASS_DEVICE
    void operator()(
        GemmCoord const &problem_shape,
        GemmCoord const &block_coord,
        GemmCoord const &actual_block_shape,
        AscendC::GlobalTensor<ElementC> g_c_in,
        LayoutC const &layout_c_in
    ) {
        AscendC::GlobalTensor<ElementBias> g_bias;
        g_bias.SetGlobalBuffer(reinterpret_cast<__gm__ ElementBias *>(params_.ptr_bias));

        EpilogueTileSwizzle tile_swizzle(actual_block_shape.GetCoordMN(), TileShape::ToCoord());
        uint32_t tile_loops = tile_swizzle.GetLoops();
        uint32_t subblock_idx = AscendC::GetSubBlockIdx();
        uint32_t subblock_num = AscendC::GetSubBlockNum();

        for (uint32_t i = subblock_idx; i < tile_loops; i += subblock_num) {
            auto tile_coord = tile_swizzle.GetTileCoord(i);
            auto actual_tile_shape = tile_swizzle.GetActualTileShape(tile_coord);
            auto tile_offset = tile_coord * TileShape::ToCoord();

            auto g_c_tile = g_c_in[layout_c_in.GetOffset(tile_offset)];
            auto layout_c_tile = layout_c_in.GetTileLayout(actual_tile_shape);
            auto ub_tile_stride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
            LayoutC layout_ub_c(actual_tile_shape, ub_tile_stride);
            CopyGmToUbC copy_c;
            copy_c(ub_c_, g_c_tile, layout_ub_c, layout_c_tile);

            auto bias_tile_offset_n = tile_offset[1];
            auto bias_tile_shape_n = actual_tile_shape[1];
            auto g_bias_tile = g_bias[params_.layout_bias.GetOffset(MakeCoord(bias_tile_offset_n))];
            auto layout_bias_tile = params_.layout_bias.GetTileLayout(MakeCoord(bias_tile_shape_n));
            LayoutBias layout_ub_bias(bias_tile_shape_n);
            CopyGmToUbBias copy_bias;
            copy_bias(ub_bias_, g_bias_tile, layout_ub_bias, layout_bias_tile);

            AscendC::PipeBarrier<PIPE_ALL>();

            if (subblock_idx == 0 && i == 0) {
                cce::printf("BIAS-DEBUG at rank: %d, Before Add: C[0]=%d, Bias[0]=%d\n", params_.rank, ub_c_.GetValue(0), ub_bias_.GetValue(0));
            }

            constexpr uint32_t maxRepeatTimes = 255;
            constexpr uint32_t eleNumPerBlk = BYTE_PER_BLK / sizeof(ElementC);
            constexpr uint32_t blkNumPerColumn = TileShape::COLUMN / eleNumPerBlk;
            AscendC::BinaryRepeatParams repeatParams;
            repeatParams.dstBlkStride = 1;
            repeatParams.src0BlkStride = 1;
            repeatParams.src1BlkStride = 1;
            repeatParams.dstRepStride = blkNumPerColumn;
            repeatParams.src0RepStride = blkNumPerColumn;
            repeatParams.src1RepStride = 0;

            constexpr uint32_t rowNumPerCompute = maxRepeatTimes;
            constexpr uint32_t colNumPerCompute = BYTE_PER_VECTOR_FRACTAL / sizeof(ElementC);

            for (uint32_t rowOffset = 0; rowOffset < actual_tile_shape[0]; rowOffset += rowNumPerCompute) {
                uint32_t residueM = actual_tile_shape[0] - rowOffset;
                uint8_t repeatTimes = static_cast<uint8_t>((residueM >= rowNumPerCompute) ? rowNumPerCompute : residueM);
                for (uint32_t colOffset = 0; colOffset < actual_tile_shape[1]; colOffset += colNumPerCompute) {
                    uint32_t residueN = actual_tile_shape[1] - colOffset;
                    uint64_t mask = (residueN >= colNumPerCompute) ? colNumPerCompute : residueN;
                    
                    AscendC::Add(
                        ub_c_[rowOffset * TileShape::COLUMN + colOffset],
                        ub_c_[rowOffset * TileShape::COLUMN + colOffset],
                        ub_bias_[colOffset],
                        mask, repeatTimes, repeatParams
                    );
                }
            }

            if (subblock_idx == 0 && i == 0) {
                cce::printf("BIAS-DEBUG at rank: %d,  After Add: C[0]=%d\n", params_.rank, ub_c_.GetValue(0));
            }
            
            AscendC::PipeBarrier<PIPE_ALL>();

            CopyUbToGmD copy_d;
            copy_d(g_c_tile, ub_c_, layout_c_tile, layout_ub_c);
        }
    }

private:
    Params params_;
    AscendC::LocalTensor<ElementC> ub_c_;
    AscendC::LocalTensor<ElementBias> ub_bias_;
};

}

#endif
