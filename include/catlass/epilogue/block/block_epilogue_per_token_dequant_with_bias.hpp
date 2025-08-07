#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_WITH_BIAS_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_WITH_BIAS_HPP

#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_broadcast.hpp"

namespace Catlass {
namespace Epilogue {
namespace Block {

template <
    typename DispatchPolicy_,
    typename CType_,
    typename ScaleType_,
    typename PerTokenScaleType_,
    typename BiasType_,
    typename DType_,
    typename TileRowBroadcastMul_,
    typename TileBroadcastOneBlk_,
    typename TileOneBlkColumnBroadcastMul_,
    typename TileCopy_,
    typename TileSwizzle_
>
class BlockEpiloguePerTokenDequantWithBias :
    public BlockEpilogue<DispatchPolicy_, CType_, ScaleType_, PerTokenScaleType_, DType_,
                         TileRowBroadcastMul_, TileBroadcastOneBlk_, TileOneBlkColumnBroadcastMul_,
                         TileCopy_, TileSwizzle_>
{
public:

    using Base = BlockEpilogue<DispatchPolicy_, CType_, ScaleType_, PerTokenScaleType_, DType_,
                               TileRowBroadcastMul_, TileBroadcastOneBlk_, TileOneBlkColumnBroadcastMul_,
                               TileCopy_, TileSwizzle_>;

    using DispatchPolicy = typename Base::DispatchPolicy;
    using CType = typename Base::CType;
    using ScaleType = typename Base::ScaleType;
    using PerTokenScaleType = typename Base::PerTokenScaleType;
    using DType = typename Base::DType;
    using TileRowBroadcastMul = typename Base::TileRowBroadcastMul;
    using TileBroadcastOneBlk = typename Base::TileBroadcastOneBlk;
    using TileOneBlkColumnBroadcastMul = typename Base::TileOneBlkColumnBroadcastMul;
    using TileCopy = typename Base::TileCopy;
    using TileSwizzle = typename Base.TileSwizzle;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileShape = typename DispatchPolicy::TileShape;
    using ElementC = typename CType::Element;
    using ElementScale = typename ScaleType::Element;
    using ElementPerTokenScale = typename PerTokenScaleType::Element;
    using ElementD = typename DType::Element;
    using LayoutC = typename CType::Layout;
    using LayoutScale = typename ScaleType::Layout;
    using LayoutPerTokenScale = typename PerTokenScaleType::Layout;
    using LayoutD = typename DType::Layout;

    using BiasType = BiasType_;
    using ElementBias = typename BiasType::Element;
    using LayoutBias = typename BiasType::Layout;

    struct Params {
        typename DispatchPolicy::Params dispatch_policy_params;
        typename TileRowBroadcastMul::Params tile_row_broadcast_mul_params;
        typename TileBroadcastOneBlk::Params tile_broadcast_one_blk_params;
        typename TileOneBlkColumnBroadcastMul::Params tile_one_blk_column_broadcast_mul_params;
        typename TileCopy::Params tile_copy_params;
        ElementScale const *ptr_scale;
        LayoutScale layout_scale;
        ElementPerTokenScale const *ptr_per_token_scale;
        LayoutPerTokenScale layout_per_token_scale;
        ElementBias const *ptr_bias;
        LayoutBias layout_bias;
        ElementD *ptr_d;
        LayoutD layout_d;

        CATLASS_DEVICE
        Params() = default;

        CATLASS_DEVICE
        Params(
            ElementScale const* ptr_scale_,
            LayoutScale layout_scale_,
            ElementPerTokenScale const* ptr_per_token_scale_,
            LayoutPerTokenScale layout_per_token_scale_,
            ElementBias const* ptr_bias_,
            LayoutBias layout_bias_,
            ElementD* ptr_d_,
            LayoutD layout_d_
        ) :
            ptr_scale(ptr_scale_), layout_scale(layout_scale_),
            ptr_per_token_scale(ptr_per_token_scale_), layout_per_token_scale(layout_per_token_scale_),
            ptr_bias(ptr_bias_), layout_bias(layout_bias_),
            ptr_d(ptr_d_), layout_d(layout_d_) {}
    };

private:

    Params params_;
    typename DispatchPolicy::SharedStorage shared_storage_;
    TileRowBroadcastMul tile_row_broadcast_mul_;
    TileBroadcastOneBlk tile_broadcast_one_blk_;
    TileOneBlkColumnBroadcastMul tile_one_blk_column_broadcast_mul_;
    TileCopy tile_copy_;
    TileSwizzle tile_swizzle_;

public:

    CATLASS_DEVICE
    BlockEpiloguePerTokenDequantWithBias(
        typename ArchTag::Resource &resource,
        Params const &params
    ) :
        Base(resource,
             typename Base::Params(params.ptr_scale, params.layout_scale,
                                   params.ptr_per_token_scale, params.layout_per_token_scale,
                                   params.ptr_d, params.layout_d)),
        params_(params),
        tile_row_broadcast_mul_(resource, params.tile_row_broadcast_mul_params),
        tile_broadcast_one_blk_(resource, params.tile_broadcast_one_blk_params),
        tile_one_blk_column_broadcast_mul_(resource, params.tile_one_blk_column_broadcast_mul_params),
        tile_copy_(resource, params.tile_copy_params),
        tile_swizzle_(params.problem_shape, params.tile_shape)
    {}

    CATLASS_DEVICE
    void operator()(
        GemmCoord tile_shape,
        GemmCoord block_tile_coord,
        GemmCoord problem_shape,
        TensorView<ElementC, LayoutC> C,
        TensorView<ElementD, LayoutD> D
    ) {
        // This is a simplified implementation based on assumptions.
        // The actual implementation will be more complex.

        TensorView<ElementScale, LayoutScale> scale_view(params_.ptr_scale, params_.layout_scale);
        TensorView<ElementPerTokenScale, LayoutPerTokenScale> per_token_scale_view(params_.ptr_per_token_scale, params_.layout_per_token_scale);
        TensorView<ElementBias, LayoutBias> bias_view(params_.ptr_bias, params_.layout_bias);

        auto tile_offset = tile_swizzle_.get_tile_offset(block_tile_coord);

        for (int m = 0; m < tile_shape.m(); ++m) {
            for (int n = 0; n < tile_shape.n(); ++n) {
                auto c_coord = GemmCoord(m, n, 0) + tile_offset;

                if (c_coord.m() < problem_shape.m() && c_coord.n() < problem_shape.n()) {

                    ElementC accum_val = C.at(c_coord);
                    ElementBias bias_val = bias_view.at(GemmCoord(0, c_coord.n(), 0));
                    ElementScale scale_val = scale_view.at(GemmCoord(0, c_coord.n(), 0));
                    ElementPerTokenScale per_token_scale_val = per_token_scale_view.at(GemmCoord(c_coord.m(), 0, 0));

                    float dequant_val = (static_cast<float>(accum_val) + static_cast<float>(bias_val)) * static_cast<float>(scale_val) * static_cast<float>(per_token_scale_val);

                    D.at(c_coord) = static_cast<ElementD>(dequant_val);
                }
            }
        }
    }
};

} // namespace Block
} // namespace Epilogue
} // namespace Catlass

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_WITH_BIAS_HPP
