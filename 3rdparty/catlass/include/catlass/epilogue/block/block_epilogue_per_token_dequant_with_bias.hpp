#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_WITH_BIAS_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_PER_TOKEN_DEQUANT_WITH_BIAS_HPP

#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/tensor_view.hpp"

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
class BlockEpiloguePerTokenDequantWithBias
{
public:

    using DispatchPolicy = DispatchPolicy_;
    using CType = CType_;
    using ScaleType = ScaleType_;
    using PerTokenScaleType = PerTokenScaleType_;
    using BiasType = BiasType_;
    using DType = DType_;
    using TileRowBroadcastMul = TileRowBroadcastMul_;
    using TileBroadcastOneBlk = TileBroadcastOneBlk_;
    using TileOneBlkColumnBroadcastMul = TileOneBlkColumnBroadcastMul_;
    using TileCopy = TileCopy_;
    using TileSwizzle = TileSwizzle_;

    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileShape = typename DispatchPolicy::TileShape;
    using ElementC = typename CType::Element;
    using ElementScale = typename ScaleType::Element;
    using ElementPerTokenScale = typename PerTokenScaleType::Element;
    using ElementBias = typename BiasType::Element;
    using ElementD = typename DType::Element;
    using LayoutC = typename CType::Layout;
    using LayoutScale = typename ScaleType::Layout;
    using LayoutPerTokenScale = typename PerTokenScaleType::Layout;
    using LayoutBias = typename BiasType::Layout;
    using LayoutD = typename DType::Layout;


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
        GemmCoord problem_shape;
        GemmCoord tile_shape;

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
    DispatchPolicy dispatch_policy_;
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
        params_(params),
        dispatch_policy_(resource, params.dispatch_policy_params),
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
        
        TensorView<ElementScale const, LayoutScale> scale_view(params_.ptr_scale, params_.layout_scale);
        TensorView<ElementPerTokenScale const, LayoutPerTokenScale> per_token_scale_view(params_.ptr_per_token_scale, params_.layout_per_token_scale);
        TensorView<ElementBias const, LayoutBias> bias_view(params_.ptr_bias, params_.layout_bias);

        auto tile_offset = tile_swizzle_.get_tile_offset(block_tile_coord);

        for (int m = 0; m < tile_shape.m(); ++m) {
            for (int n = 0; n < tile_shape.n(); ++n) {
                auto c_coord = GemmCoord(m, n, 0) + tile_offset;
                
                if (c_coord.m() < problem_shape.m() && c_coord.n() < problem_shape.n()) {
                    
                    ElementC accum_val = C.at(c_coord);
                    ElementBias bias_val = bias_view.at({0, c_coord.n()});
                    ElementScale scale_val = scale_view.at({0, c_coord.n()});
                    ElementPerTokenScale per_token_scale_val = per_token_scale_view.at({c_coord.m(), 0});

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
