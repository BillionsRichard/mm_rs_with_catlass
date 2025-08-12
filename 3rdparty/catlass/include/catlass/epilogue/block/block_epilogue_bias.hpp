#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_BIAS_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_BIAS_HPP

#include "catlass/epilogue/tile/tile_elemwise_add.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"

namespace Catlass {
namespace Epilogue {
namespace Block {

template <
    typename DispatchPolicy_,
    typename CType_,
    typename BiasType_,
    typename DType_,
    typename TileAdd_,
    typename TileCopy_,
    typename EpilogueTileSwizzle_
>
class BlockEpilogueBias {
public:
    //
    // Type Aliases
    //
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileShape = typename DispatchPolicy::TileShape;
    using CType = CType_;
    using BiasType = BiasType_;
    using DType = DType_;
    using TileAdd = TileAdd_;
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
        GM_ADDR ptr_d;
        LayoutD layout_d;
        GM_ADDR ptr_bias;
        LayoutBias layout_bias;

        CATLASS_DEVICE
        Params() = default;

        CATLASS_DEVICE
        Params(
            GM_ADDR ptr_d_,
            LayoutD const &layout_d_,
            GM_ADDR ptr_bias_,
            LayoutBias const &layout_bias_
        ) : ptr_d(ptr_d_),
            layout_d(layout_d_),
            ptr_bias(ptr_bias_),
            layout_bias(layout_bias_) {}
    };

public:

    CATLASS_DEVICE
    BlockEpilogueBias(
        Resource &resource,
        Params const &params
    ) :
        params_(params),
        tileAdd_()
    {
        size_t ub_offset = 0;
        ub_c_ = resource.ubBuf.template GetBufferByByte<ElementC>(ub_offset);
        ub_offset += TileShape::COUNT * sizeof(ElementC);
        ub_bias_ = resource.ubBuf.template GetBufferByByte<ElementBias>(ub_offset);
        ub_offset += TileShape::COLUMN * sizeof(ElementBias);
        ub_d_ = resource.ubBuf.template GetBufferByByte<ElementD>(ub_offset);
    }

    CATLASS_DEVICE
    void operator()(
        GemmCoord const &problem_shape,
        GemmCoord const &block_coord,
        GemmCoord const &actual_block_shape,
        AscendC::GlobalTensor<ElementC> g_c,
        LayoutC const &layout_c
    ) {
        AscendC::GlobalTensor<ElementD> g_d;
        g_d.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD *>(params_.ptr_d));
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

            auto g_c_tile = g_c[layout_c.GetOffset(tile_offset)];
            auto layout_c_tile = layout_c.GetTileLayout(actual_tile_shape);

            auto bias_tile_offset = tile_offset.template GetCoordByAxis<1>();
            auto bias_tile_shape = actual_tile_shape.template GetCoordByAxis<1>();
            auto g_bias_tile = g_bias[params_.layout_bias.GetOffset(bias_tile_offset)];
            auto layout_bias_tile = params_.layout_bias.GetTileLayout(bias_tile_shape);

            auto g_d_tile = g_d[params_.layout_d.GetOffset(tile_offset)];
            auto layout_d_tile = params_.layout_d.GetTileLayout(actual_tile_shape);

            auto ub_tile_stride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
            LayoutC layout_ub_c(actual_tile_shape, ub_tile_stride);
            LayoutBias layout_ub_bias(bias_tile_shape[0]);
            LayoutD layout_ub_d(actual_tile_shape, ub_tile_stride);

            CopyGmToUbC copy_c;
            copy_c(ub_c_, g_c_tile, layout_ub_c, layout_c_tile);

            CopyGmToUbBias copy_bias;
            copy_bias(ub_bias_, g_bias_tile, layout_ub_bias, layout_bias_tile);

            tileAdd_(ub_d_, ub_c_, ub_bias_);

            CopyUbToGmD copy_d;
            copy_d(g_d_tile, ub_d_, layout_ub_d, layout_d_tile);
        }
    }

private:
    Params params_;
    TileAdd tileAdd_;
    AscendC::LocalTensor<ElementC> ub_c_;
    AscendC::LocalTensor<ElementBias> ub_bias_;
    AscendC::LocalTensor<ElementD> ub_d_;
};

} // namespace Block
} // namespace Epilogue
} // namespace Catlass

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_BIAS_HPP
