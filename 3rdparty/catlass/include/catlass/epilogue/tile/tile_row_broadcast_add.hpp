#ifndef CATLASS_EPILOGUE_TILE_TILE_ROW_BROADCAST_ADD_HPP
#define CATLASS_EPILOGUE_TILE_TILE_ROW_BROADCAST_ADD_HPP

#include "catlass/catlass.hpp"

namespace Catlass::Epilogue::Tile {

template <
    class ArchTag_,
    class ComputeType_,
    class TileShape_
>
struct TileRowBroadcastAdd {
    using ArchTag = ArchTag_;
    using ElementCompute = typename ComputeType_::Element;
    using TileShape = TileShape_;

    CATLASS_DEVICE
    TileRowBroadcastAdd() {}

    CATLASS_DEVICE
    void operator()(
        AscendC::LocalTensor<ElementCompute> const &ubOut,
        AscendC::LocalTensor<ElementCompute> const &ubIn0,
        AscendC::LocalTensor<ElementCompute> const &ubIn1
    )
    {
        constexpr uint32_t maxRepeatTimes = 255;
        constexpr uint32_t eleNumPerBlk = BYTE_PER_BLK / sizeof(ElementCompute);

        constexpr uint32_t blkNumPerColumn = TileShape::COLUMN / eleNumPerBlk;
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = blkNumPerColumn;
        repeatParams.src0RepStride = blkNumPerColumn;
        repeatParams.src1RepStride = 0;

        constexpr uint32_t rowNumPerCompute = maxRepeatTimes;
        constexpr uint32_t colNumPerCompute = BYTE_PER_VECTOR_FRACTAL / sizeof(ElementCompute);
        for (uint32_t rowOffset = 0; rowOffset < TileShape::ROW; rowOffset += rowNumPerCompute) {
            uint32_t residueM = TileShape::ROW - rowOffset;
            uint8_t repeatTimes = static_cast<uint8_t>((residueM > rowNumPerCompute) ? rowNumPerCompute : residueM);
            for (uint32_t colOffset = 0; colOffset < TileShape::COLUMN; colOffset += colNumPerCompute) {
                uint32_t residueN = TileShape::COLUMN - colOffset;
                uint64_t mask = (residueN > colNumPerCompute) ? colNumPerCompute : residueN;
                AscendC::Add(
                    ubOut[rowOffset * TileShape::COLUMN + colOffset],
                    ubIn0[rowOffset * TileShape::COLUMN + colOffset],
                    ubIn1[colOffset],
                    mask, repeatTimes, repeatParams
                );
            }
        }
    }
};

} // namespace Catlass::Epilogue::Tile

#endif
