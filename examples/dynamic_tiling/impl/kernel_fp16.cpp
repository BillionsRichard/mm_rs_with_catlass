#include "impl/kernel/matmul_allreduce.h"
#include "impl/kernel/allgather_matmul.h"
#include "impl/kernel/matmul_reduce_scatter.h"
#include "impl/kernel/allgather_matmul_with_gather_result.h"

using namespace AscendC;

using ElementA = half;
using ElementB = half;
using ElementC = half;

using LayoutA0 = Catlass::layout::RowMajor;
using LayoutB0 = Catlass::layout::RowMajor;

using LayoutA1 = Catlass::layout::ColumnMajor;
using LayoutB1 = Catlass::layout::ColumnMajor;

using LayoutC = Catlass::layout::RowMajor;

void LaunchMatmulAllReduceFP16(
    void *stream, uint64_t fftsAddr,
    uint8_t *a, uint8_t *b, uint8_t *c,
    uint8_t *gatherA, uint8_t *workspace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)gatherA;
    (void)workspace;
    if (!transA && !transB) {
        MatmulAllReduce<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        MatmulAllReduce<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    } else if (transA && !transB) {
        MatmulAllReduce<ElementA, LayoutA1, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    } else {
        MatmulAllReduce<ElementA, LayoutA1, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    }
}

void LaunchAllGatherMatmulFP16(
    void *stream, uint64_t fftsAddr,
    uint8_t *a, uint8_t *b, uint8_t *c,
    uint8_t *gatherA, uint8_t *workspace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)gatherA;
    (void)workspace;
    if (!transA && !transB) {
        AllGatherMatmul<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        AllGatherMatmul<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    }
}

void LaunchMatmulReduceScatterFP16(
    void *stream, uint64_t fftsAddr,
    uint8_t *a, uint8_t *b, uint8_t *c,
    uint8_t *gatherA, uint8_t *workspace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)gatherA;
    (void)workspace;
    if (!transA && !transB) {
        MatmulReduceScatter<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        MatmulReduceScatter<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    } else if (transA && !transB) {
        MatmulReduceScatter<ElementA, LayoutA1, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    } else {
        MatmulReduceScatter<ElementA, LayoutA1, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, symmetricPtr, cocTiling);
    }
}

void LaunchAllGatherMatmulWithGatherResultFP16(
    void *stream, uint64_t fftsAddr,
    uint8_t *a, uint8_t *b, uint8_t *c,
    uint8_t *gatherA, uint8_t *workspace,
    uint8_t *symmetricPtr, CocTilingParams& cocTiling,
    uint32_t transA, uint32_t transB)
{
    (void)workspace;
    if (!transA && !transB) {
        AllGatherMatmulWithGatherResult<ElementA, LayoutA0, ElementB, LayoutB0, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, gatherA, symmetricPtr, cocTiling);
    } else if (!transA && transB) {
        AllGatherMatmulWithGatherResult<ElementA, LayoutA0, ElementB, LayoutB1, ElementC, LayoutC>
            <<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, a, b, c, gatherA, symmetricPtr, cocTiling);
    }
}