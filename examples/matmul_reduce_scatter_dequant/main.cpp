#include <acl/acl.h>

#include <iostream>
#include <vector>
#include <cstring>

// from catlass
#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/block/block_epilogue_per_token_dequant.hpp"

// shmem_host
#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_rma.h"
#include "host/shmem_host_team.h"

// utils
#include "utils.h"

#include "catcoc/catcoc.hpp"
#include "catcoc/comm_epilogue/comm_dispatch_policy.hpp"
#include "catcoc/comm_epilogue/block/comm_block_epilogue.hpp"
#include "catcoc/comm_epilogue/block/comm_block_swizzle.hpp"
#include "catcoc/comm_epilogue/tile/tile_remote_copy.hpp"
#include "catcoc/gemm/dispatch_policy.hpp"
#include "catcoc/detail/remote_copy_type.hpp"
#include "catcoc/dgemm/kernel/matmul_reduce_scatter_dequant.hpp"

static uint32_t gNpuNum = 8;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

using namespace AscendC;
using namespace Catcoc;

constexpr uint32_t BLOCK_NUM = 20;
constexpr int32_t BLOCK_SIZE_16 = 16;

using LayoutA = Catlass::layout::RowMajor;
using LayoutB = Catlass::layout::RowMajor;
using LayoutC = Catlass::layout::RowMajor;
using LayoutD = Catlass::layout::RowMajor;

CATLASS_GLOBAL
void ShmemMatmulReduceScatterDequant(
    uint64_t fftsAddr,
    GM_ADDR x1, GM_ADDR x2, GM_ADDR scale_x1, GM_ADDR scale_x2, GM_ADDR bias,
    GM_ADDR c_accum, GM_ADDR d_out, GM_ADDR symmetricPtr,
    uint32_t m, uint32_t n, uint32_t k, shmem_team_t teamIdx = 0
)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    // Define ArchTag
    using ArchTag = Catlass::Arch::AtlasA2;

    Catlass::GemmCoord problemShape{m, n, k};

    // Prepare comm address
    uint32_t rank = shmem_team_my_pe(teamIdx);
    uint32_t rankSize = shmem_team_n_pes(teamIdx);

    // Define layouts
    Catlass::layout::RowMajor layoutA{m, k, k};
    Catlass::layout::RowMajor layoutB{k, n, n};
    Catlass::layout::RowMajor layoutC_accum{m / rankSize, n, n};
    Catlass::layout::RowMajor layoutD_out{m / rankSize, n, n};
    Catlass::layout::RowMajor layout_scale_x1{m, 1, 1};
    Catlass::layout::RowMajor layout_scale_x2{n, 1, 1};
    Catlass::layout::VectorLayout layout_bias(n);

    // Define types for BlockMmad
    using AType = Catlass::Gemm::GemmType<int8_t, LayoutA>;
    using BType = Catlass::Gemm::GemmType<int8_t, LayoutB>;
    using CType = Catlass::Gemm::GemmType<int32_t, LayoutC>; // Accumulator is int32
    using BiasType = Catlass::Gemm::GemmType<int32_t, Catlass::layout::VectorLayout>;

    constexpr bool enableUnitFlag = true;
    // Use the dispatch policy that supports fused bias
    using MmadDispatchPolicy = Catlass::Gemm::MmadAtlasA2PingpongCondBias<enableUnitFlag>;
    using L1TileShape = Catlass::GemmShape<128, 256, 256>;
    using L0TileShape = Catlass::GemmShape<128, 256, 64>;
    using BlockMmad = Catlass::Gemm::Block::BlockMmad<MmadDispatchPolicy,
        L1TileShape, L0TileShape, AType, BType, CType, BiasType>;

    // Define types for ReduceScatter Epilogue (int32 -> int32)
    using ReduceScatterCType = CType;
    using ReduceScatterDType = CType;

    using BlockScheduler = typename Catlass::Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;
    using CommBlockScheduler = CommEpilogue::Block::BlockCommSwizzle<0>;

    using CopyDirect = Catcoc::detail::CopyDirect;
    using TileRemoteCopy = CommEpilogue::Tile::TileRemoteCopy<ArchTag, ReduceScatterCType, ReduceScatterDType, CopyDirect::Get>;
    using TileScheduler = Catlass::Epilogue::Tile::EpilogueIdentityTileSwizzle;

    using CommBlockShape = Catlass::MatrixShape<64, 256>;
    using CommCoreSplit = Catlass::MatrixShape<20, 1>;

    constexpr uint32_t ubStages = 2;
    using ReduceScatterTileShape = Catlass::MatrixShape<32, 256>;
    using ReduceScatterDispatch = CommEpilogue::EpilogueAtlasA2CommRemoteCopy<ubStages,
        Catcoc::detail::CopyMode::Scatter>;
    using BlockEpilogueReduceScatter = CommEpilogue::Block::CommBlockEpilogue<
        ReduceScatterDispatch,
        ReduceScatterCType, ReduceScatterDType,
        CommCoreSplit,
        CommBlockShape,
        ReduceScatterTileShape, TileRemoteCopy, TileScheduler
    >;

    // Define types for PerTokenDequant Epilogue
    using namespace Catlass::Epilogue;
    using DequantCType = CType; // int32 accumulator
    using DequantScaleType = Catlass::Gemm::GemmType<float, Catlass::layout::VectorLayout>;
    using DequantPerTokenScaleType = Catlass::Gemm::GemmType<float, Catlass::layout::VectorLayout>;
    using DequantDType = Catlass::Gemm::GemmType<half, LayoutD>;
    using DequantDispatchPolicy = EpilogueAtlasA2PerTokenDequant<2>;

    using EpilogueTileShape = Catlass::MatrixShape<64, 128>;
    using ComputeType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;

    using TileRowBroadcastMul = Tile::TileRowBroadcastMul<ArchTag, ComputeType, EpilogueTileShape>;
    using TileBroadcastOneBlk = Tile::TileBroadcastOneBlk<ArchTag, ComputeType, 64>;
    using TileOneBlkColumnBroadcastMul = Tile::TileOneBlkColumnBroadcastMul<ArchTag, ComputeType, EpilogueTileShape>;

    using TileCopy = Tile::TileCopy<Catlass::Arch::AtlasA2, DequantCType, DequantScaleType,
                                    DequantPerTokenScaleType, DequantDType>;

    using EpilogueTileSwizzle = Tile::EpilogueIdentityTileSwizzle;

    using BlockEpilogueDequant = Block::BlockEpilogue<
        DequantDispatchPolicy, DequantCType, DequantScaleType,
        DequantPerTokenScaleType, DequantDType, TileRowBroadcastMul,
        TileBroadcastOneBlk, TileOneBlkColumnBroadcastMul, TileCopy,
        EpilogueTileSwizzle>;

    // BiasAdd Epilogue is no longer needed, as it's fused into BlockMmad.

    constexpr uint32_t workspaceStages = 2;
    constexpr uint32_t commInterval = 10;
    using QuantMatmulReduceScatterKernel = DGemm::Kernel::QuantMatmulReduceScatter<
        BlockMmad,
        BlockEpilogueReduceScatter,
        BlockEpilogueDequant,
        BlockScheduler,
        CommBlockScheduler,
        workspaceStages
    >;
    Catlass::GemmCoord problemShapeInRank = problemShape / Catlass::MakeCoord<uint32_t>(rankSize, 1, 1);
    BlockScheduler matmulBlockScheduler(problemShapeInRank, Catlass::MakeCoord(L1TileShape::M, L1TileShape::N));

    Catlass::layout::RowMajor layoutPeerMemStore{
        L1TileShape::M * commInterval * BLOCK_NUM * workspaceStages, L1TileShape::N,
        L1TileShape::N
    };

    typename BlockEpilogueReduceScatter::Params reduceScatterParams{
    };
    
    uint32_t m_per_rank = m / rankSize;
    uint32_t scale_x1_offset = rank * m_per_rank;
    typename BlockEpilogueDequant::Params dequantParams{
        reinterpret_cast<__gm__ float *>(scale_x2), Catlass::layout::VectorLayout(n),
        reinterpret_cast<__gm__ float *>(scale_x1) + scale_x1_offset, Catlass::layout::VectorLayout(m_per_rank),
        reinterpret_cast<__gm__ half *>(d_out), layoutD_out
    };

    // Prepare params
    typename QuantMatmulReduceScatterKernel::Params params{
        problemShape,
        rank, rankSize, teamIdx,
        x1, layoutA,
        x2, layoutB,
        bias, layout_bias, // Pass bias pointer directly
        symmetricPtr,
        reduceScatterParams,
        dequantParams,
        c_accum, layoutC_accum,
        d_out, layoutD_out,
        commInterval
    };

    // Call kernel
    QuantMatmulReduceScatterKernel matmulCommKernel;
    matmulCommKernel(params);
}

struct Options {
    static constexpr auto HELPER =
       "Usage: matmul_reduce_scatter_quant rank_size rank_id ip_port m n k data_path [device_id_list]\n";

    int rankSize;
    int rankId;
    std::string ipPort;
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::string dataPath;
    std::vector<int> deviceIdList{};

    int Parse(int argc, char **argv)
    {
        enum ArgsIndex {
            RANK_SIZE_INDEX = 1,
            RANK_ID_INDEX,
            IP_PORT_INDEX,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            DATA_PATH_INDEX,
            DEVICE_LIST_INDEX,
            INDEX_MAX
        };

        if (argc > INDEX_MAX) {
            printf(HELPER);
            return -1;
        }

        rankSize = std::atoi(argv[RANK_SIZE_INDEX]);
        rankId = std::atoi(argv[RANK_ID_INDEX]);
        ipPort = argv[IP_PORT_INDEX];
        m = std::atoi(argv[M_INDEX]);
        n = std::atoi(argv[N_INDEX]);
        k = std::atoi(argv[K_INDEX]);
        dataPath = argv[DATA_PATH_INDEX];
        if (argc > DEVICE_LIST_INDEX) {
            char *idListStr = argv[DEVICE_LIST_INDEX];
            for (char *idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) {
                deviceIdList.push_back(std::atoi(idToken));
            }
        } else {
            for (size_t i = 0; i < rankSize; ++i) {
                deviceIdList.push_back(i);
            }
        }
        return 0;
    }

    std::string GetDataPath(std::string const &fileName = "") const
    {
        return dataPath + "/" + fileName;
    }
};

int main(int argc, char **argv)
{
    int status = SHMEM_SUCCESS;
    Options options;
    if (options.Parse(argc, argv) != 0) {
        std::cerr << "Invalid arguments\n";
        return 1;
    }
    int rankSize = options.rankSize;
    int rankId = options.rankId;
    std::string ipPort = options.ipPort;
    uint32_t m = options.m;
    uint32_t n = options.n;
    uint32_t k = options.k;
    int32_t deviceId = options.deviceIdList[rankId];

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipPort << "\n";

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));
    status = shmem_set_conf_store_tls(false, nullptr, 0);
    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rankId, rankSize, gNpuMallocSpace, ipPort.c_str(), &attributes);
    status = shmem_init_attr(attributes);
    status = shmem_init_status();

    // Prepare FFTS address
    // uint64_t fftsAddr{0};
    // uint32_t fftsLen{0};
    // ACL_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    // Memory sizes
    size_t x1Size = static_cast<size_t>(m) * k * sizeof(int8_t);
    size_t x2Size = static_cast<size_t>(k) * n * sizeof(int8_t);
    size_t scaleX1Size = static_cast<size_t>(m) * sizeof(float);
    size_t scaleX2Size = static_cast<size_t>(n) * sizeof(float);
    size_t biasSize = static_cast<size_t>(n) * sizeof(int32_t);
    size_t cAccumSize = static_cast<size_t>(m) * n * sizeof(int32_t) / rankSize;
    size_t dOutSize = static_cast<size_t>(m) * n * sizeof(bfloat16_t) / rankSize;

    // Allocate and copy x1
    uint8_t *x1Device, *x1Host;
    ACL_CHECK(aclrtMalloc((void **)(&x1Device), x1Size, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&x1Host), x1Size));
    std::string x1_filename = "x1_gm_rank" + std::to_string(rankId) + ".bin";
    ReadFile(options.GetDataPath(x1_filename), x1Host, x1Size);
    ACL_CHECK(aclrtMemcpy(x1Device, x1Size, x1Host, x1Size, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate and copy x2
    uint8_t *x2Device, *x2Host;
    ACL_CHECK(aclrtMalloc((void **)(&x2Device), x2Size, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&x2Host), x2Size));
    std::string x2_filename = "x2_gm_rank" + std::to_string(rankId) + ".bin";
    ReadFile(options.GetDataPath(x2_filename), x2Host, x2Size);
    ACL_CHECK(aclrtMemcpy(x2Device, x2Size, x2Host, x2Size, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate and copy scale_x1
    uint8_t *scaleX1Device, *scaleX1Host;
    ACL_CHECK(aclrtMalloc((void **)(&scaleX1Device), scaleX1Size, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&scaleX1Host), scaleX1Size));
    ReadFile(options.GetDataPath("scale_x1_gm.bin"), scaleX1Host, scaleX1Size);
    ACL_CHECK(aclrtMemcpy(scaleX1Device, scaleX1Size, scaleX1Host, scaleX1Size, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate and copy scale_x2
    uint8_t *scaleX2Device, *scaleX2Host;
    ACL_CHECK(aclrtMalloc((void **)(&scaleX2Device), scaleX2Size, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&scaleX2Host), scaleX2Size));
    ReadFile(options.GetDataPath("scale_x2_gm.bin"), scaleX2Host, scaleX2Size);
    ACL_CHECK(aclrtMemcpy(scaleX2Device, scaleX2Size, scaleX2Host, scaleX2Size, ACL_MEMCPY_HOST_TO_DEVICE));

    // Allocate and copy bias
    uint8_t *biasDevice, *biasHost;
    if (rankId == 0) {
        ACL_CHECK(aclrtMalloc((void **)(&biasDevice), biasSize, ACL_MEM_MALLOC_HUGE_FIRST));
        ACL_CHECK(aclrtMallocHost((void **)(&biasHost), biasSize));
        ReadFile(options.GetDataPath("bias_gm.bin"), biasHost, biasSize);
        ACL_CHECK(aclrtMemcpy(biasDevice, biasSize, biasHost, biasSize, ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        biasDevice = nullptr;
    }


    // Allocate intermediate and final output buffers
    uint8_t *cAccumDevice;
    ACL_CHECK(aclrtMalloc((void **)(&cAccumDevice), cAccumSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(cAccumDevice, cAccumSize, 0, cAccumSize));
    uint8_t *dOutDevice, *dOutHost;
    ACL_CHECK(aclrtMalloc((void **)(&dOutDevice), dOutSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMallocHost((void **)(&dOutHost), dOutSize));

    // Allocate shared memory workspace
    void *symmPtr = shmem_malloc((204 * 1024 * 1024) * sizeof(int32_t));
    uint8_t *symmetricPtr = (uint8_t *)symmPtr;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    for (int i = 0; i < 1; i++) {
        ShmemMatmulReduceScatterDequant<<<BLOCK_NUM, nullptr, stream>>>(
            shmemx_get_ffts_config(),
            x1Device, x2Device, scaleX1Device, scaleX2Device, biasDevice,
            cAccumDevice, dOutDevice, symmetricPtr, m, n, k);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(dOutHost, dOutSize, dOutDevice, dOutSize, ACL_MEMCPY_DEVICE_TO_HOST));
    
    WriteFile(options.GetDataPath("output.bin"), dOutHost, dOutSize, options.rankId * dOutSize);
    if (rankId == 0) {
        std::printf("test finished\n");
    }

    shmem_free(symmPtr);

    ACL_CHECK(aclrtFreeHost(x1Host));
    ACL_CHECK(aclrtFreeHost(x2Host));
    ACL_CHECK(aclrtFreeHost(scaleX1Host));
    ACL_CHECK(aclrtFreeHost(scaleX2Host));
    if (rankId == 0) {
        ACL_CHECK(aclrtFreeHost(biasHost));
        ACL_CHECK(aclrtFree(biasDevice));
    }
    ACL_CHECK(aclrtFreeHost(dOutHost));
    ACL_CHECK(aclrtFree(x1Device));
    ACL_CHECK(aclrtFree(x2Device));
    ACL_CHECK(aclrtFree(scaleX1Device));
    ACL_CHECK(aclrtFree(scaleX2Device));
    ACL_CHECK(aclrtFree(cAccumDevice));
    ACL_CHECK(aclrtFree(dOutDevice));

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << "\n";
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}