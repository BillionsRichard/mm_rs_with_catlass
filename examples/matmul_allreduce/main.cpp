#include <acl/acl.h>

#include <iostream>
#include <vector>


// misc
#include "helper.hpp"
#include "golden.hpp"
#include "fp16_t.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <fstream>
#include <iomanip>
#include <string>
#include <sys/file.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>


// from ascendc-templates
#include "act/act.hpp"
#include "act/arch/arch.hpp"
#include "act/epilogue/dispatch_policy.hpp"
#include "act/epilogue/block/block_epilogue.hpp"
#include "act/epilogue/tile/tile_copy.hpp"
#include "act/epilogue/tile/tile_elemwise_add.hpp"
#include "act/gemm/block/block_mmad.hpp"
#include "act/gemm/block/block_swizzle.hpp"
#include "act/gemm/dispatch_policy.hpp"
#include "act/gemm/kernel/matmul_epilogue.hpp"
#include "act/gemm/gemm_type.hpp"
#include "act/layout/layout.hpp"

// from shmem-templates
// #include "shmem-templates/epilogue/block/epilogue_dynamic_comm.hpp"
#include "shmem-templates/epilogue/block/epilogue_allreduce.hpp"
#include "shmem-templates/epilogue/tile/remote_copy_op.hpp"
#include "shmem-templates/epilogue/block/block_swizzle_dynamic.hpp"
#include "shmem-templates/gemm/kernel/matmul_epilogue_comm.hpp"


// shmem_host
#include "data_utils.h"
#include "shmem_api.h"

// shmem_device
#include "shmem_device_api.h"

static uint32_t gNpuNum = 8;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

using namespace AscendC;
using namespace Act;
using fp16_t = op::fp16_t;


struct CoCTiling {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t m0 = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    uint32_t swizzlDirect = 0;
    uint32_t swizzleOffset = 0;
    uint32_t ubMoveNum = 0;
    uint32_t pValue = 0;
    uint32_t commNpuSplit = 0;
    uint32_t commDataSplit = 0;
    uint32_t lenPerLoop = 0;
};

constexpr uint32_t BLOCK_NUM = 20;
constexpr int32_t BLOCK_SIZE_16 = 16;

inline bool ReadFile(const std::string &filePath, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("Failed to get file");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file.", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s.", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("File size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("File size is larger than buffer size.");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    file.close();
    return true;
}

inline bool WriteFile(const std::string &filePath, const void *buffer, size_t size, size_t offset = 0)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. Buffer is nullptr.");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT, 0666);
    if (!fd) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    // 尝试获取写锁tB =
    if (flock(fd, LOCK_EX) == -1) {
        std::cerr << "Failed to acquire lock: " << strerror(errno) << std::endl;
        close(fd);
        return false;
    }

    // 将文件指针定位到指定的偏移量
    if (lseek(fd, offset, SEEK_SET) == -1) {
        std::cerr << "Failed to seek in file: " << strerror(errno) << std::endl;
        close(fd);
        return false;
    }

    // file.write(static_cast<const char *>(buffer), size);
    // 写入数据
    if (write(fd, static_cast<const char *>(buffer), size) != static_cast<ssize_t>(size)) {
        std::cerr << "Failed to write to file: " << strerror(errno) << std::endl;
    }

    // 释放锁
    flock(fd, LOCK_UN);

    close(fd);
    return true;
}

using LayoutA = layout::RowMajor;
using LayoutB = layout::RowMajor;
using LayoutC = layout::RowMajor;

ACT_GLOBAL
void ShmemMatmulAllReduce(
    uint64_t fftsAddr, GemmCoord problemShape, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR gmWorkspace, CoCTiling cocTiling)
{
    // Set FFTS address
    AscendC::SetSyncBaseAddr(fftsAddr);

    // Define ArchTag
    using ArchTag = Arch::AtlasA2;

    // unzip cocTiling
    uint32_t m = cocTiling.m;
    uint32_t n = cocTiling.n;
    uint32_t k = cocTiling.k;
    uint32_t m0 = cocTiling.m0;
    uint32_t k0 = cocTiling.k0;
    uint32_t n0 = cocTiling.n0;
    uint32_t swizzleOffset = cocTiling.swizzleOffset;
    uint32_t swizzlDirect = cocTiling.swizzlDirect;
    uint32_t pValue = cocTiling.pValue;
    uint32_t commDataSplit = cocTiling.commDataSplit;
    uint32_t commNpuSplit = cocTiling.commNpuSplit;
    uint32_t ubMoveNum = cocTiling.ubMoveNum;
    uint32_t lenPerLoop = cocTiling.lenPerLoop;

    // Prepare comm address
    uint32_t rank = ShmemMype();
    uint32_t rankSize = ShmemNpes();
    using ElementC = half;
    __gm__ ElementC *peerMems[SHM_MAX_RANKS] = {};
    __gm__ void* addrGM = smem_shm_get_extra_context_addr();
    __gm__ ShmemDeviceHostState *deviceState = (__gm__ ShmemDeviceHostState *)addrGM;
    for (int i = 0; i < rankSize; i++) {
        uint64_t remotePtr = reinterpret_cast<uint64_t>(deviceState->p2pHeapBase[i]) + 1024;
        __gm__ void *rankPtr = reinterpret_cast<__gm__ void*>(remotePtr);
        __gm__ ElementC *realPtr = reinterpret_cast<__gm__ ElementC*>(rankPtr);
        peerMems[i] = realPtr;
    }

    // Block level, Define the layout of each input matrix
    layout::RowMajor layoutA{m, k, k};
    layout::RowMajor layoutB{k, n, n};
    layout::RowMajor layoutC{m, n, n};

    GemmCoord blockShape{m0, n0, k0};

    // Block level, define BlockMmad
    constexpr bool enableUnitFlag = true;
    using MmadDispatchPolicy = Gemm::MmadAtlasA2Pingpong<enableUnitFlag>;
    using L1TileShape = GemmShape<128, 256, 256>;
    using L0TileShape = GemmShape<128, 256, 64>;
    using AType = Gemm::GemmType<half, LayoutA>;
    using BType = Gemm::GemmType<half, LayoutB>;
    using CType = Gemm::GemmType<half, LayoutC>;
    using BlockMmad = Gemm::Block::BlockMmad<MmadDispatchPolicy, L1TileShape, L0TileShape, AType, BType, CType>;

    // TODO Block level, define BlockEpilogue
    using ElementStore = half;

    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<7, 1>;        // TODO Need Set Manually
    using CommBlockSwizzle = Gemm::Block::CommBlockSwizzleDynamic;

    using ComputeAttachedReduceScatter = Epilogue::Block::RemoteCopyOp<
        ArchTag, ElementStore, Gemm::Block::ReduceScatterSchedule>;
    using ComputeAttachedAllGather =  Epilogue::Block::RemoteCopyOp<
        ArchTag, ElementStore, Gemm::Block::AllGatherSchedule>;

    // Block level, define BlockEpiloguei
    using BlockAllReduceEpilogue = Epilogue::Block::EpilogueAllReduce<
        BlockScheduler, CommBlockSwizzle,
        ComputeAttachedReduceScatter,
        ComputeAttachedAllGather>;

    // Kernel level
    using MatmulAllReduceKernel = Gemm::Kernel::MatmulEpilogueComm<BlockMmad, BlockAllReduceEpilogue, BlockScheduler>;

    // Prepare EpilogueComm params
    uint32_t maxUbPingPongSize = ubMoveNum / 2; // 8192

    BlockScheduler matmulBlockScheduler(problemShape, MakeCoord(L1TileShape::M, L1TileShape::N));

    MatrixCoord commBlockShape{lenPerLoop / n0, n0};
    MatrixCoord commProcessShape{maxUbPingPongSize / n0, n0};

    CommBlockSwizzle commSwizzle{commBlockShape, rank, rankSize, 0, commDataSplit, commNpuSplit};

    AscendC::GlobalTensor<ElementStore> refC;
    refC.SetGlobalBuffer((__gm__ ElementStore *)c);
    typename BlockAllReduceEpilogue::Params epilogueCommParams{
            refC, layout::RowMajor(m, n, n),
            0,
            (__gm__ ElementC **)(peerMems),
            commBlockShape, commProcessShape,
            matmulBlockScheduler, commSwizzle};

    // Prepare params
    typename MatmulAllReduceKernel::Params params{
        problemShape, blockShape,
        pValue, rank, rankSize,
        a, layoutA,
        b, layoutB,
        reinterpret_cast<GM_ADDR>(peerMems[rank]),
        epilogueCommParams};

    // call kernel
    MatmulAllReduceKernel matmulCommKernel;
    matmulCommKernel(params);
}

extern "C" {
uint32_t GetAscendCoreSyncAddr(void **addr);
}

struct Options {
    static constexpr auto helper = 
       "Usage: matmul_allreduce m n k transA transB [--block m0 n0 k0 --ubMoveNum ubMoveNum --pValue pValue --split commNpuSplit commDataSplit lenPerLoop --swizzle swizzleOffset swizzlDirect]\n";

    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t m0 = 128;
    uint32_t k0 = 256;
    uint32_t n0 = 256;
    uint32_t transA = 0;
    uint32_t transB = 0;
    uint32_t swizzlDirect = 1;
    uint32_t swizzleOffset = 7;
    uint32_t ubMoveNum = 16 * 1024;
    uint32_t pValue = 3;
    uint32_t commNpuSplit = 2;
    uint32_t commDataSplit = 1;
    uint32_t lenPerLoop = m0 * n0 / 2;

    int Parse(int argc, char **argv)
    {
        if (argc < 6) {
            printf(helper);
            return -1;
        }

        uint32_t argIndex = 1;
        m = std::atoi(argv[argIndex++]);
        n = std::atoi(argv[argIndex++]);
        k = std::atoi(argv[argIndex++]);
        transA = std::atoi(argv[argIndex++]);
        transB = std::atoi(argv[argIndex++]);

        while (argIndex < argc) {
            std::string flag = std::string(argv[argIndex++]);

            if (flag == "--pValue") {
                pValue = std::atoi(argv[argIndex++]);
            } else if (flag == "--ubMoveNum") {
                ubMoveNum = std::atoi(argv[argIndex++]);
            } else if (flag == "--split") {
                commNpuSplit = std::atoi(argv[argIndex++]);
                commDataSplit = std::atoi(argv[argIndex++]);
                lenPerLoop = std::atoi(argv[argIndex++]);
            } else if (flag == "--block") {
                m0 = std::atoi(argv[argIndex++]);
                n0 = std::atoi(argv[argIndex++]);
                k0 = std::atoi(argv[argIndex++]);
            } else if (flag == "--swizzle") {
                swizzleOffset = std::atoi(argv[argIndex++]);
                swizzlDirect = std::atoi(argv[argIndex++]);
            } else {
                printf(helper);
                return -1;
            }
        }

        return 0;
    }
};

nt main(int argc, char **argv)
{
    int rankSize = atoi(argv[1]);
    int rankId = atoi(argv[2]);
    std::string ipport = argv[3];
    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipport << std::endl;

    ACL_CHECK(aclInit(nullptr));
    int32_t deviceId = rankId % gNpuNum;
    ACL_CHECK(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));
    ShmemInit(rankId, rankSize, gNpuMallocSpace);

    // Prepare FFTS address
    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    Options options;
    uint32_t m = 1024;
    uint32_t k = 16;
    uint32_t n = 1024;
    uint32_t m0 = 128;
    uint32_t k0 = 256;
    uint32_t n0 = 256;
    uint32_t swizzlDirect = 1;
    uint32_t swizzleOffset = 7;
    uint32_t ubMoveNum = 16 * 1024;
    uint32_t pValue = 3;
    uint32_t commNpuSplit = 2;
    uint32_t commDataSplit = 1;
    uint32_t lenPerLoop = m0 * n0 / 2;

    // m, n, k
    GemmCoord problemShape{m, n, k};

    size_t aSize = static_cast<size_t>(m) * k * sizeof(__fp16);
    size_t bSize = static_cast<size_t>(k) * n * sizeof(__fp16);
    size_t cSize = static_cast<size_t>(m) * n * sizeof(__fp16);
    size_t workspaceSize = static_cast<size_t>(m) * n * sizeof(__fp16);

    uint8_t *aDevice;
    ACL_CHECK(aclrtMalloc((void **)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *aHost;
    ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
    ReadFile("./examples/matmul_allreduce/out/a_gm.bin", aHost, aSize);
    ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *bDevice;
    ACL_CHECK(aclrtMalloc((void **)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
   uint8_t *bHost;
    ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
    ReadFile("./examples/matmul_allreduce/out/b_gm.bin", bHost, bSize);
    ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *cDevice;
    ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSize, ACL_MEM_MALLOC_HUGE_FIRST));
    uint8_t *cHost;
    ACL_CHECK(aclrtMallocHost((void **)(&cHost), cSize));
    ReadFile("./examples/matmul_allreduce/out/c_gm.bin", cHost, cSize);
    ACL_CHECK(aclrtMemcpy(cDevice, cSize, cHost, cSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *deviceWorkspace{nullptr};
    ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&deviceWorkspace), workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    CoCTiling cocTiling;
    cocTiling.m = m;
    cocTiling.n = n;
    cocTiling.k = k;
    cocTiling.m0 = m0;
    cocTiling.n0 = n0;
    cocTiling.k0 = k0;
    cocTiling.swizzleOffset = swizzleOffset;
    cocTiling.swizzlDirect = swizzlDirect;
    cocTiling.pValue = pValue;
    cocTiling.ubMoveNum = ubMoveNum;
    cocTiling.commNpuSplit = commNpuSplit;
    cocTiling.commDataSplit = commDataSplit;
    cocTiling.lenPerLoop = lenPerLoop;

    ACL_CHECK(aclrtSynchronizeStream(stream));
    for (int i = 0; i < 1; i++) {
        ShmemMatmulAllReduce<<<BLOCK_NUM, nullptr, stream>>>(fftsAddr, problemShape, aDevice, bDevice, cDevice, deviceWorkspace, cocTiling);
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    ACL_CHECK(aclrtMemcpy(cHost, cSize, cDevice, cSize, ACL_MEMCPY_DEVICE_TO_HOST));
    if (rankId == 0) {
        WriteFile("./examples/matmul_allreduce/out/output.bin", cHost, cSize);
        std::printf("test finished\n");
    }
    ACL_CHECK(aclrtFreeHost(aHost));
    ACL_CHECK(aclrtFreeHost(bHost));
    ACL_CHECK(aclrtFreeHost(cHost));
    ACL_CHECK(aclrtFree(aDevice));
    ACL_CHECK(aclrtFree(bDevice));
    ACL_CHECK(aclrtFree(cDevice));
    ACL_CHECK(aclrtFree(deviceWorkspace));

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    ShmemFinalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}