#include "kernel_operator.h"
#include "lowlevel/smem_shm_aicore_base_api.h"

#include "shmem_api.h"

class KernelUBPutNumNonContiguous {
public:
    __aicore__ inline KernelUBPutNumNonContiguous() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gvaGm = (__gm__ float *)gva;
        devGm = (__gm__ float *)dev;

        rank = smem_shm_get_global_rank();
        rankSize = smem_shm_get_global_rank_size();

        // set GM Buffer
        srcGlobal.SetGlobalBuffer(devGm);
        dstGlobal.SetGlobalBuffer(gvaGm);

        // 1x512 Bytes Buffer
        pipe.InitBuffer(bufQueue, 1, 512);
    }
    __aicore__ inline void Process()
    {
        int row = 16;
        int col = 32;
        int total_size = row * col;

        AscendC::LocalTensor<float> bufTensor = bufQueue.AllocTensor<float>();
        __ubuf__ float *buf = (__ubuf__ float *)bufTensor.address_.bufferAddr;
        AscendC::DataCopy(bufTensor, srcGlobal, total_size);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

        non_contiguous_copy_param copyParams;
        copyParams.repeat = row / 2;
        copyParams.length = col / 2;
        copyParams.src_ld = col;
        copyParams.dst_ld = col / 2;

        shmem_mte_put_mem_nbi(dstGlobal, bufTensor, copyParams, (rank + 1) % rankSize, EVENT_ID0);
        shmem_mte_put_mem_nbi(gvaGm + row * col / 4, buf + row * col / 2, copyParams, (rank + 1) % rankSize, EVENT_ID0);

        shmem_put_float_mem_nbi(dstGlobal[row * col / 2], bufTensor[col / 2], copyParams, (rank + 1) % rankSize);
        shmem_put_float_mem_nbi(gvaGm + row * col / 2 + row * col / 4, buf + row * col / 2 + col / 2, copyParams, (rank + 1) % rankSize);

        bufQueue.FreeTensor(bufTensor);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> bufQueue;

    AscendC::GlobalTensor<float> srcGlobal, dstGlobal;
    __gm__ float *gvaGm;
    __gm__ float *devGm;

    int64_t rank;
    int64_t rankSize;
};

extern "C" __global__ __aicore__ void UBPutNumNonContiguousTest(GM_ADDR gva, GM_ADDR dev)
{
    KernelUBPutNumNonContiguous op;
    op.Init(gva, dev);
    op.Process();
}

void TestUBNonContiguousPut(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)
{
    UBPutNumNonContiguousTest<<<block_dim, nullptr, stream>>>(gva, dev);
}

class KernelUBGetNumNonContiguous {
public:
    __aicore__ inline KernelUBGetNumNonContiguous() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gvaGm = (__gm__ float *)gva;
        devGm = (__gm__ float *)dev;

        rank = smem_shm_get_global_rank();
        rankSize = smem_shm_get_global_rank_size();

        // set GM Buffer
        srcGlobal.SetGlobalBuffer(gvaGm);
        dstGlobal.SetGlobalBuffer(devGm);

        // 1x512 Bytes Buffer
        pipe.InitBuffer(bufQueue, 1, 512);
    }
    __aicore__ inline void Process()
    {
        int row = 16;
        int col = 32;
        int total_size = row * col;

        AscendC::LocalTensor<float> bufTensor = bufQueue.AllocTensor<float>();
        __ubuf__ float *buf = (__ubuf__ float *)bufTensor.address_.bufferAddr;

        non_contiguous_copy_param copyParams;
        copyParams.repeat = row / 2;
        copyParams.length = col / 2;
        copyParams.src_ld = col / 2;
        copyParams.dst_ld = col;

        shmem_mte_get_mem_nbi(buf, gvaGm, copyParams, (rank + 1) % rankSize, EVENT_ID0);
        shmem_mte_get_mem_nbi(bufTensor[col / 2], srcGlobal[row * col / 2], copyParams, (rank + 1) % rankSize, EVENT_ID0);

        shmem_get_float_mem_nbi(buf + row * col / 2, gvaGm + row * col / 4, copyParams, (rank + 1) % rankSize);
        shmem_get_float_mem_nbi(bufTensor[row * col / 2 + col / 2], srcGlobal[row * col / 2 + row * col / 4], copyParams, (rank + 1) % rankSize);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

        AscendC::DataCopy(dstGlobal, bufTensor, total_size);
        bufQueue.FreeTensor(bufTensor);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> bufQueue;
    AscendC::GlobalTensor<float> srcGlobal, dstGlobal;
    __gm__ float *gvaGm;
    __gm__ float *devGm;

    int64_t rank;
    int64_t rankSize;
};

extern "C" __global__ __aicore__ void UBGetNonContiguousNumTest(GM_ADDR gva, GM_ADDR dev)
{
    KernelUBGetNumNonContiguous op;
    op.Init(gva, dev);
    op.Process();
}

void TestUBNonContiguousGet(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)
{
    UBGetNonContiguousNumTest<<<block_dim, nullptr, stream>>>(gva, dev);
}