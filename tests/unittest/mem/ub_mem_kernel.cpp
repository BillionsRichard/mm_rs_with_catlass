#include "kernel_operator.h"
#include "lowlevel/smem_shm_aicore_base_api.h"

#include "shmem_api.h"

class KernelUBPutNum {
public:
    __aicore__ inline KernelUBPutNum() {}
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
        int total_size = 512;
        int localSize = 128;

        AscendC::LocalTensor<float> bufTensor = bufQueue.AllocTensor<float>();
        __ubuf__ float *buf = (__ubuf__ float *)bufTensor.address_.bufferAddr;
        AscendC::DataCopy(bufTensor, srcGlobal, total_size);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

        shmem_mte_put_mem_nbi(dstGlobal, bufTensor, localSize, (rank + 1) % rankSize, EVENT_ID0);
        shmem_mte_put_mem_nbi(gvaGm + localSize * 1, buf + localSize * 1, localSize, (rank + 1) % rankSize, EVENT_ID0);

        shmem_put_float_mem_nbi(dstGlobal[localSize * 2], bufTensor[localSize * 2], localSize, (rank + 1) % rankSize);
        shmem_put_float_mem_nbi(gvaGm + localSize * 3, buf + localSize * 3, localSize, (rank + 1) % rankSize);

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

extern "C" __global__ __aicore__ void UBPutNumTest(GM_ADDR gva, GM_ADDR dev)
{
    KernelUBPutNum op;
    op.Init(gva, dev);
    op.Process();
}

void TestUBPut(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)
{
    UBPutNumTest<<<block_dim, nullptr, stream>>>(gva, dev);
}

class KernelUBGetNum {
public:
    __aicore__ inline KernelUBGetNum() {}
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
        int total_size = 512;
        int localSize = 128;
        
        AscendC::LocalTensor<float> bufTensor = bufQueue.AllocTensor<float>();
        __ubuf__ float *buf = (__ubuf__ float *)bufTensor.address_.bufferAddr;

        shmem_mte_get_mem_nbi(buf, gvaGm, localSize, (rank + 1) % rankSize, EVENT_ID0);
        shmem_mte_get_mem_nbi(bufTensor[localSize * 1], srcGlobal[localSize * 1], localSize, (rank + 1) % rankSize, EVENT_ID0);

        shmem_get_float_mem_nbi(buf + localSize * 2, gvaGm + localSize * 2, localSize, (rank + 1) % rankSize);
        shmem_get_float_mem_nbi(bufTensor[localSize * 3], srcGlobal[localSize * 3], localSize, (rank + 1) % rankSize);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        float scalar = 55.0f;
        AscendC::Adds(bufTensor, bufTensor, scalar, total_size);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

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

extern "C" __global__ __aicore__ void UBGetNumTest(GM_ADDR gva, GM_ADDR dev)
{
    KernelUBGetNum op;
    op.Init(gva, dev);
    op.Process();
}

void TestUBGet(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)
{
    UBGetNumTest<<<block_dim, nullptr, stream>>>(gva, dev);
}