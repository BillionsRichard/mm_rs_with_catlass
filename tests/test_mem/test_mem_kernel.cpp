#include "kernel_operator.h"
#include "low_level_api/smem_shm_aicore_base_api.h"

#include "shmem_device_api.h"

class KernelPutNum {
public:
    __aicore__ inline KernelPutNum() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gvaGm = (__gm__ float *)gva;
        devGm = (__gm__ float *)dev;

        rank = smem_shm_get_global_rank();
        rankSize = smem_shm_get_global_rank_size();

        // 1x512 Bytes Buffer
        pipe.InitBuffer(bufQueue, 1, 512);
    }
    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> bufTensor = bufQueue.AllocTensor<float>();
        __ubuf__ float *buf = (__ubuf__ float *)bufTensor.address_.bufferAddr;

        ShmemMTEPutMem(gvaGm, devGm, buf, (uint32_t)256, rankSize * 16 * sizeof(float), rank, EVENT_ID0);
        AscendC::PipeBarrier<PIPE_ALL>();

        bufQueue.FreeTensor(bufTensor);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> bufQueue;
    __gm__ float *gvaGm;
    __gm__ float *devGm;

    int64_t rank;
    int64_t rankSize;
};

extern "C" __global__ __aicore__ void PutNumTest(GM_ADDR gva, GM_ADDR dev)
{
    KernelPutNum op;
    op.Init(gva, dev);
    op.Process();
}

void TestPut(uint32_t blockDim, void* stream, uint8_t* gva, uint8_t* dev)
{
    PutNumTest<<<blockDim, nullptr, stream>>>(gva, dev);
}

class KernelGetNum {
public:
    __aicore__ inline KernelGetNum() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gvaGm = (__gm__ float *)gva;
        devGm = (__gm__ float *)dev;

        rank = smem_shm_get_global_rank();
        rankSize = smem_shm_get_global_rank_size();

        // 1x512 Bytes Buffer
        pipe.InitBuffer(bufQueue, 1, 512);
    }
    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> bufTensor = bufQueue.AllocTensor<float>();
        __ubuf__ float *buf = (__ubuf__ float *)bufTensor.address_.bufferAddr;

        for (int i = 0; i < rankSize; i++) {
            ShmemMTEGetMem(devGm + 16 * i, gvaGm, buf, (uint32_t)256, 16 * sizeof(float), i % rankSize, EVENT_ID0);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        bufQueue.FreeTensor(bufTensor);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> bufQueue;
    __gm__ float *gvaGm;
    __gm__ float *devGm;

    int64_t rank;
    int64_t rankSize;
};

extern "C" __global__ __aicore__ void GetNumTest(GM_ADDR gva, GM_ADDR dev)
{
    KernelGetNum op;
    op.Init(gva, dev);
    op.Process();
}

void TestGet(uint32_t blockDim, void* stream, uint8_t* gva, uint8_t* dev)
{
    GetNumTest<<<blockDim, nullptr, stream>>>(gva, dev);
}