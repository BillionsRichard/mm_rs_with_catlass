#include "kernel_operator.h"
#include "lowlevel/smem_shm_aicore_base_api.h"

#include "shmem_api.h"

class KernelPutNum {
public:
    __aicore__ inline KernelPutNum() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gva_gm = (__gm__ float *)gva;
        devGm = (__gm__ float *)dev;

        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();
    }
    __aicore__ inline void Process()
    {
        shmem_put_float_mem_nbi(gva_gm, devGm, rank_size * 16, rank);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
private:
    __gm__ float *gva_gm;
    __gm__ float *devGm;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void PutNumTest(GM_ADDR gva, GM_ADDR dev)
{
    KernelPutNum op;
    op.Init(gva, dev);
    op.Process();
}

void TestPut(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)
{
    PutNumTest<<<block_dim, nullptr, stream>>>(gva, dev);
}

class KernelGetNum {
public:
    __aicore__ inline KernelGetNum() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gva_gm = (__gm__ float *)gva;
        devGm = (__gm__ float *)dev;

        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();

        // 1x512 Bytes Buffer
        pipe.InitBuffer(bufQueue, 1, 512);
    }
    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> bufTensor = bufQueue.AllocTensor<float>();
        __ubuf__ float *buf = (__ubuf__ float *)bufTensor.address_.bufferAddr;

        for (int i = 0; i < rank_size; i++) {
            shmem_mte_get_mem_nbi(devGm + 16 * i, gva_gm, buf, (uint32_t)256, 16, i % rank_size, EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        bufQueue.FreeTensor(bufTensor);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> bufQueue;
    __gm__ float *gva_gm;
    __gm__ float *devGm;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void GetNumTest(GM_ADDR gva, GM_ADDR dev)
{
    KernelGetNum op;
    op.Init(gva, dev);
    op.Process();
}

void TestGet(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)
{
    GetNumTest<<<block_dim, nullptr, stream>>>(gva, dev);
}