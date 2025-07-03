#include "kernel_operator.h"

#include "shmem_api.h"

class kernel_put_num {
public:
    __aicore__ inline kernel_put_num() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gva_gm = (__gm__ float *)gva;
        dev_gm = (__gm__ float *)dev;

        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();
    }
    __aicore__ inline void Process()
    {
        shmem_put_float_mem_nbi(gva_gm, dev_gm, rank_size * 16, rank);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
private:
    __gm__ float *gva_gm;
    __gm__ float *dev_gm;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void put_num_test(GM_ADDR gva, GM_ADDR dev)
{
    kernel_put_num op;
    op.Init(gva, dev);
    op.Process();
}

void test_put(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)
{
    put_num_test<<<block_dim, nullptr, stream>>>(gva, dev);
}

class kernel_get_num {
public:
    __aicore__ inline kernel_get_num() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gva_gm = (__gm__ float *)gva;
        dev_gm = (__gm__ float *)dev;

        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();

        // 1x512 Bytes Buffer
        pipe.InitBuffer(buf_queue, 1, 512);
    }
    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> buf_tensor = buf_queue.AllocTensor<float>();
        __ubuf__ float *buf = (__ubuf__ float *)buf_tensor.address_.bufferAddr;

        for (int i = 0; i < rank_size; i++) {
            shmem_mte_get_mem_nbi(dev_gm + 16 * i, gva_gm, buf, (uint32_t)256, 16, i % rank_size, EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        buf_queue.FreeTensor(buf_tensor);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> buf_queue;
    __gm__ float *gva_gm;
    __gm__ float *dev_gm;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void get_num_test(GM_ADDR gva, GM_ADDR dev)
{
    kernel_get_num op;
    op.Init(gva, dev);
    op.Process();
}

void test_get(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)
{
    get_num_test<<<block_dim, nullptr, stream>>>(gva, dev);
}