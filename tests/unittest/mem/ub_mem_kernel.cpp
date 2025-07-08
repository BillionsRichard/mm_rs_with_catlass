#include "kernel_operator.h"
#include "smem_shm_aicore_base_api.h"

#include "shmem_api.h"

class kernel_ub_put_num {
public:
    __aicore__ inline kernel_ub_put_num() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gva_gm = (__gm__ float *)gva;
        dev_gm = (__gm__ float *)dev;

        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();

        // set GM Buffer
        src_gm.SetGlobalBuffer(dev_gm);
        dst_gm.SetGlobalBuffer(gva_gm);

        // 1x512 Bytes Buffer
        pipe.InitBuffer(buf_queue, 1, 512);
    }
    __aicore__ inline void Process()
    {
        int total_size = 512;
        int local_size = 128;

        AscendC::LocalTensor<float> buf_tensor = buf_queue.AllocTensor<float>();
        __ubuf__ float *buf = (__ubuf__ float *)buf_tensor.address_.bufferAddr;
        AscendC::DataCopy(buf_tensor, src_gm, total_size);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);

        shmem_mte_put_mem_nbi(dst_gm, buf_tensor, local_size, (rank + 1) % rank_size, EVENT_ID0);
        shmem_mte_put_mem_nbi(gva_gm + local_size * 1, buf + local_size * 1, local_size, (rank + 1) % rank_size, EVENT_ID0);

        shmem_put_float_mem_nbi(dst_gm[local_size * 2], buf_tensor[local_size * 2], local_size, (rank + 1) % rank_size);
        shmem_put_float_mem_nbi(gva_gm + local_size * 3, buf + local_size * 3, local_size, (rank + 1) % rank_size);

        buf_queue.FreeTensor(buf_tensor);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> buf_queue;

    AscendC::GlobalTensor<float> src_gm, dst_gm;
    __gm__ float *gva_gm;
    __gm__ float *dev_gm;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void ub_put_num_test(GM_ADDR gva, GM_ADDR dev)
{
    kernel_ub_put_num op;
    op.Init(gva, dev);
    op.Process();
}

void test_ub_put(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)
{
    ub_put_num_test<<<block_dim, nullptr, stream>>>(gva, dev);
}

class KernelUBGetNum {
public:
    __aicore__ inline KernelUBGetNum() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gva_gm = (__gm__ float *)gva;
        dev_gm = (__gm__ float *)dev;

        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();

        // set GM Buffer
        src_gm.SetGlobalBuffer(gva_gm);
        dst_gm.SetGlobalBuffer(dev_gm);

        // 1x512 Bytes Buffer
        pipe.InitBuffer(buf_queue, 1, 512);
    }
    __aicore__ inline void Process()
    {
        int total_size = 512;
        int local_size = 128;
        
        AscendC::LocalTensor<float> buf_tensor = buf_queue.AllocTensor<float>();
        __ubuf__ float *buf = (__ubuf__ float *)buf_tensor.address_.bufferAddr;

        shmem_mte_get_mem_nbi(buf, gva_gm, local_size, (rank + 1) % rank_size, EVENT_ID0);
        shmem_mte_get_mem_nbi(buf_tensor[local_size * 1], src_gm[local_size * 1], local_size, (rank + 1) % rank_size, EVENT_ID0);

        shmem_get_float_mem_nbi(buf + local_size * 2, gva_gm + local_size * 2, local_size, (rank + 1) % rank_size);
        shmem_get_float_mem_nbi(buf_tensor[local_size * 3], src_gm[local_size * 3], local_size, (rank + 1) % rank_size);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        float scalar = 55.0f;
        AscendC::Adds(buf_tensor, buf_tensor, scalar, total_size);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopy(dst_gm, buf_tensor, total_size);
        buf_queue.FreeTensor(buf_tensor);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2> buf_queue;
    AscendC::GlobalTensor<float> src_gm, dst_gm;
    __gm__ float *gva_gm;
    __gm__ float *dev_gm;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void UBGetNumTest(GM_ADDR gva, GM_ADDR dev)
{
    KernelUBGetNum op;
    op.Init(gva, dev);
    op.Process();
}

void test_ub_get(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)
{
    UBGetNumTest<<<block_dim, nullptr, stream>>>(gva, dev);
}