#include "kernel_operator.h"

#include "shmem_api.h"
#include "../utils/func_type.h"

#define KERNEL_PUT_NUM(NAME, TYPE)                                                                                              \
class kernel_##NAME##_put_num {                                                                                                 \
public:                                                                                                                         \
    __aicore__ inline kernel_##NAME##_put_num() {}                                                                              \
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)                                                                       \
    {                                                                                                                           \
        gva_gm = (__gm__ TYPE *)gva;                                                                                            \
        dev_gm = (__gm__ TYPE *)dev;                                                                                            \
                                                                                                                                \
        rank = smem_shm_get_global_rank();                                                                                      \
        rank_size = smem_shm_get_global_rank_size();                                                                            \
                                                                                                                                \
        /* 1x4096 Bytes Buffer */                                                                                               \           
        pipe.InitBuffer(buf_queue, 1,4096);                                                                                     \
    }                                                                                                                           \
    __aicore__ inline void Process()                                                                                            \
    {                                                                                                                           \
        AscendC::LocalTensor<TYPE> buf_tensor = buf_queue.AllocTensor<TYPE>();                                                  \
        __ubuf__ TYPE *buf = (__ubuf__ TYPE *)buf_tensor.address_.bufferAddr;                                                   \
        shmem_mte_put_mem_nbi(gva_gm, dev_gm, buf, (uint32_t)256, rank_size / 2 * 16, rank, EVENT_ID0);                         \
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);                                                             \
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);                                                            \
        shmem_put_##NAME##_mem_nbi(gva_gm + rank_size / 2 * 16, dev_gm + rank_size / 2 * 16, rank_size / 2 * 16, rank);         \
        buf_queue.FreeTensor(buf_tensor);                                                                                       \
    }                                                                                                                           \
private:                                                                                                                        \
    AscendC::TPipe pipe;                                                                                                        \
    AscendC::TQue<AscendC::TPosition::VECIN, 2> buf_queue;                                                                      \
                                                                                                                                \
    __gm__ TYPE *gva_gm;                                                                                                        \
    __gm__ TYPE *dev_gm;                                                                                                        \
                                                                                                                                \
    int64_t rank;                                                                                                               \
    int64_t rank_size;                                                                                                          \
}

SHMEM_FUNC_TYPE_KERNEL(KERNEL_PUT_NUM);

#define PUT_NUM_TEST(NAME, TYPE)                                                            \
extern "C" __global__ __aicore__ void put_##NAME##_num_test(GM_ADDR gva, GM_ADDR dev)       \
{                                                                                           \
    kernel_##NAME##_put_num op;                                                             \
    op.Init(gva, dev);                                                                      \
    op.Process();                                                                           \
}

SHMEM_FUNC_TYPE_KERNEL(PUT_NUM_TEST);

#define TEST_PUT(NAME, TYPE)                                                            \
void test_##NAME##_put(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)    \
{                                                                                       \
    put_##NAME##_num_test<<<block_dim, nullptr, stream>>>(gva, dev);                    \
}

SHMEM_FUNC_TYPE_KERNEL(TEST_PUT);

#define KERNEL_GET_NUM(NAME, TYPE)                                                                                                  \                                                                     
class kernel_##NAME##_get_num {                                                                                                     \
public:                                                                                                                             \
    __aicore__ inline kernel_##NAME##_get_num() {}                                                                                  \
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)                                                                           \
    {                                                                                                                               \
        gva_gm = (__gm__ TYPE *)gva;                                                                                                \
        dev_gm = (__gm__ TYPE *)dev;                                                                                                \
                                                                                                                                    \
        rank = smem_shm_get_global_rank();                                                                                          \
        rank_size = smem_shm_get_global_rank_size();                                                                                \
                                                                                                                                    \
        /* 1x4096 Bytes Buffer */                                                                                                   \
        pipe.InitBuffer(buf_queue, 1, 4096);                                                                                        \
    }                                                                                                                               \
    __aicore__ inline void Process()                                                                                                \
    {                                                                                                                               \
        AscendC::LocalTensor<TYPE> buf_tensor = buf_queue.AllocTensor<TYPE>();                                                      \
        __ubuf__ TYPE *buf = (__ubuf__ TYPE *)buf_tensor.address_.bufferAddr;                                                       \
                                                                                                                                    \
        for (int i = 0; i < rank_size / 2; i++) {                                                                                   \
            shmem_mte_get_mem_nbi(dev_gm + 16 * i, gva_gm, buf, (uint32_t)256, 16, i % rank_size, EVENT_ID0);                       \
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);                                                             \
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);                                                            \
        }                                                                                                                           \
                                                                                                                                    \
        for (int i = rank_size / 2; i < rank_size; i++) {                                                                           \
            shmem_get_##NAME##_mem_nbi(dev_gm + 16 * i, gva_gm, 16, i % rank_size);                                                 \          
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);                                                             \
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);                                                            \
        }                                                                                                                           \
                                                                                                                                    \
        buf_queue.FreeTensor(buf_tensor);                                                                                           \
    }                                                                                                                               \
private:                                                                                                                            \
    AscendC::TPipe pipe;                                                                                                            \
    AscendC::TQue<AscendC::TPosition::VECIN, 2> buf_queue;                                                                          \
    __gm__ TYPE *gva_gm;                                                                                                            \
    __gm__ TYPE *dev_gm;                                                                                                            \
                                                                                                                                    \
    int64_t rank;                                                                                                                   \
    int64_t rank_size;                                                                                                              \
}

SHMEM_FUNC_TYPE_KERNEL(KERNEL_GET_NUM);

#define GET_NUM_TEST(NAME,TYPE)                                                         \
extern "C" __global__ __aicore__ void get_##NAME##_num_test(GM_ADDR gva, GM_ADDR dev)   \
{                                                                                       \
    kernel_##NAME##_get_num op;                                                         \
    op.Init(gva, dev);                                                                  \
    op.Process();                                                                       \
}

SHMEM_FUNC_TYPE_KERNEL(GET_NUM_TEST);

#define TEST_GET(NAME, TYPE)                                                            \
void test_##NAME##_get(uint32_t block_dim, void* stream, uint8_t* gva, uint8_t* dev)    \
{                                                                                       \
    get_##NAME##_num_test<<<block_dim, nullptr, stream>>>(gva, dev);                    \
}

SHMEM_FUNC_TYPE_KERNEL(TEST_GET);