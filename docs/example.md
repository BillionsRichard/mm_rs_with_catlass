# AllGather
该样例工程位于examples\allgather文件夹下。

实现了对各卡上16个int32_t数据的allgather功能。

## 核函数实现
```c++
#ifndef _KERNEL_ALLGATHER_
#define _KERNEL_ALLGATHER_

#include "kernel_operator.h"
#include "shmem_api.h"

// 纯vec不能全核同步，需添加cube逻辑
SHMEM_DEVICE void cube_guard()
{
    using namespace AscendC;

#ifdef __DAV_C220_CUBE__
    LocalTensor<float> result;
    result.address_.logicPos = (uint8_t)TPosition::CO1;
    result.InitBuffer(0, 256);
    
    LocalTensor<half> left;
    left.address_.logicPos = (uint8_t)TPosition::A2;
    left.InitBuffer(0, 256);

    LocalTensor<half> right;
    right.address_.logicPos = (uint8_t)TPosition::B2;
    right.InitBuffer(0, 256);

    MmadParams param;
    param.m = 16;
    param.n = 16;
    param.k = 16;

    Mmad<float, half, half>(result, left, right, param);
#endif
}

// all_gather简易实现
extern "C" __global__ __aicore__ void device_all_gather_test(GM_ADDR gva, int elements)
{
    int64_t my_rank = shmem_my_pe();
    int64_t pe_size = shmem_n_pes();
    __gm__ int32_t* gva_gm = (__gm__ int32_t *)gva;
    AscendC::PipeBarrier<PIPE_ALL>();
    cube_guard();
    // All Gather
    for (int i = 0; i < pe_size; i++) {
        shmem_put_int32_mem_nbi(gva_gm + elements * my_rank, gva_gm + elements * my_rank, elements, i);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    shmem_barrier_all();
}

void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva, int elements)
{
    device_all_gather_test<<<block_dim, nullptr, stream>>>(gva, elements);
}

#endif  // _KERNEL_ALLGATHER_
```

## host调用

```c++
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "shmem_api.h"

int g_npus = 8;
const char *ipport;
int f_rank = 0;
int f_npu = 0;
extern void allgather_demo(uint32_t block_dim, void* stream, uint8_t* gva, int elements);

int test_shmem_team_all_gather(int rank_id, int n_ranks, uint64_t local_mem_size)
{
    // 初始化ACL和SHMEM
    int32_t device_id = rank_id % g_npus + f_npu;
    int status = 0;
    aclrtStream stream = nullptr;

    status = aclInit(nullptr);
    status = aclrtSetDevice(device_id);
    status = aclrtCreateStream(&stream);

    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rank_id, n_ranks, local_mem_size, ipport, &attributes);
    status = shmem_init_attr(attributes);

    void *ptr = shmem_malloc(1024);

    // 初始化数据
    uint32_t trans_size = 16;
    std::vector<int32_t> input(trans_size, 0);
    for (int i = 0; i < trans_size; i++) {
        input[i] = (rank_id + 10);
    }

    status = aclrtMemcpy(ptr + shmem_my_pe() * trans_size * sizeof(int32_t), trans_size * sizeof(int32_t),
                         input.data(), trans_size * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);

    // AllGather
    allgather_demo(1, stream, (uint8_t *)ptr, trans_size);
    status = aclrtSynchronizeStream(stream);

    // 结果校验打印
    int32_t *y_host;
    size_t input_size = n_ranks * trans_size * sizeof(int32_t);
    status = aclrtMallocHost(reinterpret_cast<void**>(&y_host), input_size);
    status = aclrtMemcpy(y_host, input_size, ptr, input_size, ACL_MEMCPY_DEVICE_TO_HOST);
    
    for (int i = 0; i < n_ranks; i++) {
        if (y_host[trans_size * i] != 10 + i) {
            std::cout << y_host[trans_size * i] << " != " << 10 + i << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    std::cout << "rank: " << rank_id << " [";
    for (int j = 0; j < trans_size * n_ranks; j++) {
        std::cout << y_host[j] << ", ";
    }
    std::cout << "]" << std::endl;
    // 去初始化
    status = aclrtFreeHost(y_host);
    shmem_free(ptr);
    status = shmem_finalize();
    status = aclrtDestroyStream(stream);
    status = aclrtResetDevice(device_id);
    status = aclFinalize();
    return 0;
}

int main(int argc, char *argv[])
{   
    int status = 0;
    int n_ranks = atoi(argv[1]);
    int rank_id = atoi(argv[2]);
    ipport = argv[3];
    g_npus = atoi(argv[4]);
    f_rank = atoi(argv[5]);
    f_npu = atoi(argv[6]);
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    status = test_shmem_team_all_gather(rank_id, n_ranks, local_mem_size);
    std::cout << "[SUCCESS] demo run success in rank " << rank_id << std::endl;
    
    return 0;
}


```