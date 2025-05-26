#include <iostream>
#include <string>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem_api.h"
#include "internal/host_device/shmemi_types.h"

using namespace std;
extern int32_t test_gnpu_num;
extern const char* test_global_ipport;
extern int test_first_npu;
extern void test_mutil_task(std::function<void(int32_t, int32_t, uint64_t)> func, uint64_t local_mem_size, int32_t process_count);
extern void test_init(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void test_finalize(aclrtStream stream, int32_t device_id);

extern void fetch_addr_do(void* stream, uint8_t* sync_array, uint8_t* sync_counter);
extern void barrier_do(void* stream, uint8_t *stub);
extern void increase_do(void* stream, uint8_t *addr, int32_t rank_id, int32_t rank_size);

constexpr int32_t SHMEM_BARRIER_TEST_NUM = 3;

static void fetch_flags(uint32_t rank_id, int32_t t, void *sync_array, void *sync_counter) {
    static int32_t tmp[SHMEMI_SYNCBIT_SIZE / sizeof(int32_t) * 8];

    EXPECT_EQ(aclrtMemcpy(tmp, SHMEMI_SYNCBIT_SIZE, sync_counter, SHMEMI_SYNCBIT_SIZE, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    EXPECT_EQ(tmp[0], t + 1);

    EXPECT_EQ(aclrtMemcpy(tmp, SHMEMI_SYNCBIT_SIZE * 8, sync_array, SHMEMI_SYNCBIT_SIZE * 8, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (int32_t i = 0; i < test_gnpu_num; i++) {
        int32_t val = 0;
        for (int32_t k = 0; k < 3; k++) {
            if (rank_id == (i + (1 << k)) % test_gnpu_num) {
                val = t;
            }
        }

        EXPECT_EQ(tmp[i * SHMEMI_SYNCBIT_SIZE / sizeof(int32_t)], val);
    }
}

static void test_barrier_white_box(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    ASSERT_EQ(test_gnpu_num, 8); // fetchFlags函数仅支持8卡验证
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    void *sync_array, *sync_counter;
    // get flag addr
    void *sync_array_host, *sync_counter_host;
    void *sync_array_device, *sync_counter_device;
    ASSERT_EQ(aclrtMallocHost(&sync_array_host, sizeof(void *)), 0);
    ASSERT_EQ(aclrtMallocHost(&sync_counter_host, sizeof(void *)), 0);
    ASSERT_EQ(aclrtMalloc(&sync_array_device, sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST), 0);
    ASSERT_EQ(aclrtMalloc(&sync_counter_device, sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST), 0);

    fetch_addr_do(stream, (uint8_t *)sync_array_device, (uint8_t *)sync_counter_device);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    ASSERT_EQ(aclrtMemcpy(sync_array_host, sizeof(void *), sync_array_device, sizeof(void *), ACL_MEMCPY_DEVICE_TO_HOST), 0);
    ASSERT_EQ(aclrtMemcpy(sync_counter_host, sizeof(void *), sync_counter_device, sizeof(void *), ACL_MEMCPY_DEVICE_TO_HOST), 0);

    sync_array = (void *) *((uint64_t *) sync_array_host);
    sync_counter = (void *) *((uint64_t *) sync_counter_host);

    ASSERT_EQ(aclrtFreeHost(sync_array_host), 0);
    ASSERT_EQ(aclrtFreeHost(sync_counter_host), 0);
    ASSERT_EQ(aclrtFree(sync_array_device), 0);
    ASSERT_EQ(aclrtFree(sync_counter_device), 0);

    // run barrier and check flags
    for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
        barrier_do(stream, nullptr);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        fetch_flags(rank_id, i, sync_array, sync_counter);
        if (i < SHMEM_BARRIER_TEST_NUM) {
            sleep(1); // 确保check完成再进行下一轮
        }
    }

    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

static void test_barrier_black_box(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size) {
    int32_t device_id = rank_id % test_gnpu_num + test_first_npu;
    aclrtStream stream;
    test_init(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    uint64_t *addr_dev = (uint64_t *)shmem_malloc(sizeof(uint64_t));
    uint64_t *addr_host;
    ASSERT_EQ(aclrtMallocHost((void **)&addr_host, sizeof(uint64_t)), 0);
    *addr_host = 0;

    for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
        std::cout << "[TEST] barriers test blackbox rank_id: " << rank_id << " time: " << i << std::endl;
        increase_do(stream, (uint8_t *)addr_dev, rank_id, n_ranks);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        ASSERT_EQ(aclrtMemcpy(addr_host, sizeof(uint64_t), addr_dev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST), 0);
        ASSERT_EQ((*addr_host), i);
        if (i < SHMEM_BARRIER_TEST_NUM) {
            sleep(1); // 确保check完成再进行下一轮
        }
    }

    ASSERT_EQ(aclrtFreeHost(addr_host), 0);

    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TestBarrierApi, TestBarrierWhiteBox)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_barrier_white_box, local_mem_size, process_count);
}

TEST(TestBarrierApi, TestBarrierBlackBox)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    test_mutil_task(test_barrier_black_box, local_mem_size, process_count);
}