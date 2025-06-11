#include <iostream>
#include <string>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem_api.h"
#include "shmemi_host_common.h"

extern int32_t test_gnpu_num;
extern const char* test_global_ipport;
extern int test_first_npu;

void test_mutil_task(std::function<void(int32_t, int32_t, uint64_t)> func, uint64_t local_mem_size, int32_t process_count);
void test_init(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size, aclrtStream *st);
void test_finalize(aclrtStream stream, int32_t device_id);

void fetch_addr_do(void* stream, uint8_t* syncArray, uint8_t* syncCounter);
void barrier_do(void* stream, uint8_t *stub);
void increase_do(void* stream, uint8_t *addr, int32_t rankId, int32_t rankSize);
void p2p_chain_do(void *stream, uint8_t *addr, int rank_id, int rank_size);

constexpr int32_t SHMEM_BARRIER_TEST_NUM = 3;

static void fetch_flags(uint32_t rank_id, int32_t t, void *sync_array, void *sync_counter) {
    void* tmp;
    ASSERT_EQ(aclrtMallocHost(&tmp, SHMEMI_SYNCBIT_SIZE * test_gnpu_num), 0);

    EXPECT_EQ(aclrtMemcpy(tmp, SHMEMI_SYNCBIT_SIZE, sync_counter, SHMEMI_SYNCBIT_SIZE, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    EXPECT_EQ(((int32_t*)tmp)[0], t + 1);

    EXPECT_EQ(aclrtMemcpy(tmp, SHMEMI_SYNCBIT_SIZE * test_gnpu_num, sync_array, SHMEMI_SYNCBIT_SIZE * test_gnpu_num, ACL_MEMCPY_DEVICE_TO_HOST), 0);
    for (int32_t i = 1; i < test_gnpu_num; i = i * 2) {
        EXPECT_EQ(((int32_t*)tmp)[((rank_id + test_gnpu_num - i) % test_gnpu_num) * SHMEMI_SYNCBIT_SIZE / sizeof(int32_t)], t);
    }
}

static void test_barrier_white_box(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
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
        shm::shmemi_control_barrier_all();
        fetch_flags(rank_id, i, sync_array, sync_counter);
        shm::shmemi_control_barrier_all();
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
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(uint64_t), 0, sizeof(uint64_t)), 0);
    uint64_t *addr_host;
    ASSERT_EQ(aclrtMallocHost((void **)&addr_host, sizeof(uint64_t)), 0);

    for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
        std::cout << "[TEST] barriers test blackbox rank_id: " << rank_id << " time: " << i << std::endl;
        increase_do(stream, (uint8_t *)addr_dev, rank_id, n_ranks);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        ASSERT_EQ(aclrtMemcpy(addr_host, sizeof(uint64_t), addr_dev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST), 0);
        ASSERT_EQ((*addr_host), i);
        shm::shmemi_control_barrier_all();
    }

    ASSERT_EQ(aclrtFreeHost(addr_host), 0);

    test_finalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

static void test_p2p(int rank_id, int rank_size, uint64_t local_mem_size) {
    aclrtStream stream;
    test_init(rank_id, rank_size, local_mem_size, &stream);

    int32_t *addr_dev = (int32_t *)shmem_malloc(sizeof(int32_t));
    ASSERT_EQ(aclrtMemset(addr_dev, sizeof(int32_t), 0, sizeof(int32_t)), 0);
    p2p_chain_do(stream, (uint8_t *)addr_dev, rank_id, rank_size);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);

    int32_t dev_id = rank_id % test_gnpu_num + test_first_npu;
    test_finalize(stream, dev_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TEST_SYNC_API, test_barrier_white_box)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_barrier_white_box, local_mem_size, process_count);
}

TEST(TEST_SYNC_API, test_barrier_black_box)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_barrier_black_box, local_mem_size, process_count);
}

TEST(TEST_SYNC_API, test_p2p)
{
    const int32_t process_count = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 16;
    test_mutil_task(test_p2p, local_mem_size, process_count);
}