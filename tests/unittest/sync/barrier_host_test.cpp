#include <iostream>
#include <string>
#include <gtest/gtest.h>

#include "acl/acl.h"
#include "shmem_api.h"
#include "internal/host_device/shmemi_types.h"

using namespace std;
extern int32_t test_gnpu_num;
extern const char* test_global_ipport;
extern int testFirstNpu;
extern void TestMutilTask(std::function<void(int32_t, int32_t, uint64_t)> func, uint64_t local_mem_size, int32_t processCount);
extern void TestInit(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size, aclrtStream *st);
extern void TestFinalize(aclrtStream stream, int32_t device_id);

extern void fetchAddrDo(void* stream, uint8_t* sync_array, uint8_t* sync_counter);
extern void barrierDo(void* stream, uint8_t *stub);
extern void increaseDo(void* stream, uint8_t *addr, int32_t rank_id, int32_t rank_size);

constexpr int32_t SHMEM_BARRIER_TEST_NUM = 3;

static void fetchFlags(uint32_t rank_id, int32_t t, void *sync_array, void *sync_counter) {
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

static void TestBarrierWhiteBox(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size)
{
    ASSERT_EQ(test_gnpu_num, 8); // fetchFlags函数仅支持8卡验证
    int32_t device_id = rank_id % test_gnpu_num + testFirstNpu;
    aclrtStream stream;
    TestInit(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    void *sync_array, *sync_counter;
    // get flag addr
    void *syncArrayHost, *syncCounterHost;
    void *syncArrayDevice, *syncCounterDevice;
    ASSERT_EQ(aclrtMallocHost(&syncArrayHost, sizeof(void *)), 0);
    ASSERT_EQ(aclrtMallocHost(&syncCounterHost, sizeof(void *)), 0);
    ASSERT_EQ(aclrtMalloc(&syncArrayDevice, sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST), 0);
    ASSERT_EQ(aclrtMalloc(&syncCounterDevice, sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST), 0);

    fetchAddrDo(stream, (uint8_t *)syncArrayDevice, (uint8_t *)syncCounterDevice);
    ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
    ASSERT_EQ(aclrtMemcpy(syncArrayHost, sizeof(void *), syncArrayDevice, sizeof(void *), ACL_MEMCPY_DEVICE_TO_HOST), 0);
    ASSERT_EQ(aclrtMemcpy(syncCounterHost, sizeof(void *), syncCounterDevice, sizeof(void *), ACL_MEMCPY_DEVICE_TO_HOST), 0);

    sync_array = (void *) *((uint64_t *) syncArrayHost);
    sync_counter = (void *) *((uint64_t *) syncCounterHost);

    ASSERT_EQ(aclrtFreeHost(syncArrayHost), 0);
    ASSERT_EQ(aclrtFreeHost(syncCounterHost), 0);
    ASSERT_EQ(aclrtFree(syncArrayDevice), 0);
    ASSERT_EQ(aclrtFree(syncCounterDevice), 0);

    // run barrier and check flags
    for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
        barrierDo(stream, nullptr);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        fetchFlags(rank_id, i, sync_array, sync_counter);
        if (i < SHMEM_BARRIER_TEST_NUM) {
            sleep(1); // 确保check完成再进行下一轮
        }
    }

    TestFinalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

static void TestBarrierBlackBox(int32_t rank_id, int32_t n_ranks, uint64_t local_mem_size) {
    int32_t device_id = rank_id % test_gnpu_num + testFirstNpu;
    aclrtStream stream;
    TestInit(rank_id, n_ranks, local_mem_size, &stream);
    ASSERT_NE(stream, nullptr);

    uint64_t *addrDev = (uint64_t *)shmem_malloc(sizeof(uint64_t));
    uint64_t *addrHost;
    ASSERT_EQ(aclrtMallocHost((void **)&addrHost, sizeof(uint64_t)), 0);
    *addrHost = 0;

    for (int32_t i = 1; i <= SHMEM_BARRIER_TEST_NUM; i++) {
        std::cout << "[TEST] barriers test blackbox rank_id: " << rank_id << " time: " << i << std::endl;
        increaseDo(stream, (uint8_t *)addrDev, rank_id, n_ranks);
        ASSERT_EQ(aclrtSynchronizeStream(stream), 0);
        ASSERT_EQ(aclrtMemcpy(addrHost, sizeof(uint64_t), addrDev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST), 0);
        ASSERT_EQ((*addrHost), i);
        if (i < SHMEM_BARRIER_TEST_NUM) {
            sleep(1); // 确保check完成再进行下一轮
        }
    }

    ASSERT_EQ(aclrtFreeHost(addrHost), 0);

    TestFinalize(stream, device_id);
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}

TEST(TestBarrierApi, TestBarrierWhiteBox)
{
    const int32_t processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    TestMutilTask(TestBarrierWhiteBox, local_mem_size, processCount);
}

TEST(TestBarrierApi, TestBarrierBlackBox)
{
    const int32_t processCount = test_gnpu_num;
    uint64_t local_mem_size = 1024UL * 1024UL * 1024;
    TestMutilTask(TestBarrierBlackBox, local_mem_size, processCount);
}