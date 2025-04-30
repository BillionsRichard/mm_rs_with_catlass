#include <iostream>
#include <string>
using namespace std;

#include "acl/acl.h"
#include "shmem_host_api.h"
#include "shmemi_host_intf.h"
#include <gtest/gtest.h>
extern int testGlobalRanks;
extern int testGNpuNum;
extern const char* testGlobalIpport;
extern void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t localMemSize, int processCount);


extern void fetchAddrDo(uint8_t* syncArray, uint8_t* syncCounter);

extern void barrierDo(uint8_t *stub);

extern void increaseDo(uint8_t *addr, int rankId, int rankSize);

static void fetchFlags(uint32_t rankId, void *syncArray, void *syncCounter) {
    static int32_t tmp[SHMEMI_SYNCBIT_SIZE / sizeof(int32_t) * 8];

    std::cout << "Rank " << rankId << ": ";

    CHECK_ACL(aclrtMemcpy(tmp, SHMEMI_SYNCBIT_SIZE, syncCounter, SHMEMI_SYNCBIT_SIZE, ACL_MEMCPY_DEVICE_TO_HOST));
    std::cout << "counter = " << *tmp << ", " << "flags = ";
    
    CHECK_ACL(aclrtMemcpy(tmp, SHMEMI_SYNCBIT_SIZE * 8, syncArray, SHMEMI_SYNCBIT_SIZE * 8, ACL_MEMCPY_DEVICE_TO_HOST));
    for (int i = 0; i < 8; i++) {
        std::cout << *(tmp + i * SHMEMI_SYNCBIT_SIZE / sizeof(int32_t)) << " ";
    }
    std::cout << std::endl;
}

static void TestBarrierWhiteBox(aclrtStream stream, uint32_t rankId, uint32_t rankSize)
{
    void *syncArray, *syncCounter;

    // get flag addr
    void *syncArrayHost, *syncCounterHost;
    void *syncArrayDevice, *syncCounterDevice;
    CHECK_ACL(aclrtMallocHost(&syncArrayHost, sizeof(void *)));
    CHECK_ACL(aclrtMallocHost(&syncCounterHost, sizeof(void *)));
    CHECK_ACL(aclrtMalloc(&syncArrayDevice, sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&syncCounterDevice, sizeof(void *), ACL_MEM_MALLOC_HUGE_FIRST));

    fetchAddrDo((uint8_t *)syncArrayDevice, (uint8_t *)syncCounterDevice);
    CHECK_ACL(aclrtSynchronizeStream(nullptr));
    CHECK_ACL(aclrtMemcpy(syncArrayHost, sizeof(void *), syncArrayDevice, sizeof(void *), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(syncCounterHost, sizeof(void *), syncCounterDevice, sizeof(void *), ACL_MEMCPY_DEVICE_TO_HOST));

    syncArray = (void *) *((uint64_t *) syncArrayHost);
    syncCounter = (void *) *((uint64_t *) syncCounterHost);

    CHECK_ACL(aclrtFreeHost(syncArrayHost));
    CHECK_ACL(aclrtFreeHost(syncCounterHost));
    CHECK_ACL(aclrtFree(syncArrayDevice));
    CHECK_ACL(aclrtFree(syncCounterDevice));

    // run barrier and check flags
    for (int i = 0; i < 100; i++) {
        barrierDo(nullptr);
        CHECK_ACL(aclrtSynchronizeStream(nullptr));
        fetchFlags(rankId, syncArray, syncCounter);

        // better insert an out-of-band barrier here, eg. MPI_Barrier
    }
}

static void TestBarrierBlackBox(aclrtStream stream, uint32_t rankId, uint32_t rankSize) {
    uint64_t *addrDev = (uint64_t *)ShmemMalloc(sizeof(uint64_t));
    uint64_t *addrHost;
    CHECK_ACL(aclrtMallocHost((void **)&addrHost, sizeof(uint64_t)));
    *addrHost = 0;
    CHECK_ACL(aclrtMemcpy(addrDev, sizeof(uint64_t), addrHost, sizeof(uint64_t), ACL_MEMCPY_HOST_TO_DEVICE));

    for (int i = 0; i < 100; i++) {
        increaseDo((uint8_t *)addrDev, rankId, rankSize);
        CHECK_ACL(aclrtSynchronizeStream(nullptr));
        CHECK_ACL(aclrtMemcpy(addrHost, sizeof(uint64_t), addrDev, sizeof(uint64_t), ACL_MEMCPY_DEVICE_TO_HOST));
        std::cout << "Rank " << rankId << ": " << *addrHost << std::endl;
    }
    
    CHECK_ACL(aclrtFreeHost(addrHost));
}

void ShmemBarrierAll();

static void TestShmemBarrier(int rankId, int nRanks, uint64_t localMemSize) {
    int status = SHMEM_SUCCESS;
    std::cout << "[TEST] input rank_size: " << nRanks << " rank_id:" << rankId << " input_ip: " << testGlobalIpport << std::endl;

    if (nRanks != (nRanks & (~(nRanks - 1)))) {
        std::cout << "[TEST] input rank_size: "<< nRanks << " is not the power of 2" << std::endl;
        status = ERROR_INVALID_VALUE;
    }
    EXPECT_EQ(status, SHMEM_SUCCESS);
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = rankId % testGNpuNum;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    ShmemInitAttrT* attributes;
    ShmemSetAttr(rankId, nRanks, localMemSize, testGlobalIpport, &attributes);
    status = ShmemInit();
    EXPECT_EQ(status, SHMEM_SUCCESS);

    TestBarrierWhiteBox(stream, rankId, nRanks);
    TestBarrierBlackBox(stream, rankId, nRanks);

    status = ShmemFinalize();
    EXPECT_EQ(status, SHMEM_SUCCESS);
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    if (::testing::Test::HasFailure()){
        exit(1);
    }
}



TEST(TestBarrierApi, TestShmemBarrier)
{   
    const int processCount = testGlobalRanks;
    uint64_t localMemSize = 1024UL * 1024UL * 1024;
    TestMutilTask(TestShmemBarrier, localMemSize, processCount);
}