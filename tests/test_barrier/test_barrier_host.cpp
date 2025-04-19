#include <iostream>
#include <string>
using namespace std;

#include "acl/acl.h"
#include "data_utils.h"

#include "shmem_api.h"

extern void fetchAddrDo(uint8_t* syncArray, uint8_t* syncCounter);

extern void barrierDo(uint8_t *stub);

extern void increaseDo(uint8_t *addr, int rankId, int rankSize);

static uint32_t gNpuNum = 8;
static uint64_t gNpuMallocSpace = 1024UL * 1024UL * 16;

static void fetchFlags(uint32_t rankId, void *syncArray, void *syncCounter) {
    static uint64_t tmp[SYNCBIT_SIZE / sizeof(uint64_t) * 8];

    std::cout << "Rank " << rankId << ": ";

    CHECK_ACL(aclrtMemcpy(tmp, SYNCBIT_SIZE, syncCounter, SYNCBIT_SIZE, ACL_MEMCPY_DEVICE_TO_HOST));
    std::cout << "counter = " << *tmp << ", " << "flags = ";
    
    CHECK_ACL(aclrtMemcpy(tmp, SYNCBIT_SIZE * 8, syncArray, SYNCBIT_SIZE * 8, ACL_MEMCPY_DEVICE_TO_HOST));
    for (int i = 0; i < 8; i++) {
        std::cout << *(tmp + i * SYNCBIT_SIZE / sizeof(uint64_t)) << " ";
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
    // TODO: use ShmemMalloc;
    uint64_t *addrDev = (uint64_t * ) (0x17ffc0000000 + SYNC_ARRAY_SIZE + SYNC_COUNTER_SIZE + rankId * (gNpuMallocSpace + SHMEM_EXTRA_SIZE));

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

int main(int argc, char* argv[]) 
{
    int rankSize = atoi(argv[1]);
    int rankId = atoi(argv[2]);
    std::string ipport = argv[3];
    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id:" << rankId << " input_ip: " << ipport << std::endl;

    if (rankSize != (rankSize & (~(rankSize - 1)))) {
        std::cout << "[TEST] input rank_size: "<< rankSize << " is not the power of 2" << std::endl;
        return -1;
    }

    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = rankId % gNpuNum;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;

    CHECK_ACL(aclrtCreateStream(&stream));
    ShmemInit(rankId, rankSize, gNpuMallocSpace);

    TestBarrierWhiteBox(stream, rankId, rankSize);
    TestBarrierBlackBox(stream, rankId, rankSize);

    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    ShmemFinalize();
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}