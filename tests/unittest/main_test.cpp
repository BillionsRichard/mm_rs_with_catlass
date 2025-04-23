#include <gtest/gtest.h>
#include <iostream>

#define DEFAULT_RANKS 8
#define DEFAULT_NPU_NUM 8
#define DEFAULT_IPPORT "tcp://127.0.0.1:8666"

int testGlobalRanks = DEFAULT_RANKS;
int testGNpuNum = DEFAULT_NPU_NUM;
const char* testGlobalIpport = DEFAULT_IPPORT;

void TestMutilTask(std::function<void(int, int, uint64_t)> func, uint64_t localMemSize, int processCount){
    pid_t pids[processCount];
    int status[processCount];
    for (int i = 0; i < processCount; ++i) {
        pids[i] = fork();
        if (pids[i] < 0) {
            std::cout << "fork failed ! " << pids[i] << std::endl;
        } else if (pids[i] == 0) {
            func(i, processCount, localMemSize);
            exit(0);
        }
    }
    for (int i = 0; i < processCount; ++i) {
        waitpid(pids[i], &status[i], 0);
        if (WIFEXITED(status[i]) && WEXITSTATUS(status[i]) != 0) {
            FAIL();
        }
    }
}

int main(int argc, char** argv) {
    if (argc > 1) {
        testGlobalRanks = std::atoi(argv[1]);
        testGlobalIpport = argv[2];
        testGNpuNum = std::atoi(argv[3]);
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}