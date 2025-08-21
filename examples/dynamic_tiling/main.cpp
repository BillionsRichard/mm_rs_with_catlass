/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>
#include <runtime/rt_ffts.h>
#include <fstream>
#include <string>

#include <iostream>
#include <vector>
#include <cstring>
#include <sstream>
#include <map>
#include <ctime>
#include <iomanip>

#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_rma.h"
#include "host/shmem_host_team.h"

#include "utils.h"
#include "info.h"
#include "tiling.h"
#include "launch_map.h"
#include "coc_tiling_lut.h"

using half = __fp16;

const std::map<CocCommType, std::string> commTypeMap = {
    { MATMUL_ALLREDUCE, "MatmulAllReduce" },
    { ALLGATHER_MATMUL, "AllGatherMatmul" },
    { MATMUL_REDUCE_SCATTER, "MatmulReduceScatter" }
};

struct Options {
    CocCommType commType;
    CocDataType dataType;
    int rankSize;
    int rankId;
    std::string ipPort{};
    uint32_t m{0};
    uint32_t n{0};
    uint32_t k{0};
    std::vector<int> deviceIdList{};
    uint32_t test_start_line{0};
    uint32_t test_collect_rows{0};
    std::string parentPath{};
    std::string csv_file{};
    std::string data_file{};

    int Parse(int argc, char **argv)
    {
        enum ArgsIndex {
            COMM_TYPE_INDEX = 1,
            DATA_TYPE_INDEX,
            RANK_SIZE_INDEX,
            RANK_ID_INDEX,
            IP_PORT_INDEX,
            M_INDEX,
            N_INDEX,
            K_INDEX,
            START_LINE_INDEX,
            COLLECT_ROWS_INDEX,
            PARENT_PATH_INDEX,
            CSV_FILE_INDEX,
            DEVICE_LIST_INDEX,
            DATA_FILE_INDEX,
            INDEX_MAX
        };

        if (argc > INDEX_MAX) {
            return -1;
        }

        commType = static_cast<CocCommType>(std::atoi(argv[COMM_TYPE_INDEX]));
        dataType = static_cast<CocDataType>(std::atoi(argv[DATA_TYPE_INDEX]));
        rankSize = std::atoi(argv[RANK_SIZE_INDEX]);
        rankId = std::atoi(argv[RANK_ID_INDEX]);
        ipPort = argv[IP_PORT_INDEX];
        m = std::atoi(argv[M_INDEX]);
        n = std::atoi(argv[N_INDEX]);
        k = std::atoi(argv[K_INDEX]);
        test_start_line = std::atoi(argv[START_LINE_INDEX]);
        test_collect_rows = std::atoi(argv[COLLECT_ROWS_INDEX]);
        parentPath = argv[PARENT_PATH_INDEX];
        csv_file = argv[CSV_FILE_INDEX];
        if (argc > DEVICE_LIST_INDEX) {
            char *idListStr = argv[DEVICE_LIST_INDEX];
            for (char *idToken = std::strtok(idListStr, ","); idToken; idToken = std::strtok(nullptr, ",")) {
                deviceIdList.push_back(std::atoi(idToken));
            }
        } else {
            for (size_t i = 0; i < rankSize; ++i) {
                deviceIdList.push_back(i);
            }
        }
        if (argc > DATA_FILE_INDEX) {
            data_file = argv[DATA_FILE_INDEX];
        }
        return 0;
    }
};

std::vector<std::vector<uint32_t>> InitTestShapes(const Options &options)
{
    uint32_t startLine = options.test_start_line;
    uint32_t collectRows = options.test_collect_rows;
    std::string shapeFileName = options.csv_file;
    std::vector<std::string> headers = {};
    std::vector<std::vector<uint32_t>> shapes = {};
    std::ifstream file(shapeFileName);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << shapeFileName << std::endl;
        return shapes;
    }

    std::string line;

    if (getline(file, line)) {
        std::stringstream ss(line);
        std::string header;
        while (getline(ss, header, ',')) {
            headers.push_back(header);
        }
    } else {
        std::cerr << "The file is empty or the header line fails to be read." << std::endl;
        return shapes;
    }

    int rowIndex = 0;
    int added = 0;

    while (getline(file, line)) {
        if (line.empty()) continue;
        if (rowIndex < startLine) {
            ++rowIndex;
            continue;
        }
        if (added >= collectRows) {
            break;
        }

        std::stringstream ss(line);
        std::vector<uint32_t> shape;
        std::string cell;
        while (getline(ss, cell, ',')) {
            shape.push_back(std::stoi(cell));
        }

        if (shape.size() != headers.size()) {
            std::cerr << "The number of data columns in row " << rowIndex << " does not match the number of header columns: " << line << std::endl;
        } else {
            shapes.push_back(shape);
            ++added;
        }
        ++rowIndex;
    }
    file.close();

    return shapes;
}

std::string GetCurrentTime()
{
    std::time_t now = std::time(nullptr);
    std::tm tm = *std::localtime(&now);

    std::stringstream ss;
    ss << std::put_time(&tm, "%Y%m%d%H%M%S");
    return ss.str();
}

int main(int argc, char **argv)
{
    int status = SHMEM_SUCCESS;
    Options options;
    options.Parse(argc, argv);
    CocCommType commType = options.commType;
    CocDataType dataType = options.dataType;
    int rankSize = options.rankSize;
    int rankId = options.rankId;
    std::string ipPort = options.ipPort;
    int32_t deviceId = options.deviceIdList[rankId];
    std::string data_file = options.data_file;
    const std::vector<std::vector<uint32_t>> shapes = InitTestShapes(options);

    std::cout << "[TEST] input rank_size: " << rankSize << " rank_id: " << rankId << " input_ip: " << ipPort << "\n";

    aclrtStream stream = nullptr;
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtCreateStream(&stream));
    status = shmem_set_conf_store_tls(false, nullptr, 0);
    shmem_init_attr_t *attributes;
    status = shmem_set_attr(rankId, rankSize, SHMEM_MALLOC_MAX_SIZE, ipPort.c_str(), &attributes);
    status = shmem_init_attr(attributes);
    status = shmem_init_status();

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    RT_CHECK(rtGetC2cCtrlAddr(&fftsAddr, &fftsLen));

    std::string currentTime = GetCurrentTime();
    std::string opName = commTypeMap.at(commType);
    std::string currentDir = options.parentPath;
    std::string tilingFileName = currentDir + "/output/tiling/tilingData_" + currentTime + ".csv";
    if (rankId == 0) {
        CreateTilingFile(tilingFileName);
    }

    for (size_t i = 0; i < shapes.size(); i++) {
        uint32_t m = shapes[i][0];
        uint32_t k = shapes[i][1];
        uint32_t n = shapes[i][2];
        uint32_t transA = shapes[i][3];
        uint32_t transB = shapes[i][4];

        CocTilingParams cocTiling;
        cocTiling.m = m;
        cocTiling.n = n;
        cocTiling.k = k;
        COCMatMulInfo info{ int64_t(m), int64_t(k), int64_t(n) };
        cocTiling.m0 = M0;
        cocTiling.n0 = N0;
        cocTiling.k0 = K0;
        cocTiling.commTileM = 64;
        cocTiling.commInterval = 3;
        cocTiling.commNpuSplit = 1;
        cocTiling.commDataSplit = 20;
        cocTiling.commBlockM = 64;
        cocTiling.rankSize = rankSize;

        size_t aSize = static_cast<size_t>(m) * k * sizeof(half);
        size_t bSize = static_cast<size_t>(k) * n * sizeof(half);
        size_t cSize = static_cast<size_t>(m) * n * sizeof(half);
        size_t cSizePerRank;
        if (commType == MATMUL_REDUCE_SCATTER) {
            cSizePerRank = cSize / rankSize;
        } else if (commType == ALLGATHER_MATMUL) {
            cSizePerRank = cSize * rankSize;
        } else {
            cSizePerRank = cSize;
        }

        uint8_t *aDevice;
        ACL_CHECK(aclrtMalloc((void **)(&aDevice), aSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *aHost;
        if (data_file != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&aHost), aSize));
            ReadFile(data_file + "/rank_" + std::to_string(rankId) + "_a.bin", aHost, aSize);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<half> matrixA(m * k, 1);
            ACL_CHECK(aclrtMemcpy(aDevice, aSize, matrixA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *bDevice;
        ACL_CHECK(aclrtMalloc((void **)(&bDevice), bSize, ACL_MEM_MALLOC_HUGE_FIRST));
        uint8_t *bHost;
        if (data_file != "") {
            ACL_CHECK(aclrtMallocHost((void **)(&bHost), bSize));
            ReadFile(data_file + "/rank_" + std::to_string(rankId) + "_b.bin", bHost, bSize);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        } else {
            std::vector<half> matrixB(k * n, 1);
            ACL_CHECK(aclrtMemcpy(bDevice, bSize, matrixB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        uint8_t *cDevice;
        ACL_CHECK(aclrtMalloc((void **)(&cDevice), cSizePerRank, ACL_MEM_MALLOC_HUGE_FIRST));
        if (commType == MATMUL_REDUCE_SCATTER) {
            std::vector<uint8_t> matrixCInit(cSizePerRank, 0);
            ACL_CHECK(aclrtMemcpy(cDevice, cSizePerRank, matrixCInit.data(), cSizePerRank, ACL_MEMCPY_HOST_TO_DEVICE));
        }

        void *symmPtr = shmem_malloc(SHMEM_BUFF_BYTES);
        uint8_t *gmSymmetric = (uint8_t *)symmPtr;
        
        uint32_t warmUpTimes = std::getenv("WARM_UP_TIMES") == nullptr ? WARM_UP_TIMES : std::stoull(std::getenv("WARM_UP_TIMES"));
        uint32_t perfTestCycleTimes = std::getenv("PERF_TEST_CYCLE_TIMES") == nullptr ? PERF_TEST_CYCLE_TIMES : std::stoull(std::getenv("PERF_TEST_CYCLE_TIMES"));
        uint32_t searchparams = (std::getenv("SEARCH_PARAMS") == nullptr) ? 1U : std::stoul(std::getenv("SEARCH_PARAMS"));

        std::vector<CocTilingParams> cocTilings;
        if (warmUpTimes == 0) {
            cocTilings.push_back(cocTiling);
        } else {
            if (searchparams == 1) {
                // 搜索 tiling
                GetTilings(cocTilings, cocTiling, commType, rankSize);
            } else {
                bool ok = ApplyLookupTable(info, commType, rankSize, cocTiling);
                if (!ok) {
                    std::cerr << "[LUT] no table for (" << opName << "," << rankSize << "), using defaults\n";
                }
                cocTilings.push_back(cocTiling);
            }
        }

        ACL_CHECK(aclrtSynchronizeStream(stream));

        auto kernelFunc = KernelDispatcher::GetKernelFunc(commType, dataType);

        for (size_t i = 0; i < warmUpTimes; i++) {
            kernelFunc(stream, fftsAddr, aDevice, bDevice, cDevice, nullptr, nullptr, gmSymmetric, cocTilings[0], transA, transB);
        }

        for (CocTilingParams tiling : cocTilings) {
            for (size_t i = 0; i < perfTestCycleTimes; i++) {
                kernelFunc(stream, fftsAddr, aDevice, bDevice, cDevice, nullptr, nullptr, gmSymmetric, tiling, transA, transB);
            }
        }

        ACL_CHECK(aclrtSynchronizeStream(stream));

        uint8_t *cHost;
        ACL_CHECK(aclrtMallocHost((void **)(&cHost), cSizePerRank));
        ACL_CHECK(aclrtMemcpy(cHost, cSizePerRank, cDevice, cSizePerRank, ACL_MEMCPY_DEVICE_TO_HOST));

        if (data_file != "") {
            if (commType == MATMUL_ALLREDUCE) {
                if (rankId == 0) {
                    WriteFile(data_file + "/output.bin", cHost, cSizePerRank);
                }
            } else if (commType == ALLGATHER_MATMUL) {
                if (rankId == 0) {
                    WriteFile(data_file + "/output.bin", cHost, cSizePerRank);
                }
            } else if (commType == MATMUL_REDUCE_SCATTER) {
                WriteFile(data_file + "/output.bin", cHost, cSizePerRank, rankId * cSizePerRank);
            }
        }

        if (rankId == 0) {
            WriteTilingInfos(opName, cocTilings, tilingFileName, transA, transB);
            std::printf("M: %d, K: %d, N: %d aclrtSynchronizeStream success!\n", cocTiling.m, cocTiling.k, cocTiling.n);
        }

        shmem_free(symmPtr);

        if (data_file != "") {
            ACL_CHECK(aclrtFreeHost(aHost));
            ACL_CHECK(aclrtFreeHost(bHost));
        }
        ACL_CHECK(aclrtFreeHost(cHost));
        ACL_CHECK(aclrtFree(aDevice));
        ACL_CHECK(aclrtFree(bDevice));
        ACL_CHECK(aclrtFree(cDevice));
    }
    std::cout << "[TEST] begin to exit...... rankId: " << rankId << std::endl;
    status = shmem_finalize();
    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(deviceId));
    ACL_CHECK(aclFinalize());

    return 0;
}