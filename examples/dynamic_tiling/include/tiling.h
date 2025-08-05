/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef TILING_H
#define TILING_H

#include "info.h"
#include "launch_map.h"
#include <sstream>
#include <vector>

// tiling 搜索空间
std::vector<uint32_t> vCommInterval = {4, 6, 8, 12, 14};
std::vector<uint32_t> vCommTileM = {4, 8, 16, 32, 64};
std::vector<std::pair<uint32_t, uint32_t>> vCommSplitNpuDataPair = {{1, 16}, {1, 20}};
std::vector<std::vector<uint32_t>> allParams = {vCommInterval, vCommTileM};

int32_t CeilDev(int32_t num, int32_t div)
{
    if (div == 0) {
        return 0;
    }
    return (num + div - 1) / div;
}

bool CheckCommIntervalReduceScatter(const CocTilingParams &tiling, int rankSize)
{
    constexpr int32_t blockNum= BLOCK_NUM;   
    int64_t product = static_cast<int64_t>(blockNum) * tiling.commInterval;

    if (product % rankSize != 0) {
        return false;
    }
    return true;
}
 
bool CheckCommIntervalAllReduce(const CocTilingParams &tiling, int rankSize)
{
    auto blockCount = MAX_BLOCK_COUNT;
    int32_t maxPeerMemPerRank = (LCAL_BUFF_BYTES - FLAG_BUFF_BYTES) / INPUT_DTYPE / rankSize / blockCount;
    if (tiling.commInterval * tiling.m0 * tiling.n0 * BLOCK_NUM >= maxPeerMemPerRank) {
        return false;
    }
    return true;
}

bool CheckCommIntervalAllGather(const CocTilingParams &tiling, int rankSize)
{
    auto blockCount = MAX_BLOCK_COUNT;
    uint32_t kLoop = CeilDev(tiling.k, tiling.k0);
    int32_t maxPeerMemPerRank = (LCAL_BUFF_BYTES - FLAG_BUFF_BYTES) / INPUT_DTYPE / rankSize / blockCount;
    if (tiling.commInterval * tiling.m0 * tiling.k0 * kLoop >= maxPeerMemPerRank) {
        return false;
    }
    return true;
}

void GetParamFromSearchSpace(std::vector<uint32_t>& curParams, 
                             std::vector<std::vector<uint32_t>> &results, 
                             int pos) {
    if (pos == allParams.size()) {
        for (int i = 0; i < vCommSplitNpuDataPair.size(); i++) {
            std::vector<uint32_t> tmpParams(curParams.begin(), curParams.end());
            tmpParams.push_back(vCommSplitNpuDataPair[i].first);
            tmpParams.push_back(vCommSplitNpuDataPair[i].second);
            results.push_back(tmpParams);
        }
    }
    else {
        for (int i = 0; i < allParams[pos].size(); i++) {
            curParams[pos] = allParams[pos][i];
            GetParamFromSearchSpace(curParams, results, pos + 1);
        }
    }
}

void GetTilings(std::vector<CocTilingParams> &tilings, CocTilingParams &t,
    CocCommType commType, int rankSize) {
    std::vector<uint32_t> curParams(allParams.size(), 0);
    std::vector<std::vector<uint32_t>> allTilings;
    GetParamFromSearchSpace(curParams, allTilings, 0);
    for (const auto &tiling : allTilings) {
        uint32_t idx = 0;
        t.commInterval = tiling[idx++];
        t.commTileM = tiling[idx++];
        t.commBlockM = t.commTileM;
        t.commNpuSplit = tiling[idx++];
        t.commDataSplit = tiling[idx++];

        if (commType == ALLGATHER_MATMUL && !CheckCommIntervalAllGather(t, rankSize)) 
            continue;
        if (commType == MATMUL_REDUCE_SCATTER && !CheckCommIntervalReduceScatter(t, rankSize))
            continue;
        if (commType == MATMUL_ALLREDUCE && !CheckCommIntervalAllReduce(t, rankSize))
            continue;

        tilings.push_back(t);
    }
}

bool CreateTilingFile(const std::string filename)
{
    std::ofstream outFile(filename, std::ios::out);
    if (!outFile.is_open()) {
        std::cerr << "Open file failed." << std::endl;
        return false;
    }
    outFile << "Op,M,K,N,Transpose A,Transpose B,commInterval,commTileM,commBlockM,commNpuSplit,commDataSplit,Time(us)\n";
    outFile.close();
    return true;
}

bool WriteTilingInfos(std::string opName, std::vector<CocTilingParams> &cocTilings, const std::string filename, 
                      int transA = 0, int transB = 1) {
    std::ofstream ouputFile(filename, std::ios::out | std::ios::app);
    if (!ouputFile) {
        ERROR_LOG("Open file failed. path = %s, error = %s", filename.c_str(), strerror(errno));
        return false;
    }
        
    for (CocTilingParams cocTiling : cocTilings) {
        ouputFile << opName 
                  << "," << cocTiling.m
                  << "," << cocTiling.k
                  << "," << cocTiling.n
                  << "," << transA
                  << "," << transB
                  << "," << cocTiling.commInterval
                  << "," << cocTiling.commTileM
                  << "," << cocTiling.commBlockM
                  << "," << cocTiling.commNpuSplit
                  << "," << cocTiling.commDataSplit
                  << "," << "\n";
    }

    ouputFile.close();
    return true;
}

#endif // TILING_H