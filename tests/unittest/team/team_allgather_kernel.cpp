#include "kernel_operator.h"
#include "device/shmem_device_def.h"
#include "shmem_api.h"

SHMEM_DEVICE void CubeGuard() {
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

extern "C" __global__ __aicore__ void DeviceTeamAllGatherTest(GM_ADDR gva, int teamId)
{
    int64_t teamRank = shmem_team_my_pe(teamId);
    int64_t teamSize = shmem_team_n_pes(teamId);
    __gm__ int32_t* gvaGm = (__gm__ int32_t *)gva;
    AscendC::PipeBarrier<PIPE_ALL>();
    CubeGuard();
    // All Gather
    for (int i = 0; i < teamSize - 1; i++) {
        int64_t dstRank = shmem_team_translate_pe(teamId, (teamRank + 1 + i) % teamSize, SHMEM_TEAM_WORLD);
        shmem_put_int32_mem_nbi(gvaGm + 16 * teamRank, gvaGm + 16 * teamRank, 16, dstRank);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    ShmemiBarrier(teamId);
}

void TeamAllGather(uint32_t blockDim, void* stream, uint8_t* gva, shmem_team_t teamId)
{
    DeviceTeamAllGatherTest<<<blockDim, nullptr, stream>>>(gva, (int)teamId);
}