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

class TeamAllGatherTest {
public:
    __aicore__ inline TeamAllGatherTest() {}
    __aicore__ inline void Init(GM_ADDR gva, shmem_team_t teamId)
    {
        gvaGm = (__gm__ int32_t *)gva;
        teamIdx= teamId;

        rank = smem_shm_get_global_rank();
        rankSize = smem_shm_get_global_rank_size();
        teamRank = shmem_team_my_pe(teamIdx);
        teamSize = shmem_team_n_pes(teamIdx);
    }
    __aicore__ inline void Process()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        CubeGuard();
        // All Gather
        for (int i = 0; i < teamSize - 1; i++) {
            int64_t dstRank = shmem_team_translate_pe(teamIdx, (teamRank + 1 + i) % teamSize, SHMEM_TEAM_WORLD);
            shmem_put_int32_mem_nbi(gvaGm + 16 * teamRank, gvaGm + 16 * teamRank, 16, dstRank);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        ShmemiBarrier(teamIdx);
    }
private:
    __gm__ int32_t *gvaGm;
    shmem_team_t teamIdx;

    int64_t rank;
    int64_t rankSize;
    int64_t teamRank;
    int64_t teamSize;
};

extern "C" __global__ __aicore__ void DeviceTeamAllGatherTest(GM_ADDR gva, int teamId)
{
    TeamAllGatherTest op;
    op.Init(gva, (shmem_team_t)teamId);
    op.Process();
}

void TeamAllGather(uint32_t blockDim, void* stream, uint8_t* gva, shmem_team_t teamId)
{
    DeviceTeamAllGatherTest<<<blockDim, nullptr, stream>>>(gva, (int)teamId);
}