#include "kernel_operator.h"
#include "lowlevel/smem_shm_aicore_base_api.h"

#include "shmem_device_api.h"

class KernelStateTest {
public:
    __aicore__ inline KernelStateTest() {}
    __aicore__ inline void Init(GM_ADDR gva, ShmemTeam_t teamId)
    {
        gvaGm = (__gm__ int *)gva;
        teamIdx= teamId;

        rank = smem_shm_get_global_rank();
        rankSize = smem_shm_get_global_rank_size();
    }
    __aicore__ inline void Process()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        ShmemPInt(gvaGm, ShmemNpes(), rank);
        ShmemPInt(gvaGm + 1, ShmemMype(), rank);
        ShmemPInt(gvaGm + 2, ShmemTeamMype(teamIdx), rank);
        ShmemPInt(gvaGm + 3, ShmemTeamNpes(teamIdx), rank);
        ShmemPInt(gvaGm + 4, ShmemTeamTranslatePE(teamIdx, 1, SHMEM_TEAM_WORLD), rank);
    }
private:
    __gm__ int *gvaGm;
    ShmemTeam_t teamIdx;

    int64_t rank;
    int64_t rankSize;
};

extern "C" __global__ __aicore__ void DeviceStateTest(GM_ADDR gva, int teamId)
{
    KernelStateTest op;
    op.Init(gva, (ShmemTeam_t)teamId);
    op.Process();
}

void GetDeviceState(uint32_t blockDim, void* stream, uint8_t* gva, ShmemTeam_t teamId)
{
    DeviceStateTest<<<blockDim, nullptr, stream>>>(gva, (int)teamId);
}