#include "kernel_operator.h"
#include "shmem_api.h"

class KernelStateTest {
public:
    __aicore__ inline KernelStateTest() {}
    __aicore__ inline void Init(GM_ADDR gva, shmem_team_t teamId)
    {
        gvaGm = (__gm__ int *)gva;
        teamIdx= teamId;

        rank = smem_shm_get_global_rank();
        rankSize = smem_shm_get_global_rank_size();
    }
    __aicore__ inline void Process()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        shmem_int32_p(gvaGm, shmem_n_pes(), rank);
        shmem_int32_p(gvaGm + 1, shmem_my_pe(), rank);
        shmem_int32_p(gvaGm + 2, shmem_team_my_pe(teamIdx), rank);
        shmem_int32_p(gvaGm + 3, shmem_team_n_pes(teamIdx), rank);
        shmem_int32_p(gvaGm + 4, shmem_team_translate_pe(teamIdx, 1, SHMEM_TEAM_WORLD), rank);
    }
private:
    __gm__ int *gvaGm;
    shmem_team_t teamIdx;

    int64_t rank;
    int64_t rankSize;
};

extern "C" __global__ __aicore__ void DeviceStateTest(GM_ADDR gva, int teamId)
{
    KernelStateTest op;
    op.Init(gva, (shmem_team_t)teamId);
    op.Process();
}

void GetDeviceState(uint32_t blockDim, void* stream, uint8_t* gva, shmem_team_t teamId)
{
    DeviceStateTest<<<blockDim, nullptr, stream>>>(gva, (int)teamId);
}