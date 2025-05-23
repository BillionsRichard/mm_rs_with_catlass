#include "kernel_operator.h"
#include "shmem_api.h"

class KernelStateTest {
public:
    __aicore__ inline KernelStateTest() {}
    __aicore__ inline void Init(GM_ADDR gva, shmem_team_t team_id)
    {
        gva_gm = (__gm__ int *)gva;
        team_idx= team_id;

        rank = smem_shm_get_global_rank();
        rank_size = smem_shm_get_global_rank_size();
    }
    __aicore__ inline void Process()
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        shmem_int32_p(gva_gm, shmem_n_pes(), rank);
        shmem_int32_p(gva_gm + 1, shmem_my_pe(), rank);
        shmem_int32_p(gva_gm + 2, shmem_team_my_pe(team_idx), rank);
        shmem_int32_p(gva_gm + 3, shmem_team_n_pes(team_idx), rank);
        shmem_int32_p(gva_gm + 4, shmem_team_translate_pe(team_idx, 1, SHMEM_TEAM_WORLD), rank);
    }
private:
    __gm__ int *gva_gm;
    shmem_team_t team_idx;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void DeviceStateTest(GM_ADDR gva, int team_id)
{
    KernelStateTest op;
    op.Init(gva, (shmem_team_t)team_id);
    op.Process();
}

void GetDeviceState(uint32_t block_dim, void* stream, uint8_t* gva, shmem_team_t team_id)
{
    DeviceStateTest<<<block_dim, nullptr, stream>>>(gva, (int)team_id);
}