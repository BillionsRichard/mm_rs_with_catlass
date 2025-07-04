#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "shmemi_device_intf.h"

extern "C" int rtGetC2cCtrlAddr(uint64_t *config, uint32_t *len);

namespace shm {
static uint64_t ffts_config;

int32_t shmemi_sync_init() {
    uint32_t len;
    return rtGetC2cCtrlAddr(&ffts_config, &len);
}

} // namespace

uint64_t shmemx_get_ffts_config() {
    return shm::ffts_config;
}

void shmem_barrier(shmem_team_t tid) {
    // using default stream to do barrier
    shmemi_barrier_on_stream(tid, nullptr);
}

void shmem_barrier_all() {
    shmem_barrier(SHMEM_TEAM_WORLD);
}

void shmem_barrier_on_stream(shmem_team_t tid, aclrtStream stream)
{
    shmemi_barrier_on_stream(tid, stream);
}

void shmem_barrier_all_on_stream(aclrtStream stream)
{
    shmemi_barrier_on_stream(SHMEM_TEAM_WORLD, stream);
}