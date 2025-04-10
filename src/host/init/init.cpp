#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace std;

#include "init.h"
#include "team.h"
#include "smem.h"
#include "smem_shm.h"

shmem_state_t *shmem_state;

void shmem_init(int rank, int size)
{
    shmem_state = (shmem_state_t *)calloc(1, sizeof(shmem_state_t *));

    // 静态堆初始化

    // team能力初始化
    shmem_team_init(rank, size);
}

void shmem_finalize()
{
    // team能力析构，后初始化先析构
    shmem_team_finalize();

    // 静态堆析构

    free(shmem_state);
}
