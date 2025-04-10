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

ShmemState_t *shmemState;

void ShmemInit(int rank, int size)
{
    shmemState = (ShmemState_t *)calloc(1, sizeof(ShmemState_t *));

    // 静态堆初始化

    // team能力初始化
    ShmemTeamInit(rank, size);
}

void ShmemFinalize()
{
    // team能力析构，后初始化先析构
    ShmemTeamFinalize();

    // 静态堆析构

    free(shmemState);
}
