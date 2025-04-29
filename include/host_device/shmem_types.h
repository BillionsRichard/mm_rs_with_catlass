#ifndef SHMEM_TYPES_H
#define SHMEM_TYPES_H

enum {
    SHMEM_TEAM_INVALID = -1,
    SHMEM_TEAM_WORLD = 0
};

enum DataOpEngineType {
    SHMEM_DATA_OP_MTE = 0x01,
};

typedef int ShmemTeam;

#endif /*SHMEM_TYPES_H*/