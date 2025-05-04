#ifndef SHMEM_TYPES_H
#define SHMEM_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

enum {
    SHMEM_TEAM_INVALID = -1,
    SHMEM_TEAM_WORLD = 0
};

enum data_op_engine_type_t {
    SHMEM_DATA_OP_MTE = 0x01,
};

typedef int shmem_team_t;

#ifdef __cplusplus
}
#endif

#endif /*SHMEM_TYPES_H*/