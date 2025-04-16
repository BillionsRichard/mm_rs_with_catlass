#ifndef SHMEM_MEM_H
#define SHMEM_MEM_H

#include <climits>
#include <cstdlib>
#include <cstdbool>
#include <acl/acl.h>

#include "shmem_internal.h"

void* ShmemPtr(void *ptr, int pe);

#endif