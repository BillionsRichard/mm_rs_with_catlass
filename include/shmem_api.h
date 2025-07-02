#ifndef SHMEM_API_H
#define SHMEM_API_H

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#include "device/shmem_device_def.h"
#include "device/shmem_device_rma.h"
#include "device/shmem_device_sync.h"
#include "device/shmem_device_team.h"
#endif

#include "host/shmem_host_def.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_rma.h"
#include "host/shmem_host_sync.h"
#include "host/shmem_host_team.h"

#endif // SHMEM_API_H