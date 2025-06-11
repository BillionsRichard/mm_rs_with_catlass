/*  
This file provides device-side collective synchronization implementations, ensuring that:
1. ALL VEC CORES of all ranks of a team reach a sychonization point before doing subsequent operations.
2. All operations of ALL VEC CORES of all ranks of the team before the synchronization point are visible to ALL VEC CORES of all ranks of the team after the synchronization point.

*/

#ifndef SHEMEI_BARRIER_H
#define SHEMEI_BARRIER_H

#include "../shmemi_device_common.h"
#include "shmemi_device_quiet.h"
#include "shmemi_device_p2p.h"

#include "kernel_operator.h"

/* Level 1: barrier between vec cores (within a device) */
template<bool isAIVOnly = true>
SHMEM_DEVICE void shmemi_barrier_core() {
    AscendC::SyncAll<isAIVOnly>();
}

SHMEM_DEVICE 
__gm__ shmemi_sync_bit *shmemi_get_team_sync_array(shmem_team_t team_idx) {
    uint64_t addr = (uint64_t) shmemi_get_state()->sync_pool;
    addr += team_idx * SYNC_ARRAY_SIZE;
    return (__gm__ shmemi_sync_bit *) addr;
}

SHMEM_DEVICE 
__gm__ shmemi_sync_bit *shmemi_get_team_sync_counter(shmem_team_t team_idx) {
    uint64_t addr = (uint64_t) shmemi_get_state()->sync_counter;
    addr += team_idx * SYNC_COUNTER_SIZE;
    return (__gm__ shmemi_sync_bit *) addr;
}

/* Level 2: barrier between devices (within a host)

Dissemination Barrier

1. Algorithm process

The algorithm process could be separated into multiple rounds. 
In each round, every participating rank signals its succeeding rank and waits its preceding rank's signal.  
The distance of a rank and its successor increases exponentially with the round.

An 8-rank example is shown below:

           round 1            round 2            round 3
  rank 0  --------→  rank 1  --------→  rank 3  --------→  rank 7
  rank 1  --------→  rank 2  --------→  rank 4  --------→  rank 0
  rank 2  --------→  rank 3  --------→  rank 5  --------→  rank 1
  rank 3  --------→  rank 4  --------→  rank 6  --------→  rank 2
  rank 4  --------→  rank 5  --------→  rank 7  --------→  rank 3
  rank 5  --------→  rank 6  --------→  rank 0  --------→  rank 4
  rank 6  --------→  rank 7  --------→  rank 1  --------→  rank 5
  rank 7  --------→  rank 0  --------→  rank 2  --------→  rank 6

Refer to https://www.inf.ed.ac.uk/teaching/courses/ppls/BarrierPaper.pdf for more details.
  
2. Implementation details

Current implementation maintains an array of MAX_RANK_SIZE for each rank, with element of pos i indicating whether the rank has received signal of rank i.
In each round, every rank writes remote array and check local array to decide whether this round has finished. Once all rounds finished, barrier ends. 

Theoretically, each element is writen by only 1 rank and read by self, involving only p2p synchronization.
However, separate elements may exist on the same cacheline, so that concurrent write acctually happens and may cause wrong result.

For example:
a. rank n is waiting for rank n-1's signal (in round 1).
             ↑   n
--------------------------------------------
      ...  | 0 | 0 | ...
--------------------------------------------

b. rank n-1 reads rank n's array, and write the element at position n-1 (in round 1).
             ↓   n
--------------------------------------------
      ...  | 1 | 0 | ...
--------------------------------------------

c. rank n-2 reads staled rank n's array (no cache consistency ensurance), and write the element at position n-2 (in round 2).
         ↓       n
--------------------------------------------
   ... | 1 | 0 | 0 | ...
--------------------------------------------

d. rank n-2 overwrites rank n-1，so rank n may miss rank n-1's signal and wait forever.
             ↑   n
--------------------------------------------
   ... | 1 | 0 | 0 | ...
--------------------------------------------

To avoid this issue, separate elements must exist on different cachelines. See shmemi_sync_bit for detailed definition.

Additionly, instead of simply write a flag, each rank writes a 64-bit number into the array, indicating how many times this team has performed barrier. 

The temporal and spatial complexity of this implementation are O(logN) and O(N), respectively. 

3. Futher development
  a. Hierarchical synchronization. 
    Sync within the host first, then sync between host. May achieve better performance by utilizing low-latency in-host network better.

  b. Group dissemination.
    Group the ranks so that each rank could issue multiple signals and waits concurrently, instead of 1 signal and 1 wait as above.

  c. Optimize spatial complexity to O(logN).
*/

SHMEM_DEVICE void shmemi_barrier_npu(shmemi_team_t *team) {
    if (AscendC::GetBlockIdx() != 0)
        return;

    int my_pe = shmemi_get_state()->team_pools[SHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    auto sync_array = shmemi_get_team_sync_array(team->team_idx);
    auto sync_counter = shmemi_get_team_sync_counter(team->team_idx);

    int shift = 1;
    int my_pe_in_team = (my_pe - start) / stride;
    int32_t count = shmemi_load<int32_t>((__gm__ uint8_t *)sync_counter);

    while (shift < size) {
        int pre_pe_in_team = (my_pe_in_team - shift + size) % size;
        int next_pe_in_team = (my_pe_in_team + shift) % size;

        int pre_pe = start + pre_pe_in_team * stride;
        int next_pe = start + next_pe_in_team * stride;

        // signal next pe
        shmemi_signal<int32_t>((__gm__ uint8_t *)(sync_array + my_pe), next_pe, count);

        // wait pre pe
        shmemi_wait<int32_t>((__gm__ uint8_t *)(sync_array + pre_pe), count);
        
        shift *= 2;
    } 

    shmemi_store<int32_t>((__gm__ uint8_t *)sync_counter, count + 1);
}

// v2 -- use multi vector cores
SHMEM_DEVICE void shmemi_barrier_npu_v2(shmemi_team_t *team) {
    int vec_id = AscendC::GetBlockIdx();
    int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();

    int my_pe = shmemi_get_state()->team_pools[SHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    auto sync_array = shmemi_get_team_sync_array(team->team_idx);
    auto sync_counter = shmemi_get_team_sync_counter(team->team_idx);

    int shift = 1;
    int k = size < SHMEM_BARRIER_TG_DISSEM_KVAL ? size : SHMEM_BARRIER_TG_DISSEM_KVAL;
    int my_pe_in_team = (my_pe - start) / stride;
    int32_t count = shmemi_load<int32_t>((__gm__ uint8_t *)sync_counter);

    while (shift < size) {
        for (int i = vec_id + 1; i < k; i += vec_size) {
            int next_pe_in_team = (my_pe_in_team + i * shift) % size;
            int next_pe = start + next_pe_in_team * stride;

            // signal next pe
            shmemi_signal<int32_t>((__gm__ uint8_t *)(sync_array + my_pe), next_pe, count);
        }

        for (int i = vec_id + 1; i < k; i += vec_size) {
            int pre_pe_in_team = (my_pe_in_team - i * shift + size) % size;
            int pre_pe = start + pre_pe_in_team * stride;

            // wait pre pe
            shmemi_wait<int32_t>((__gm__ uint8_t *)(sync_array + pre_pe), count);
        }
        
        shift *= k;
    } 

    shmemi_store<int32_t>((__gm__ uint8_t *)sync_counter, count + 1);
}

// v3 -- use multi vector cores + mte
SHMEM_DEVICE void shmemi_barrier_npu_v3(shmemi_team_t *team) {
    int vec_id = AscendC::GetBlockIdx();
    int vec_size = AscendC::GetBlockNum() * AscendC::GetTaskRation();

    __gm__ shmemi_device_host_state_t *device_state = shmemi_get_state();
    /* CopyUB Config Set */
    uint64_t copy_ub = 0;
    uint32_t copy_ub_size = 32;
    AscendC::TEventID copy_event_id = (AscendC::TEventID)0;

    int my_pe = shmemi_get_state()->team_pools[SHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    auto sync_array = shmemi_get_team_sync_array(team->team_idx);
    auto sync_counter = shmemi_get_team_sync_counter(team->team_idx);

    int shift = 1;
    int k = size < SHMEM_BARRIER_TG_DISSEM_KVAL ? size : SHMEM_BARRIER_TG_DISSEM_KVAL;
    int my_pe_in_team = (my_pe - start) / stride;
    int32_t count = shmemi_load<int32_t>((__gm__ uint8_t *)sync_counter);

    while (shift < size) {
        for (int i = vec_id + 1; i < k; i += vec_size) {
            int next_pe_in_team = (my_pe_in_team + i * shift) % size;
            int next_pe = start + next_pe_in_team * stride;

            // signal next pe
            shmemi_signal_v2((__gm__ uint32_t *)(sync_array + my_pe), next_pe, count, (__ubuf__ uint32_t *)copy_ub, copy_event_id);
        }

        for (int i = vec_id + 1; i < k; i += vec_size) {
            int pre_pe_in_team = (my_pe_in_team - i * shift + size) % size;
            int pre_pe = start + pre_pe_in_team * stride;

            // wait pre pe
            shmemi_wait_v2((__gm__ uint32_t *)(sync_array + pre_pe), count, (__ubuf__ uint32_t *)copy_ub, copy_event_id);
        }
        
        shift *= k;
    }

    shmemi_store<int32_t>((__gm__ uint8_t *)sync_counter, count + 1);
}

/* Level 3: barrier between hosts, TO BE IMPLEMENTED.*/ 
SHMEM_DEVICE void shmemi_barrier_sys() {}

template<bool isAIVOnly = true>
SHMEM_DEVICE void shmemi_barrier(shmem_team_t tid) {
    shmemi_team_t *team = shmemi_get_state()->team_pools[tid];

    int mype = shmemi_get_state()->team_pools[SHMEM_TEAM_WORLD]->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;

    if ((mype - start) % stride != 0) {
        // not in this team
        return;
    }

    shmemi_barrier_core<isAIVOnly>();

    if ASCEND_IS_AIV {
        // shmemi_barrier_npu(team);
        shmemi_barrier_npu_v2(team);
        // shmemi_barrier_npu_v3(team);
    }

    shmemi_barrier_core<isAIVOnly>();
}

#endif