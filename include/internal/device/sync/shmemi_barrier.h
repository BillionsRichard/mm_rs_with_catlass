/*  
This file provides device-side collective synchronization implementations, ensuring that:
1. all ranks of a team reach a sychonization point before doing subsequent operations.
2. all REMOTE operations of all ranks of the team before the synchronization point are visible to all ranks of the team after the synchronization point.

Be careful that synchronization between blocks is not guaranteed.
*/

#ifndef SHEMEI_BARRIER_H
#define SHEMEI_BARRIER_H

#include "internal/device/arch.h"
#include "internal/device/shmemi_device_common.h"
#include "internal/device/team/shmemi_team.h"
#include "shmemi_quiet.h"
#include "shmemi_p2p.h"

#include "kernel_operator.h"

/* Level 1: barrier between cores (within a device) */
SHMEM_AICORE_INLINE void ShmemiBarrierL1() {
    AscendC::SyncAll<true>();
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

To avoid this issue, separate elements must exist on different cachelines. See SyncBit for detailed definition.

Additionly, instead of simply write a flag, each rank writes a 64-bit number into the array, indicating how many times this team has performed barrier. 

The temporal and spatial complexity of this implementation are O(logN) and O(N), respectively. 

3. Futher development
  a. Hierarchical synchronization. 
    Sync within the host first, then sync between host. May achieve better performance by utilizing low-latency in-host network better.

  b. Group dissemination.
    Group the ranks so that each rank could issue multiple signals and waits concurrently, instead of 1 signal and 1 wait as above.

  c. Optimize spatial complexity to O(logN).
*/

SHMEM_AICORE_INLINE void ShmemiBarrierL2(ShmemTeam *team) {
    int myPe = team->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;
    auto syncArrayL2 = ShmemiGetTeamSyncArrayL2(team);
    auto syncCounterL2 = ShmemiGetTeamSyncCounterL2(team);

    int shift = 1;
    int myPeInTeam = (myPe - start) / stride;
    int32_t count = load<int32_t>((__gm__ uint8_t *)syncCounterL2);

    while (shift < size) {
        int prePeInTeam = (myPeInTeam - shift + size) % size;
        int nextPeInTeam = (myPeInTeam + shift) % size;

        int prePe = start + prePeInTeam * stride;
        int nextPe = start + nextPeInTeam * stride;

        // signal next pe
        ShmemiSignal<int32_t>((__gm__ uint8_t *)(syncArrayL2 + myPe), nextPe, count);

        // wait pre pe
        ShmemiWait<int32_t>((__gm__ uint8_t *)(syncArrayL2 + prePe), count);
        
        shift *= 2;
    } 

    store<int32_t>((__gm__ uint8_t *)syncCounterL2, count + 1);
}

/* Level 3: barrier between hosts, TO BE IMPLEMENTED.*/ 
SHMEM_AICORE_INLINE void ShmemiBarrierL3() {}

SHMEM_AICORE_INLINE void ShmemiBarrier(ShmemTeam_t tid) {
    ShmemTeam *team = getState()->teamPools[tid];

    int mype = team->mype;
    int start = team->start;
    int stride = team->stride;
    int size = team->size;

    if ((mype - start) % stride != 0) {
        // not in this team
        return;
    }

    ShmemiQuiet();

    ShmemiBarrierL1();

    if (AscendC::GetBlockIdx() == 0 && AscendC::GetSubBlockIdx() == 1) {
        ShmemiBarrierL2(team);
    }

    ShmemiBarrierL1();
}

#endif