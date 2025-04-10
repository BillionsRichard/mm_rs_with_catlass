#ifndef _SHMEM_DEVICE_MEM_GETPUT_HPP_
#define _SHMEM_DEVICE_MEM_GETPUT_HPP_

#include "kernel_operator.h"
#include "smem_shm_aicore.h"
#include "smem_shm_aicore_common.h"

constexpr uint64_t gNpuMallocSpace = 1024UL * 1024UL * 1024;

template <typename T>
class ShmemMem {
public:
    __aicore__ inline ShmemMem() {}
    __aicore__ inline void Init(__gm__ T* dst, __gm__ T* src, __ubuf__ T* tmpUb, uint32_t size, uint32_t ubSize, int EVENT_ID)
    {
        dstGM = dst;
        srcGM = src;
        tmpUbuf = tmpUb;

        // one instrution dataSize.
        blockSize = ubSize / sizeof(T) * sizeof(T);

        elemSize = size;
        remain = size % blockSize;
    }
    __aicore__ inline void Process()
    {
        if (AscendC::GetSubBlockIdx() != 0) {
            return;
        }
        // TODO: USE DoubleBuffer.
        int repeat_times = elemSize / blockSize;
        for (int i = 0; i < repeat_times; i++){
            smem_copy_gm2ub(tmpUbuf, srcGM + i * blockSize * sizeof(T), blockSize);
            AscendC::TQueSync<PIPE_MTE2, PIPE_MTE3> sync1;
            sync1.SetFlag(eventID);
            sync1.WaitFlag(eventID);
            smem_copy_ub2gm(srcGM + i * blockSize * sizeof(T), tmpUbuf, blockSize);
            AscendC::TQueSync<PIPE_MTE3, PIPE_MTE2> sync2;
            sync2.SetFlag(eventID);
            sync2.WaitFlag(eventID);
        }
        if (remain > 0) {
            smem_copy_gm2ub(tmpUbuf, srcGM + repeat_times * blockSize * sizeof(T), remain);
            AscendC::TQueSync<PIPE_MTE2, PIPE_MTE3> sync1;
            sync1.SetFlag(eventID);
            sync1.WaitFlag(eventID);
            smem_copy_ub2gm(srcGM + repeat_times * blockSize * sizeof(T), tmpUbuf, remain);
        }
    }
private:
    __gm__ T* dstGM = nullptr;
    __gm__ T* srcGM = nullptr;

    __ubuf__ T* tmpUbuf = nullptr;

    int64_t elemSize = 0;
    int64_t remain = 0;
    uint32_t blockSize = 0; 
    TEventID eventID = 0;
};

template <typename T>
__aicore__ inline ShmemCopyUbuf(__ubuf__ T* srcUb, uint32_t size)
{
    smem_set_copy_ubuf(srcUb, size);
}

template <typename T>
__aicore__ inline ShmemMTEGetMem(__gm__ T* dst, __gm__ T* src, uint32_t copySize, int pe, TEventID EVENT_ID)
{
    ShmemMem<T> memKernel;
    // address translate
    uint64_t offset = SMEM_GET_SYMMETRIC_SIZE();
    uint64_t src64 = reinterpret_cast<uint64_t>(src) + offset * pe;

    memKernel.Init(dst, reinterpret_cast<__gm__ T*>(src64), reinterpret_cast<__ubuf__ T*>(globalUbuf), copySize, globalUbSize, EVENT_ID);
    memKernel.Process();
}




#endif // _SHMEM_DEVICE_MEM_GETPUT_HPP_