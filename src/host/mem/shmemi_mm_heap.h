/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 */
#ifndef SHMEMI_MM_HEAP_H
#define SHMEMI_MM_HEAP_H

#include <pthread.h>
#include <cstdint>
#include <map>
#include <set>

namespace shm {
struct MemoryRange {
    const uint64_t offset;
    const uint64_t size;

    MemoryRange(uint64_t o, uint64_t s) noexcept : offset{o}, size{s} {}
};

struct RangeSizeFirstComparator {
    bool operator()(const MemoryRange &mr1, const MemoryRange &mr2) const noexcept;
};

class MemoryHeap {
public:
    MemoryHeap(void *base, uint64_t size) noexcept;
    ~MemoryHeap() noexcept;

public:
    void *Allocate(uint64_t size) noexcept;
    void *AlignedAllocate(uint64_t alignment, uint64_t size) noexcept;
    bool ChangeSize(void *address, uint64_t size) noexcept;
    int32_t Release(void *address) noexcept;
    bool AllocatedSize(void *address, uint64_t &size) const noexcept;

private:
    static uint64_t AllocateSizeAlignUp(uint64_t inputSize) noexcept;
    static bool AlignmentMatches(const MemoryRange &mr, uint64_t alignment, uint64_t size, uint64_t &headSkip) noexcept;
    void ReduceSizeInLock(const std::map<uint64_t, uint64_t>::iterator &pos, uint64_t newSize) noexcept;
    bool ExpendSizeInLock(const std::map<uint64_t, uint64_t>::iterator &pos, uint64_t newSize) noexcept;

private:
    uint8_t *const base_;
    const uint64_t size_;
    mutable pthread_spinlock_t spinlock_{};
    std::map<uint64_t, uint64_t> addressIdleTree_;
    std::map<uint64_t, uint64_t> addressUsedTree_;
    std::set<MemoryRange, RangeSizeFirstComparator> sizeIdleTree_;
};
}

#endif  // SHMEMI_MM_HEAP_H
