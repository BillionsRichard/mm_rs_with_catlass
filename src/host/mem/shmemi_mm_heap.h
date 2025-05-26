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
struct memory_range {
    const uint64_t offset;
    const uint64_t size;

    memory_range(uint64_t o, uint64_t s) noexcept : offset{o}, size{s} {}
};

struct range_size_first_comparator {
    bool operator()(const memory_range &mr1, const memory_range &mr2) const noexcept;
};

class memory_heap {
public:
    memory_heap(void *base, uint64_t size) noexcept;
    ~memory_heap() noexcept;

public:
    void *allocate(uint64_t size) noexcept;
    void *aligned_allocate(uint64_t alignment, uint64_t size) noexcept;
    bool change_size(void *address, uint64_t size) noexcept;
    int32_t release(void *address) noexcept;
    bool allocated_size(void *address, uint64_t &size) const noexcept;

private:
    static uint64_t allocated_size_align_up(uint64_t input_size) noexcept;
    static bool alignment_matches(const memory_range &mr, uint64_t alignment, uint64_t size, uint64_t &head_skip) noexcept;
    void reduce_size_in_lock(const std::map<uint64_t, uint64_t>::iterator &pos, uint64_t new_size) noexcept;
    bool expend_size_in_lock(const std::map<uint64_t, uint64_t>::iterator &pos, uint64_t new_size) noexcept;

private:
    uint8_t *const base_;
    const uint64_t size_;
    mutable pthread_spinlock_t spinlock_{};
    std::map<uint64_t, uint64_t> address_idle_tree_;
    std::map<uint64_t, uint64_t> address_used_tree_;
    std::set<memory_range, range_size_first_comparator> size_idle_tree_;
};
}

#endif  // SHMEMI_MM_HEAP_H
