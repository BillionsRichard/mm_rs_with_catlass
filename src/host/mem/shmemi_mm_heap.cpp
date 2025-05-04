#include "shmemi_host_common.h"
#include "shmemi_mm_heap.h"

namespace shm {
bool RangeSizeFirstComparator::operator()(const MemoryRange &mr1, const MemoryRange &mr2) const noexcept
{
    if (mr1.size != mr2.size) {
        return mr1.size < mr2.size;
    }

    return mr1.offset < mr2.offset;
}

MemoryHeap::MemoryHeap(void *base, uint64_t size) noexcept : base_{reinterpret_cast<uint8_t *>(base)}, size_{size}
{
    pthread_spin_init(&spinlock_, 0);
    addressIdleTree_[0] = size;
    sizeIdleTree_.insert({0, size});
}

MemoryHeap::~MemoryHeap() noexcept
{
    pthread_spin_destroy(&spinlock_);
}

void *MemoryHeap::Allocate(uint64_t size) noexcept
{
    if (size == 0) {
        SHM_LOG_ERROR("cannot allocate with size 0.");
        return nullptr;
    }

    auto alignedSize = AllocateSizeAlignUp(size);
    MemoryRange anchor{0, alignedSize};

    pthread_spin_lock(&spinlock_);
    auto sizePos = sizeIdleTree_.lower_bound(anchor);
    if (sizePos == sizeIdleTree_.end()) {
        pthread_spin_unlock(&spinlock_);
        SHM_LOG_ERROR("cannot allocate with size: " << size);
        return nullptr;
    }

    auto targetOffset = sizePos->offset;
    auto targetSize = sizePos->size;
    auto addrPos = addressIdleTree_.find(targetOffset);
    if (addrPos == addressIdleTree_.end()) {
        pthread_spin_unlock(&spinlock_);
        SHM_LOG_ERROR("offset(" << targetOffset << ") size(" << targetSize << ") in size tree, not in address tree.");
        return nullptr;
    }

    sizeIdleTree_.erase(sizePos);
    addressIdleTree_.erase(addrPos);
    addressUsedTree_.emplace(targetOffset, alignedSize);
    if (targetSize > alignedSize) {
        MemoryRange left{targetOffset + alignedSize, targetSize - alignedSize};
        addressIdleTree_.emplace(left.offset, left.size);
        sizeIdleTree_.emplace(left);
    }
    pthread_spin_unlock(&spinlock_);

    return base_ + targetOffset;
}

void *MemoryHeap::AlignedAllocate(uint64_t alignment, uint64_t size) noexcept
{
    if (size == 0 || alignment == 0) {
        SHM_LOG_ERROR("alignment and size should not be zero.");
        return nullptr;
    }

    if ((alignment & (alignment - 1UL)) != 0) {
        SHM_LOG_ERROR("alignment should be power of 2, but real " << alignment);
        return nullptr;
    }

    uint64_t headSkip = 0;
    auto alignedSize = AllocateSizeAlignUp(size);
    MemoryRange anchor{0, alignedSize};

    pthread_spin_lock(&spinlock_);
    auto sizePos = sizeIdleTree_.lower_bound(anchor);
    while (sizePos != sizeIdleTree_.end() && !AlignmentMatches(*sizePos, alignment, alignedSize, headSkip)) {
        ++sizePos;
    }

    if (sizePos == sizeIdleTree_.end()) {
        pthread_spin_unlock(&spinlock_);
        SHM_LOG_ERROR("cannot allocate with size: " << size << ", alignment: " << alignment);
        return nullptr;
    }

    auto targetOffset = sizePos->offset;
    auto targetSize = sizePos->size;
    MemoryRange resultRange{sizePos->offset + headSkip, alignedSize};
    sizeIdleTree_.erase(sizePos);

    if (headSkip > 0) {
        sizeIdleTree_.emplace(MemoryRange{targetOffset, headSkip});
        addressIdleTree_.emplace(targetOffset, headSkip);
    }

    if (headSkip + alignedSize < targetSize) {
        MemoryRange leftMR{targetOffset + headSkip + alignedSize, targetSize - headSkip - alignedSize};
        sizeIdleTree_.emplace(leftMR);
        addressIdleTree_.emplace(leftMR.offset, leftMR.size);
    }

    addressUsedTree_.emplace(resultRange.offset, resultRange.size);
    pthread_spin_unlock(&spinlock_);

    return base_ + resultRange.offset;
}

bool MemoryHeap::ChangeSize(void *address, uint64_t size) noexcept
{
    auto u8a = reinterpret_cast<uint8_t *>(address);
    if (u8a < base_ || u8a >= base_ + size_) {
        SHM_LOG_ERROR("release invalid address " << address);
        return false;
    }

    if (size == 0) {
        Release(address);
        return true;
    }

    auto offset = u8a - base_;
    pthread_spin_lock(&spinlock_);
    auto pos = addressUsedTree_.find(offset);
    if (pos == addressUsedTree_.end()) {
        pthread_spin_unlock(&spinlock_);
        SHM_LOG_ERROR("change size for address " << address << " not allocated.");
        return false;
    }

    // size不变
    if (pos->second == size) {
        pthread_spin_unlock(&spinlock_);
        return true;
    }

    // 缩小size
    if (pos->second > size) {
        ReduceSizeInLock(pos, size);
        pthread_spin_unlock(&spinlock_);
        return true;
    }

    // 扩大size
    auto success = ExpendSizeInLock(pos, size);
    pthread_spin_unlock(&spinlock_);

    return success;
}

int32_t MemoryHeap::Release(void *address) noexcept
{
    auto u8a = reinterpret_cast<uint8_t *>(address);
    if (u8a < base_ || u8a >= base_ + size_) {
        SHM_LOG_ERROR("release invalid address " << address);
        return -1;
    }

    auto offset = u8a - base_;
    pthread_spin_lock(&spinlock_);
    auto pos = addressUsedTree_.find(offset);
    if (pos == addressUsedTree_.end()) {
        pthread_spin_unlock(&spinlock_);
        SHM_LOG_ERROR("release address " << address << " not allocated.");
        return -1;
    }

    auto size = pos->second;
    uint64_t finalOffset = offset;
    uint64_t finalSize = size;
    addressUsedTree_.erase(pos);

    auto prevAddrPos = addressIdleTree_.lower_bound(offset);
    if (prevAddrPos != addressIdleTree_.begin()) {
        --prevAddrPos;
        if (prevAddrPos != addressIdleTree_.end() && prevAddrPos->first + prevAddrPos->second == offset) {
            // 合并前一个range
            finalOffset = prevAddrPos->first;
            finalSize += prevAddrPos->second;
            addressIdleTree_.erase(prevAddrPos);
            sizeIdleTree_.erase(MemoryRange{prevAddrPos->first, prevAddrPos->second});
        }
    }

    auto nextAddrPos = addressIdleTree_.find(offset + size);
    if (nextAddrPos != addressIdleTree_.end()) {  // 合并后一个range
        finalSize += nextAddrPos->second;
        addressIdleTree_.erase(nextAddrPos);
        sizeIdleTree_.erase(MemoryRange{nextAddrPos->first, nextAddrPos->second});
    }
    addressIdleTree_.emplace(finalOffset, finalSize);
    sizeIdleTree_.emplace(MemoryRange{finalOffset, finalSize});
    pthread_spin_unlock(&spinlock_);

    return 0;
}

bool MemoryHeap::AllocatedSize(void *address, uint64_t &size) const noexcept
{
    auto u8a = reinterpret_cast<uint8_t *>(address);
    if (u8a < base_ || u8a >= base_ + size_) {
        SHM_LOG_ERROR("release invalid address " << address);
        return false;
    }

    auto offset = u8a - base_;
    bool exist = false;
    pthread_spin_lock(&spinlock_);
    auto pos = addressUsedTree_.find(offset);
    if (pos != addressUsedTree_.end()) {
        exist = true;
        size = pos->second;
    }
    pthread_spin_unlock(&spinlock_);

    return exist;
}

uint64_t MemoryHeap::AllocateSizeAlignUp(uint64_t inputSize) noexcept
{
    constexpr uint64_t alignSize = 16UL;
    constexpr uint64_t alignSizeMask = ~(alignSize - 1UL);
    return (inputSize + alignSize - 1UL) & alignSizeMask;
}

bool MemoryHeap::AlignmentMatches(const MemoryRange &mr, uint64_t alignment, uint64_t size, uint64_t &headSkip) noexcept
{
    if (mr.size < size) {
        return false;
    }

    if ((mr.offset & (alignment - 1UL)) == 0UL) {
        headSkip = 0;
        return true;
    }

    auto alignedOffset = ((mr.offset + alignment - 1UL) & (~(alignment - 1UL)));
    headSkip = alignedOffset - mr.offset;
    return mr.size >= size + headSkip;
}

void MemoryHeap::ReduceSizeInLock(const std::map<uint64_t, uint64_t>::iterator &pos, uint64_t newSize) noexcept
{
    auto offset = pos->first;
    auto oldSize = pos->second;
    pos->second = newSize;
    auto nextAddrPos = addressIdleTree_.find(offset + oldSize);
    if (nextAddrPos == addressIdleTree_.end()) {
        addressIdleTree_.emplace(offset + newSize, oldSize - newSize);
        sizeIdleTree_.emplace(MemoryRange{offset + newSize, oldSize - newSize});
    } else {
        auto nextSizePos = sizeIdleTree_.find(MemoryRange{nextAddrPos->first, nextAddrPos->second});
        sizeIdleTree_.erase(nextSizePos);
        nextAddrPos->second += (oldSize - newSize);
        sizeIdleTree_.emplace(MemoryRange{nextAddrPos->first, nextAddrPos->second});
    }
}

bool MemoryHeap::ExpendSizeInLock(const std::map<uint64_t, uint64_t>::iterator &pos, uint64_t newSize) noexcept
{
    auto offset = pos->first;
    auto oldSize = pos->second;
    auto delta = newSize - oldSize;

    auto nextAddrPos = addressIdleTree_.find(offset + oldSize);
    if (nextAddrPos == addressIdleTree_.end() || nextAddrPos->second < delta) {
        return false;
    }

    pos->second = newSize;
    auto nextSizePos = sizeIdleTree_.find(MemoryRange{nextAddrPos->first, nextAddrPos->second});
    if (nextAddrPos->second == delta) {
        sizeIdleTree_.erase(nextSizePos);
        addressIdleTree_.erase(nextAddrPos);
    } else {
        sizeIdleTree_.erase(nextSizePos);
        nextAddrPos->second -= delta;
        sizeIdleTree_.emplace(MemoryRange{nextAddrPos->first, nextAddrPos->second});
    }

    return true;
}
}