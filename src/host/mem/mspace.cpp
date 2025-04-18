#include <cstdio>
#include <vector>
#include <iostream>

#include "mspace.h"

#define alignSize(size) (((size) + SHMEMI_ALIGN_MASK) & (~SHMEMI_ALIGN_MASK))

void Mspace::AddNewChunk(void *base, size_t size)
{
    totalSize_ += size;
    AddFreeChunk(base, size);
}

void Mspace::AddFreeChunk(void *base, size_t size)
{
    size_t mergeSize = size;
    char *mergeStart = (char *)base;
    char *mergeEnd = (char *)base + size;

    // merge with free chunk before
    if (freeChunkEnd_.find(base) != freeChunkEnd_.end()) {
        size_t asize = freeChunkEnd_[base];
        mergeSize += asize;
        mergeStart -= asize;
        freeChunkEnd_.erase(base);
    }

    // merge with free chunk after
    if (freeChunkStart_.find(mergeEnd) != freeChunkStart_.end()) {
        size_t bsize = freeChunkStart_[mergeEnd];
        mergeSize += bsize;
        mergeEnd += bsize;
        freeChunkStart_.erase((char *)base + size);
    }

    freeChunkEnd_[mergeEnd] = mergeSize;
    freeChunkStart_[mergeStart] = mergeSize;
}

void *Mspace::alloc(size_t size)
{
    if (size == 0) {
        return nullptr;
    }
    size = alignSize(size);
    for (auto it : freeChunkStart_) {
        if (it.second >= size) {
            char *base = (char *)it.first;
            auto remainSize = it.second - size;
            if (remainSize > 0) {
                char *newBase = base + size;
                freeChunkStart_[newBase] = remainSize;
                freeChunkEnd_[newBase + remainSize] = remainSize;
                freeChunkStart_.erase(base);
            } else {
                freeChunkEnd_.erase(base + it.second);
                freeChunkStart_.erase(base);
            }
            inuseChunkStart_[base] = size;
            return base;
        }
    }
    return nullptr;
}

void Mspace::free(void *mem)
{
    auto pos = inuseChunkStart_.find(mem);
    if (pos == inuseChunkStart_.end()) {
        printf("free error\n");
        return;
    }
    size_t size = pos->second;
    inuseChunkStart_.erase(pos);

    AddFreeChunk((char *)mem, size);
}

void *Mspace::allocAlign(size_t alignment, size_t size)
{
    if ((alignment % sizeof(void *) != 0) || ((alignment & (alignment - 1)) != 0)) {
        printf("error");
    }
    size = alignSize(size);
    size = (size + (alignment - 1)) & (~(alignment - 1)); // check

    auto base = alloc(size);
    return base;
}