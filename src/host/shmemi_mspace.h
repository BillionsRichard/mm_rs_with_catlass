#ifndef _MSPACE_H
#define _MSPACE_H
#include <cstddef>
#include <map>

#define SHMEMI_MALLOC_ALIGNMENT ((size_t)512)
#define SHMEMI_ALIGN_MASK (0x1FF) // 512 - 1


class Mspace {
public:
    Mspace() = default;
    ~Mspace() = default;

    void AddNewChunk(void *base, size_t size);
    void AddFreeChunk(void *base, size_t size);
    void *alloc(size_t size);
    void free(void *mem);
    void *allocAlign(size_t alignment, size_t size);

private:
    size_t totalSize_ = 0;
    std::map<void *, size_t> freeChunkStart_;
    std::map<void *, size_t> freeChunkEnd_;
    std::map<void *, size_t> inuseChunkStart_;
};
#endif