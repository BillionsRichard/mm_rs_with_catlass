import os
import torch
import torch.distributed as dist
import torch_npu
import aclshmem as ash


g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024


def run_tests():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # 1. test init
    ret = ash.aclshmem_init(rank, world_size, g_ash_size)
    if ret != 0:
        raise ValueError('[ERROR] aclshmem_init failed')
    # 2. test malloc
    shmem_ptr = ash.aclshmem_malloc(g_malloc_size)
    print(f'rank[{rank}]: shmem_ptr:{shmem_ptr} with type{type(shmem_ptr)}')
    if shmem_ptr is None:
        raise ValueError('[ERROR] aclshmem_malloc failed')
    # 3. test pe
    my_pe, pe_count = ash.my_pe(), ash.pe_count()
    print(f'rank[{rank}]: my_pe:{my_pe} and pe_count:{pe_count}')
    if not (my_pe == rank and pe_count == world_size):
        raise ValueError('[ERROR] pe/world failed')
    # 4. test free
    _ = ash.aclshmem_free(shmem_ptr)
    # 5. test finialize
    _ = ash.aclshmem_finialize()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    # aclshmem init must comes after torch.npu.set_device(or any other aclInit device action)
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=local_rank)
    run_tests()
