import os
import torch
import torch.distributed as dist
import torch_npu
import shmem as ash


g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024

def decypt_handler_test(input_cipher):
    return input_cipher

def run_register_decrypt_tests():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # 1. test init
    ret = ash.shmem_init(rank, world_size, g_ash_size)
    if ret != 0:
        raise ValueError('[ERROR] shmem_init failed')

    # 2. test register
    ret = ash.register_decrypt_handler(decypt_handler_test)
    print(f'rank[{rank}]: register hander ret={ret}')

    # 3. test finialize
    _ = ash.shmem_finialize()

def run_set_tls_info():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # 1. test set tls info
    ret = ash.set_conf_store_tls(False, "")

    # 2. test init
    ret = ash.shmem_init(rank, world_size, g_ash_size)
    if ret != 0:
        raise ValueError('[ERROR] shmem_init failed')

    print(f'rank[{rank}]: register hander ret={ret}')

    # 3. test finialize
    _ = ash.shmem_finialize()


def run_tests():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # 1. test init
    ret = ash.shmem_init(rank, world_size, g_ash_size)
    if ret != 0:
        raise ValueError('[ERROR] shmem_init failed')
    # 2. test malloc
    shmem_ptr = ash.shmem_malloc(g_malloc_size)
    print(f'rank[{rank}]: shmem_ptr:{shmem_ptr} with type{type(shmem_ptr)}')
    if shmem_ptr is None:
        raise ValueError('[ERROR] shmem_malloc failed')
    # 3. test pe
    my_pe, pe_count = ash.my_pe(), ash.pe_count()
    print(f'rank[{rank}]: my_pe:{my_pe} and pe_count:{pe_count}')
    if not (my_pe == rank and pe_count == world_size):
        raise ValueError('[ERROR] pe/world failed')
    # 4. test free
    _ = ash.shmem_free(shmem_ptr)
    # 5. test finialize
    _ = ash.shmem_finialize()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    # shmem init must comes after torch.npu.set_device(or any other aclInit device action)
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", rank=local_rank)
    run_tests()
    print("test.py running success!")
