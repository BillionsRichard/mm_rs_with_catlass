#include <iostream>
#include <cstdlib>
#include <string>
#include <acl/acl.h>

using namespace std;

#include <mpi.h>
#include <mutex>
#include <vector>

#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "runtime/dev.h"

#include "init.h"

int main(int argc, char** argv) {

    aclInit(nullptr);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    aclrtSetDevice(rank);

    string pFlag = "[Process " + to_string(rank) + "] ";

    shmem_init(rank, size);

    // #################### 子通信域切分测试 ############################
    shmem_team_t team_odd, team_even;
    int start = 1;
    int stride = 2;
    int team_size = 4;
    auto status = shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, team_size, team_odd);

    // #################### host侧取值测试 ##############################
    std::cout << pFlag << "shmem_team_n_pes(team_odd): " << shmem_team_n_pes(team_odd) << std::endl;
    std::cout << pFlag << "shmem_team_mype(team_odd): " << shmem_team_mype(team_odd) << std::endl;
    std::cout << pFlag << "shmem_n_pes(): " << shmem_n_pes() << std::endl;
    std::cout << pFlag << "shmem_mype(): " << shmem_mype() << std::endl;

    // 保证前序子team创建完成，有个全局数组
    MPI_Barrier(MPI_COMM_WORLD);

    start = 0;
    stride = 2;
    team_size = 4;
    status = shmem_team_split_strided(SHMEM_TEAM_WORLD, start, stride, team_size, team_even);

    // 保证子team创建完成
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << pFlag << "shmem_team_translate_pe(team_even, 2, SHMEM_TEAM_WORLD): " << shmem_team_translate_pe(team_even, 2, SHMEM_TEAM_WORLD) << std::endl;

    // #################### 相关资源释放 ##############################
    shmem_team_destroy(team_odd);
    shmem_team_destroy(team_even);

    shmem_finalize();
    MPI_Finalize();
    aclFinalize();
    return 0;
}