unset LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/home/z00583051/codes/tmp_quant_matmul_reduce_scatter/3rdparty/shmem/install/memfabric_hybrid/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/z00583051/codes/tmp_quant_matmul_reduce_scatter/3rdparty/shmem/install/shmem/lib/:$LD_LIBRARY_PATH
export SHMEM_HOME_PATH=/home/z00583051/codes/tmp_quant_matmul_reduce_scatter/3rdparty/shmem/install
export MEMFABRIC_HYBRID_TLS_ENABLE=0