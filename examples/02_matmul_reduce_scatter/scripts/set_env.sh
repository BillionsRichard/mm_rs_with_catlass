unset LD_LIBRARY_PATH
# source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/tmp/ascend
export LD_LIBRARY_PATH=/tmp/shmem/install/memfabric_hybrid/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/tmp/shmem/install/shmem/lib/:$LD_LIBRARY_PATH
export SHMEM_HOME_PATH=/tmp/shmem/install
export MEMFABRIC_HYBRID_TLS_ENABLE=0