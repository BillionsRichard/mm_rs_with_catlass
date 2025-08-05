source /path/to/ascend-toolkit/set_env.sh
export SHMEM_HOME_PATH=/path/to/shmem/install
export LD_LIBRARY_PATH=$SHMEM_HOME_PATH/shmem/lib:$SHMEM_HOME_PATH/memfabric_hybrid/lib:$LD_LIBRARY_PATH

rm -rf ../../build

bash scripts/build.sh
bash scripts/run.sh 1 27 0 10 0,1

