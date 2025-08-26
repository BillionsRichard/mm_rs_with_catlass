unset LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Get project root directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=$( dirname $( dirname $(dirname "$SCRIPT_DIR")))

export LD_LIBRARY_PATH=${PROJECT_ROOT}/install/memfabric_hybrid/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PROJECT_ROOT}/install/shmem/lib/:$LD_LIBRARY_PATH
export SHMEM_HOME_PATH=${PROJECT_ROOT}/install
export MEMFABRIC_HYBRID_TLS_ENABLE=0