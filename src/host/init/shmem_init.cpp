#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "acl/acl.h"
#include "shmemi_host_common.h"

using namespace std;

namespace shm {

#define DEFAULT_MY_PE -1
#define DEFAULT_N_PES -1
#define DEFAULT_FLAG 0
#define DEFAULT_ID 0
#define DEFAULT_TIMEOUT 120

// initializer
#define SHMEM_DEVICE_HOST_STATE_INITALIZER                                            \
    {                                                                                 \
        (1 << 16) + sizeof(shmemi_device_host_state_t),  /* version */                     \
            DEFAULT_MY_PE,                           /* mype */                       \
            DEFAULT_N_PES,                           /* npes */                       \
            NULL,                                    /* heap_base */                   \
            {NULL},                                  /* p2p_heap_base */                \
            {NULL},                                  /* sdma_heap_base */               \
            {NULL},                                  /* roce_heap_base */               \
            SIZE_MAX,                                /* heap_size */                   \
            {NULL},                                  /* team_pools */                  \
            NULL,                                    /* sync_pool */                  \
            NULL,                                    /* sync_counter */                \
            false,                                   /* shmem_is_shmem_initialized */ \
            false,                                   /* shmem_is_shmem_created */     \
            {0, 16 * 1024, 0},                       /* shmem_mte_config */           \
    }

shmemi_device_host_state_t g_state = SHMEM_DEVICE_HOST_STATE_INITALIZER;
shmem_init_attr_t g_attr;
static smem_shm_t g_smem_handle = nullptr;
static bool g_attr_init = false;
static char* g_ipport = nullptr;

int32_t version_compatible()
{
    int32_t status = SHMEM_SUCCESS;
    return status;
}

int32_t shmemi_options_init()
{
    int32_t status = SHMEM_SUCCESS;
    return status;
}

int32_t shmemi_state_init_attr(shmem_init_attr_t *attributes)
{
    int32_t status = SHMEM_SUCCESS;
    g_state.mype = attributes->my_rank;
    g_state.npes = attributes->n_ranks;
    g_state.heap_size = attributes->local_mem_size + SHMEM_EXTRA_SIZE;
    return status;
}

int32_t shmemi_heap_init(shmem_init_attr_t *attributes)
{
    void *gva = nullptr;
    int32_t status = SHMEM_SUCCESS;
    uint64_t smem_global_size = g_state.heap_size * g_state.npes;
    int32_t device_id;
    SHMEM_CHECK_RET(aclrtGetDevice(&device_id));

    status = smem_api::smem_init(DEFAULT_FLAG);
    if (status != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("smem_init Failed");
        return SHMEM_SMEM_ERROR;
    }
    smem_shm_config_t config;
    (void) smem_api::smem_shm_config_init(&config);
    status = smem_api::smem_shm_init(attributes->ip_port, attributes->n_ranks, attributes->my_rank, device_id, smem_global_size,
             &config);
    if (status != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("smem_init Failed");
        return SHMEM_SMEM_ERROR;
    }

    config.shmInitTimeout = attributes->option_attr.shm_init_timeout;
    config.shmCreateTimeout = attributes->option_attr.shm_create_timeout;
    config.controlOperationTimeout = attributes->option_attr.control_operation_timeout;

    g_smem_handle = smem_api::smem_shm_create(DEFAULT_ID, attributes->n_ranks, attributes->my_rank, g_state.heap_size,
                  static_cast<smem_shm_data_op_type>(attributes->option_attr.data_op_engine_type),DEFAULT_FLAG, &gva);

    if (g_smem_handle == nullptr || gva == nullptr) {
        SHM_LOG_ERROR("smem_shm_create Failed");
        return SHMEM_SMEM_ERROR;
    }
    g_state.heap_base = (void *) ((uintptr_t) gva + g_state.heap_size * attributes->my_rank);
    uint32_t reach_info = 0;
    for (int32_t i = 0; i < g_state.npes; i++) {
        status = smem_api::smem_shm_topo_can_reach(g_smem_handle, i, &reach_info);
        if (reach_info & SMEMS_DATA_OP_MTE) {
            g_state.p2p_heap_base[i] = (void *) ((uintptr_t) gva + g_state.heap_size * i);
        } else {
            g_state.p2p_heap_base[i] = NULL;
        }
        if (reach_info & SMEMS_DATA_OP_SDMA) {
            g_state.sdma_heap_base[i] = (void *) ((uintptr_t) gva + g_state.heap_size * i);
        } else {
            g_state.sdma_heap_base[i] = NULL;
        }
        if (reach_info & SMEMS_DATA_OP_ROCE) {
            g_state.roce_heap_base[i] = (void *) ((uintptr_t) gva + g_state.heap_size * i);
        } else {
            g_state.roce_heap_base[i] = NULL;
        }
    }
    if (shm::g_ipport != nullptr) {
        delete[] shm::g_ipport;
        attributes->ip_port = nullptr;
    } else {
         SHM_LOG_WARN("my_rank:" << attributes->my_rank << " shm::g_ipport is released in advance!");
         attributes->ip_port = nullptr;
    }
    g_state.is_shmem_created = true;
    return status;
}

int32_t shmemi_control_barrier_all()
{
    SHM_ASSERT_RETURN(g_smem_handle != nullptr, SHMEM_INVALID_PARAM);
    smem_shm_team_t obj = smem_api::smem_shm_get_global_team(g_smem_handle);
    SHM_ASSERT_RETURN(obj != nullptr, SHMEM_INVALID_PARAM);
    return smem_api::smem_shm_control_barrier(obj);
}

int32_t update_device_state()
{
    if (!g_state.is_shmem_created) {
        return SHMEM_NOT_INITED;
    }
    return smem_api::smem_shm_set_extra_context(g_smem_handle, (void *) &g_state, sizeof(shmemi_device_host_state_t));
}

int32_t check_attr(shmem_init_attr_t *attributes)
{
    if ((attributes->my_rank < 0) || (attributes->n_ranks <= 0)) {
        SHM_LOG_ERROR("my_rank:" << attributes->my_rank << " and n_ranks: " << attributes->n_ranks <<
            " cannot be less 0 , n_ranks still cannot be equal 0");
        return SHMEM_INVALID_VALUE;
    } else if (attributes->my_rank >= attributes->n_ranks) {
        SHM_LOG_ERROR("n_ranks:" << attributes->n_ranks << " cannot be less than my_rank:" << attributes->my_rank);
        return SHMEM_INVALID_PARAM;
    } else if (attributes->local_mem_size <= 0) {
        SHM_LOG_ERROR("local_mem_size:" << attributes->local_mem_size << " cannot be less or equal 0");
        return SHMEM_INVALID_VALUE;
    }
    return SHMEM_SUCCESS;
}

int32_t shmemi_load_lib()
{
    auto ret = shm::smem_api::load_library("");
    if (ret != SHMEM_SUCCESS) {
        SHM_LOG_ERROR("load smem library failed, please set LD_LIBRARY_PATH, ret: " << ret);
        return ret;
    }

    return SHMEM_SUCCESS;
}

} // namespace shm

int32_t shmem_set_data_op_engine_type(shmem_init_attr_t *attributes, data_op_engine_type_t value)
{
    attributes->option_attr.data_op_engine_type = value;
    return SHMEM_SUCCESS;
}

int32_t shmem_set_timeout(shmem_init_attr_t *attributes, uint32_t value)
{
    attributes->option_attr.shm_init_timeout = value;
    attributes->option_attr.shm_create_timeout = value;
    attributes->option_attr.control_operation_timeout = value;
    return SHMEM_SUCCESS;
}

int32_t shmem_set_attr(int32_t my_rank, int32_t n_ranks, uint64_t local_mem_size, const char *ip_port,
                       shmem_init_attr_t **attributes)
{
    *attributes = &shm::g_attr;
    size_t ip_len = strlen(ip_port);
    shm::g_ipport = new char[ip_len + 1];
    strcpy(shm::g_ipport, ip_port);
    if (shm::g_ipport == nullptr) {
        SHM_LOG_ERROR("my_rank:" << my_rank << " shm::g_ipport is nullptr!");
        return SHMEM_INVALID_VALUE;
    }
    int attr_version = (1 << 16) + sizeof(shmem_init_attr_t);
    shm::g_attr.my_rank = my_rank;
    shm::g_attr.n_ranks = n_ranks;
    shm::g_attr.ip_port = shm::g_ipport;
    shm::g_attr.local_mem_size = local_mem_size;
    shm::g_attr.option_attr = {attr_version, SHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    shm::g_attr_init = true;
    return SHMEM_SUCCESS;
}

int32_t shmem_init_status()
{
    if (!shm::g_state.is_shmem_created) return SHMEM_STATUS_NOT_INITALIZED;
    else if (!shm::g_state.is_shmem_initialized) return SHMEM_STATUS_SHM_CREATED;
    else if (shm::g_state.is_shmem_initialized) return SHMEM_STATUS_IS_INITALIZED;
    else return SHMEM_STATUS_INVALID;
}

int32_t shmem_init_attr(shmem_init_attr_t *attributes)
{
    int32_t ret;
    
    SHM_ASSERT_RETURN(attributes != nullptr, SHMEM_INVALID_PARAM);
    SHMEM_CHECK_RET(shm::check_attr(attributes));
    SHMEM_CHECK_RET(shm::version_compatible());
    SHMEM_CHECK_RET(shm::shmemi_options_init());

    SHMEM_CHECK_RET(shm::shmemi_state_init_attr(attributes));
    SHMEM_CHECK_RET(shm::shmemi_load_lib());
    SHMEM_CHECK_RET(shm::shmemi_heap_init(attributes));
    SHMEM_CHECK_RET(shm::update_device_state());

    SHMEM_CHECK_RET(shm::memory_manager_initialize(shm::g_state.heap_base, shm::g_state.heap_size));
    SHMEM_CHECK_RET(shm::shmemi_team_init(shm::g_state.mype, shm::g_state.npes));
    SHMEM_CHECK_RET(shm::update_device_state());
    shm::g_state.is_shmem_initialized = true;
    SHMEM_CHECK_RET(shm::shmemi_control_barrier_all());
    return SHMEM_SUCCESS;
}

int32_t shmem_finalize()
{
    SHMEM_CHECK_RET(shm::shmemi_team_finalize());
    if (shm::g_smem_handle != nullptr) {
        (void)shm::smem_api::smem_shm_destroy(shm::g_smem_handle, 0);
        shm::g_smem_handle = nullptr;
    }
    shm::smem_api::smem_un_init();
    return SHMEM_SUCCESS;
}