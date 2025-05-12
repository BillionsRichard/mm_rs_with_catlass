/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 */
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>

#include "torch/extension.h"
#include "ATen/ops/from_blob.h"
#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/aten/common/from_blob.h"
#include "host/shmem_host_init.h"
#include "host/shmem_host_heap.h"
#include "host/shmem_host_rma.h"
#include "host/shmem_host_team.h"
#include "host/shmem_host_sync.h"

namespace py = pybind11;

namespace shm {
namespace {
inline std::string GetConnectURL()
{
    auto address = std::getenv("SHMEM_MASTER_ADDR");
    auto port = std::getenv("SHMEM_MASTER_PORT");
    if (address != nullptr && port != nullptr) {
        return std::string("tcp://").append(address).append(":").append(port);
    }

    address = std::getenv("MASTER_ADDR");
    port = std::getenv("MASTER_PORT");
    if (address == nullptr || port == nullptr) {
        return "";
    }

    auto portInt = std::strtol(port, nullptr, 10) + 11;
    return std::string("tcp://").append(address).append(":").append(std::to_string(portInt));
}

int ShmemInitialize(int rank, int worldSize, int64_t memSize)
{
    shmem_init_attr_t attribute{
        0, rank, worldSize, "", static_cast<uint64_t>(memSize), {SHMEM_DATA_OP_MTE, 120, 120, 120}};
    auto url = GetConnectURL();
    if (url.empty()) {
        std::cerr << "cannot get store connect URL(" << url << ") from ENV." << std::endl;
        return -1;
    }

    attribute.ipPort = url.c_str();
    auto ret = shmem_init_attr(&attribute);
    if (ret != 0) {
        std::cerr << "initialize with mype: " << rank << ", npes: " << worldSize << " failed: " << ret;
        return ret;
    }

    return 0;
}
}
}

PYBIND11_MODULE(_pyaclshmem, m)
{
    m.def("aclshmem_init", &shm::ShmemInitialize, py::call_guard<py::gil_scoped_release>(), py::arg("mype"),
          py::arg("npes"), py::arg("mem_size"), R"(
Initialize share memory module.

Arguments:
    mype(int): local processing element index, range in [0, npes).
    npes(int): total count of processing elements.
    mem_size(int): memory size for each processing element in bytes.
Returns:
    returns zero on success. On error, -1 is returned.
    )");

    m.def("aclshmem_finialize", &shmem_finalize, py::call_guard<py::gil_scoped_release>(),
          R"(
Finalize share memory module.
    )");

    m.def(
        "aclshmem_malloc",
        [](size_t size) {
            auto ptr = shmem_malloc(size);
            if (ptr == nullptr) {
                throw std::runtime_error("aclshmem_malloc failed");
            }
            return (intptr_t)ptr;
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("size"),
        R"(
Allocates size bytes and returns a pointer to the allocated memory. The memory is not initialized. If size is 0, then
aclshmem_malloc() returns NULL.
    )");

    m.def(
        "aclshmem_free",
        [](intptr_t ptr) {
            auto mem = (void *)ptr;
            shmem_free(mem);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("ptr"),
        R"(
Frees the memory space pointed to by ptr, which must have been returned by a previous call to aclshmem_malloc.
    )");

    m.def(
        "aclshmem_ptr", [](intptr_t ptr, int pe) { return (intptr_t)shmem_ptr((void *)ptr, pe); },
        py::call_guard<py::gil_scoped_release>(), py::arg("ptr"), py::arg("peer"), R"(
Get address that may be used to directly reference dest on the specified PE.

Arguments:
    ptr(int): The symmetric address of the remotely accessible data.
    pe(int): PE number
    )");

    m.def("my_pe", &shmem_my_pe, py::call_guard<py::gil_scoped_release>(), R"(Get my PE number.)");

    m.def("pe_count", &shmem_n_pes, py::call_guard<py::gil_scoped_release>(), R"(Get number of PEs.)");

    m.def("mte_set_ub_params", &shmem_mte_set_ub_params, py::call_guard<py::gil_scoped_release>(), py::arg("offset"),
          py::arg("size"), py::arg("event"), R"(
Set the params of UB used for MTE operation initiated by NPU.

Arguments:
    offset(int): start offset of UB
    size(int): size of UB
    event(int): eventId used for sync
    )");

    m.def(
        "team_split_strided",
        [](int parent, int start, int stride, int size) {
            shmem_team_t newTeam;
            auto ret = shmem_team_split_strided(parent, start, stride, size, &newTeam);
            if (ret != 0) {
                std::cerr << "split parent team(" << parent << ") failed: " << ret << std::endl;
                return ret;
            }
            return newTeam;
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("parent"), py::arg("start"), py::arg("stride"),
        py::arg("size"), R"(
Split team from an existing parent team, this is a collective operation.

Arguments:
    parent(int): parent team id
    start(int): the lowest PE number of the subset of PEs from parent team that will form the new team
    stride(int): the stride between team PE numbers in the parent team
    size(int): the number of PEs from the parent team
Returns:
    On success, returns new team id. On error, -1 is returned.
    )");

    m.def("team_translate_pe", &shmem_team_translate_pe, py::call_guard<py::gil_scoped_release>(),
          py::arg("src_team"), py::arg("src_pe"), py::arg("dest_team"), R"(
Translate a given PE number in one team into the corresponding PE number in another team

Arguments:
    src_team(int): source team id
    src_pe(int): source PE number
    dest_team(int): destination team id
Returns:
    On success, returns the specified PE’s number in the dest_team. On error, -1 is returned.
    )");

    m.def("team_destroy", &shmem_team_destroy, py::call_guard<py::gil_scoped_release>(), py::arg("team"), R"(
Destroy a team with team id

Arguments:
    team(int): team id to be destroyed
    )");

    m.def("barrier_all", &shmem_barrier_all, py::call_guard<py::gil_scoped_release>(), R"(
Do barrier operation on global team with default stream.
    )");

    m.def("barrier_team", &shmem_barrier, py::call_guard<py::gil_scoped_release>(), py::arg("team"), R"(
Do barrier operation on specified team with default stream

Arguments:
    team(int): team id
    )");

    m.def("my_pe", &shmem_team_my_pe, py::call_guard<py::gil_scoped_release>(), py::arg("team"), R"(
Get my PE number within a team, i.e. index of the PE

Arguments:
    team(int): team id
Returns:
    On success, returns the PE’s number in the specified team. On error, -1 is returned.
    )");

    m.def("pe_count", &shmem_team_n_pes, py::call_guard<py::gil_scoped_release>(), py::arg("team"), R"(
Get number of PEs with in a team, i.e. how many PEs in the team.

Arguments:
    team(int): team id
Returns:
    On success, returns total number of PEs in the specified team. On error, -1 is returned.
    )");
}
