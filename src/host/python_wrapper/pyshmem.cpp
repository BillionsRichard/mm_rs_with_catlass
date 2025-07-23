/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <vector>

#include "shmem_api.h"

namespace py = pybind11;

namespace shm {
namespace {
inline std::string get_connect_url()
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

    auto port_int = std::strtol(port, nullptr, 10) + 11;
    return std::string("tcp://").append(address).append(":").append(std::to_string(port_int));
}

int shmem_initialize(int rank, int world_size, int64_t mem_size)
{
    shmem_init_attr_t attribute{
        rank, world_size, "", static_cast<uint64_t>(mem_size), {0, SHMEM_DATA_OP_MTE, 120, 120, 120}};
    auto url = get_connect_url();
    if (url.empty()) {
        std::cerr << "cannot get store connect URL(" << url << ") from ENV." << std::endl;
        return -1;
    }

    attribute.ip_port = url.c_str();
    auto ret = shmem_init_attr(&attribute);
    if (ret != 0) {
        std::cerr << "initialize with mype: " << rank << ", npes: " << world_size << " failed: " << ret;
        return ret;
    }

    return 0;
}
}
}

PYBIND11_MODULE(_pyaclshmem, m)
{
    m.def("aclshmem_init", &shm::shmem_initialize, py::call_guard<py::gil_scoped_release>(), py::arg("mype"),
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
    event(int): event_id used for sync
    )");

    m.def(
        "team_split_strided",
        [](int parent, int start, int stride, int size) {
            shmem_team_t new_team;
            auto ret = shmem_team_split_strided(parent, start, stride, size, &new_team);
            if (ret != 0) {
                std::cerr << "split parent team(" << parent << ") failed: " << ret << std::endl;
                return ret;
            }
            return new_team;
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
