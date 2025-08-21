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
#include <iostream>

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
    // use pta addr:port+11 if shmem env not set
    address = std::getenv("MASTER_ADDR");
    port = std::getenv("MASTER_PORT");
    if (address == nullptr || port == nullptr) {
        std::cerr << "[ERROR] invlaid address and port" << std::endl;
        return "";
    }

    char *endptr;
    const long usePtaPortOffset = 11;
    auto port_long = std::strtol(port, &endptr, 10);
    // master port + 11 as non-master port.
    if (endptr == port || *endptr != '\0' || port_long <= 0 || port_long > UINT16_MAX - usePtaPortOffset) {
        // SHM_LOG_ERROR is not available in this file, use std::cerr
        std::cerr << "[ERROR] Invalid MASTER_PORT value from environment: " << port << std::endl;
        return "";
    }
    port_long = port_long + usePtaPortOffset;
    return std::string("tcp://").append(address).append(":").append(std::to_string(port_long));
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

static int py_decrypt_handler_wrapper(const char *cipherText, size_t cipherTextLen, char *plainText, size_t &plainTextLen)
{
    if (cipherTextLen > MAX_CIPHER_LEN || !g_py_decrypt_func || g_py_decrypt_func.is_none()) {
        std::cerr << "input cipher len is too long or decrypt func invalid." << std::endl;
        return -1;
    }

    try {
        py::str py_cipher = py::str(cipherText, cipherTextLen);
        std::string plain = py::cast<std::string>(g_py_decrypt_func(py_cipher).cast<py::str>());
        if (plain.size() >= plainTextLen) {
            std::cerr << "output cipher len is too long" << std::endl;
            return -1;
        }

        std::copy(plain.begin(), plain.end(), plainText);
        plainText[plain.size()] = '\0';
        plainTextLen = plain.size();
        return 0;
    } catch (const py::error_already_set &e) {
        return -1;
    }
}

int32_t register_python_decrypt_handler(py::function py_decrypt_func)
{
    if (!py_decrypt_func || py_decrypt_func.is_none()) {
        return shmem_register_decrypt_handler(nullptr);
    }

    g_py_decrypt_func = py_decrypt_func;
    return shmem_register_decrypt_handler(py_decrypt_handler_wrapper);
}

int32_t shmem_set_conf_store_tls_info(bool enable, std::string &tls_info)
{
    return shmem_set_conf_store_tls(enable, tls_info.c_str(), tls_info.size());
}
}
}

PYBIND11_MODULE(_pyshmem, m)
{
    m.def("shmem_init", &shm::shmem_initialize, py::call_guard<py::gil_scoped_release>(), py::arg("mype"),
          py::arg("npes"), py::arg("mem_size"), R"(
Initialize share memory module.

Arguments:
    mype(int): local processing element index, range in [0, npes).
    npes(int): total count of processing elements.
    mem_size(int): memory size for each processing element in bytes.
Returns:
    returns zero on success. On error, -1 is returned.
    )");

    m.def("shmem_finialize", &shmem_finalize, py::call_guard<py::gil_scoped_release>(),
          R"(
Finalize share memory module.
    )");

    m.def("register_decrypt_handler", &shm::register_python_decrypt_handler, py::call_guard<py::gil_scoped_release>(),
          py::arg("py_decrypt_func"), R"(
Register a Python decrypt handler.
Parameters:
    py_decrypt_func (callable): Python function that accepts (str cipher_text) and returns (str plain_text)
        cipher_text: the encrypted text (private key password)
        plain_text: the decrypted text (private key password)
Returns:
    returns zero on success. On error, none-zero is returned.
)");

    m.def("set_conf_store_tls", &shm::shmem_set_conf_store_tls_info, py::call_guard<py::gil_scoped_release>(),
          py::arg("enable"), py::arg("tls_info"), R"(
set the config store tls info.
Parameters:
    enable (boolean): enable config store tls or not
        tls_info (string): tls config string
Returns:
    returns zero on success. On error, none-zero is returned.
)");


    m.def(
        "shmem_malloc",
        [](size_t size) {
            auto ptr = shmem_malloc(size);
            if (ptr == nullptr) {
                throw std::runtime_error("shmem_malloc failed");
            }
            return (intptr_t)ptr;
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("size"),
        R"(
Allocates size bytes and returns a pointer to the allocated memory. The memory is not initialized. If size is 0, then
shmem_malloc() returns NULL.
    )");

    m.def(
        "shmem_free",
        [](intptr_t ptr) {
            auto mem = (void *)ptr;
            shmem_free(mem);
        },
        py::call_guard<py::gil_scoped_release>(), py::arg("ptr"),
        R"(
Frees the memory space pointed to by ptr, which must have been returned by a previous call to shmem_malloc.
    )");

    m.def(
        "shmem_ptr", [](intptr_t ptr, int pe) { return (intptr_t)shmem_ptr((void *)ptr, pe); },
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
    On success, returns the specified PE's number in the dest_team. On error, -1 is returned.
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
    On success, returns the PE's number in the specified team. On error, -1 is returned.
    )");

    m.def("pe_count", &shmem_team_n_pes, py::call_guard<py::gil_scoped_release>(), py::arg("team"), R"(
Get number of PEs with in a team, i.e. how many PEs in the team.

Arguments:
    team(int): team id
Returns:
    On success, returns total number of PEs in the specified team. On error, -1 is returned.
    )");
}
