#put_get接口测试
add_executable(scalar_putget_kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/test_scalar_get_put.cpp
)

target_compile_options(scalar_putget_kernels
    PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

target_include_directories(scalar_putget_kernels
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/output/include
)

target_link_directories(scalar_putget_kernels
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/output
)

target_link_libraries(scalar_putget_kernels
    PRIVATE
    shmem
    ascendc_kernels_npu
    mf_smem
    mf_hybrid_bmm
)

# team能力测试
add_executable(team_example
    team_example.cpp
)

target_include_directories(team_example
    PRIVATE
    ${MPI_CXX_INCLUDE_PATH}
)

target_link_libraries(team_example
    PRIVATE
    ${MPI_CXX_LIBRARIES}
    shmem
    ascendc_kernels_npu
)
