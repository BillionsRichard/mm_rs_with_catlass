使用方式：开发自测试用

1.将稳定版本解压到3rdparty目录下

2.bash scripts/build.sh

3.bash scripts/run.sh

跑默认8卡用例

4.bash scripts/run.sh -ranks 8 -ipport tcp://127.0.0.1:8666 -gnpus 8

run.sh目前支持-ranks -ipport -gnpus三个入参分别设置总rank数，ip和端口，单机卡数。

不输入参数时各参数默认值分别为8，tcp://127.0.0.1:8666，8。


用户使用指南：
1. 下载代码 （直接使用SHMEM交付件开发算子的场景，可以跳过这个步骤1和2）
   下载代码请复制以下命令到终端执行
   git clone https://gitee.com/ascend/shmem.git

2. 编译SHMEM
   a. 确保编译机上正确安装了CANN。对于内部用户， 可以执行下面的命令快速安装：
       cd /opt/package/
       bash install_and_enavle_cann.sh
       source /usr/local/Ascend/ascend-toolkit/set_env.sh
   b. 获取依赖包 memfibric_bybrid.zip, 并解压到 SHMEMd的3rdparty目录下
       正式获取地址：待补充
       解压方式： unzip memfibric_bybrid.zip
   c. 编译SHMEM
       SHMEM提供了完整的编译脚本， 直接在编译机的终端运行下面的脚本即可
       sh scripts/build.sh

3. 拷贝SHMEM的输出到算子编译环境
   a. 当前SHMEM的输出在 output目录下， 包含静态编译文件lib/*, 对外头文件include/*。 需要统一拷贝。
   
        (base) [root@AscendLab-247 install]# ls
        include  lib  lib64
        (base) [root@AscendLab-247 install]# ls include/
        constants.h  data_utils.h  mem_device.h  mem.h  shmem_api.h  shmem_device_api.h  shmem_heap.h  team_device.h  team.h  test_scalar_npu  types.h
        (base) [root@AscendLab-247 install]# ls lib
        libshmem.so  libtest_scalar_npu.so
        (base) [root@AscendLab-247 install]# ls lib64
        libtest_scalar_npu.so

   b. 用户从网页上获取所有SHMEM的压缩包 shmem.zip
      获取地址：待补充
    
   
4. 编码算子文件。 具体示例可以参考本仓 test/test_barrier/ 中的源码

5. 编译算子工程
   
   实例CMakeList.txt

   include_directories(
        ${PROJECT_SOURCE_DIR}/include/host/
        ${PROJECT_SOURCE_DIR}/include/device/
        ${PROJECT_SOURCE_DIR}/include/host_device/
        ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/host/
        ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/aicore/
    )
    
    file(GLOB_RECURSE KERNEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*_kernel.cpp")
    
    ascendc_library(test_scalar_npu SHARED ${KERNEL_FILES})
    ascendc_include_directories(test_scalar_npu
        PUBLIC
        ${PROJECT_SOURCE_DIR}/include/
        ${PROJECT_SOURCE_DIR}/include/host/
        ${PROJECT_SOURCE_DIR}/include/device/
        ${PROJECT_SOURCE_DIR}/include/host_device/
        ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/smem/host/
        ${PROJECT_SOURCE_DIR}/3rdparty/memfabric_hybrid/include/smem/device/
    )
    
    ascendc_compile_definitions(test_scalar_npu PRIVATE
        -DASCENDC_DUMP=1
    )
    
    install(TARGETS test_scalar_npu
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include
    )
    
    add_subdirectory(unittest)
    add_subdirectory(test_barrier)