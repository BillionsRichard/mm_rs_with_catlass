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
   编译算子代码时， 需要使用到之前SHMEM的编译结果。 建议从按照3.b 指示的路径获取标准包。并将其解压
   a. 获取SHMEM的标准包 shmem.zip
   b. 将上述获取到的shmem.zip 解压到待编译算子工程的某个目录， 下面以 3rdparty为例
       cd 3rdparty
       unzip shmem.zip 
   c. 编写CMakeList.txt, 使 include路径包含SHMEM的头文件。
       参考 test/test_barrier/CMakeList.txt
   注意： 这里可以选择静态链接方式将shmem.a编译到算子so文件中， 也可以选在动态链接，在后续加载时链接到shmem.so
6. 加载
   如果动态方式， 则还需要将shmem加载到执行机上。