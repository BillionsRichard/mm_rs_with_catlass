SHMEM
===

## 介绍
本系统主要面向昇腾平台上的模型和算子开发者，提供便携易用的多机多卡内存访问方式，方便用户开发在卡间同步数据，加速通信或通算融合类算子开发。  

## 软件架构
共享内存库接口主要分为host和device接口部分：
- host侧接口提供初始化、内存管理、通信域管理以及同步功能。
- device侧接口提供内存访问、同步以及通信域管理功能。

## 目录结构说明
详细介绍见[code_organization](docs/code_organization.md)
``` 
├── 3rdparty // 依赖的第三方库
├── docs     // 文档
├── examples // 使用样例
├── include  // 头文件
├── scripts  // 相关脚本
├── src      // 源代码
└── tests    // 测试用例
```

## 软件硬件配套说明
- 硬件型号支持 
  - Atlas 800I A2/A3 系列产品
  - Atlas 800T A2/A3 系列产品
- 平台：aarch64/x86
- 配套软件：驱动固件 Ascend HDK 25.0.RC1.1、 CANN 8.1.RC1及之后版本（参考《[CANN软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)》安装CANN开发套件包以及配套固件和驱动）  
cmake >= 3.19  
GLIBC >= 2.28

## 快速上手
详细资料请参考[SHMEM](https://shmem-doc.pages.dev/)
 - 设置CANN环境变量<br>
    ```sh
    # root用户安装（默认路径）
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```
 - 共享内存库编译<br>
    编译共享内存库，设置共享内存库环境变量：
    ```sh
    cd shmem
    bash scripts/build.sh
    source install/set_env.sh
    ```
 - run包使用<br>
    软件包名为：SHMEM_{version}_linux-{arch}.run <br>
    其中，{version}表示软件版本号，{arch}表示CPU架构。<br>
    安装run包（需要依赖cann环境）<br>

    ```sh
    chmod +x 软件包名.run # 增加对软件包的可执行权限
    ./软件包名.run --check # 校验软件包安装文件的一致性和完整性
    ./软件包名.run --install # 安装软件，可使用--help查询相关安装选项
    ```
    出现提示`xxx install success!`则安装成功

执行一个样例matmul_allreduce算子。  
1.在3rdparty目录下, clone catlass代码仓:

```sh
git clone https://gitee.com/ascend/catlass.git
```

2.在shmem/examples/matmul_allreduce目录下进行demo编译:

```sh
bash scripts/build.sh
```

3.在shmem/examples/matmul_allreduce目录执行demo:

```sh
bash scripts/run.sh -ranks 2 -M 1024 -K 2048 -N 8192
```

## 功能自测用例

 - 共享内存库接口单元测试
在工程目录下执行
```sh
bash scripts/build.sh -uttests
bash scripts/run.sh
```
run.sh脚本提供-ranks -ipport -test_filter等参数自定义执行用例的卡数、ip端口、gtest_filter等  

例

```sh
# 8卡，ip:port 127.0.0.1:8666，运行所有*Init*用例
bash scripts/run.sh -ranks 8 -ipport tcp://127.0.0.1:8666 -test_filter Init
```

## 安全声明
[安全声明](docs/security.md)

## 合作贡献者
待补充

## 版权声明
待补充

## 许可证
待补充