# 快速开始

## 介绍
本系统主要面向昇腾平台上的模型和算子开发者，提供便携易用的多机多卡内存访问方式，方便用户开发在卡间同步数据，加速通信或通算融合类算子开发。  

## 软件架构
共享内存库接口主要分为host和device接口部分：
- host侧接口提供初始化、内存管理、通信域管理以及同步功能。
- device侧接口提供内存访问、同步以及通信域管理功能。

## 目录结构说明
详细介绍见[code_organization](code_organization.md)
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

## 执行样例 matmul_allreduce算子
1.在3rdparty目录下, clone AscendC Templates代码仓：

```sh
git clone https://gitee.com/ascend/catlass.git
```

2.在shmem/examples/matmul_allreduce目录下进行demo编译: 

```sh
bash build.sh
```

3.在shmem/examples/matmul_allreduce目录下生成golden数据：

```sh
python3 utils/gen_data.py 1 2 1024 1024 16 0 0
```

4.在shmem/examples/matmul_allreduce目录执行demo：

```sh
bash run.sh
```
5.在shmem/examples/matmul_allreduce目录验证算子精度：
    
```sh
python3 utils/verify_result.py ./out/output.bin ./out/golden.bin 1 1024 1024 16
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

## python侧test用例

1. 在scripts目录下编译的时候，带上build python的选项

```sh
bash build.sh -python_extension
```

2. 在install目录下，source环境变量

```sh
source set_env.sh
```

3. 在src/python目录下，进行setup，获取到wheel安装包

```sh
python3 setup.py bdist_wheel
```

4. 在src/python/dist目录下，安装wheel包

```sh
pip3 install shmem-xxx.whl --force-reinstall
```

5. 设置是否开启TLS认证，默认开启，若关闭TLS认证，请使用如下接口

```python
import shmem as shm
shm.set_conf_store_tls(False, "")   # 关闭tls认证
```

```python
import shmem as shm
tls_info = "xxx"
shm.set_conf_store_tls(True, tls_info)      # 开启TLS认证
```

6. 使用torchrun运行测试demo

```sh
torchrun --nproco-per-node=k test.py // k为想运行的ranksize
```
看到日志中打印出“test.py running success!”即为demo运行成功