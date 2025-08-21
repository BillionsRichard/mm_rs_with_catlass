# 安全加固

## 通信矩阵

|组件|tcp store|
|----------------|--------|
|源设备|tcp client|
|源IP|device IP|
|源端口|操作系统自动分配，分配范围由操作系统的自身配置决定|
|目的设备|tcp server|
|目的IP|设备地址IP|
|目的端口（侦听）|用户指定，端口号1025~65535|
|协议|TCP|
|端口说明|server与client TCP协议消息接口|
|侦听端口是否可更改|是|
|认证方式|支持，由初始化入参指定是否开启|
|加密方式|无|
|所属平面|管理面|
|版本|所有版本|
|特殊场景|无|

说明：
支持通过接口 `shmem_set_conf_store_tls` 配置TLS秘钥证书等，进行tls安全连接，建议用户开启TLS加密配置，保证通信通信安全。系统启动后，建议删除本地秘钥证书等信息敏感文件。调用该接口时，传入的文件路径不能包含英文分号、逗号、冒号。
支持通过环境变量 `ACCLINK_CHECK_PERIOD_HOURS`和`ACCLINK_CERT_CHECK_AHEAD_DAYS` 配置证书检查周期与证书过期预警时间

使用接口里子：
```c
// 配置关闭tls:
smem_set_conf_store_tls(false, nullptr, 0);

// 配置打开tls:

char *tls_info ="                               \
    tlsCaPath: /etc/ssl/certs/;                 \
    tlsCert: /etc/ssl/certs/server.crt;         \
    tlsPk: /etc/ssl/private/server.key;         \
    tlsPkPwd: /etc/ssl/private/key_pwd.txt;     \
    tlsCrlPath: /etc/ssl/crl/;                  \
    tlsCrlFile: server_crl1.pem,server_crl2.pem;\
    tlsCaFile: ca.pem1,ca.pem2;                 \
    packagePath: /etc/lib"
int32_t ret = smem_set_conf_store_tls(true, tls_info, strlen(tls_info));
```
| 环境变量 | 说明                                         |
|------|--------------------------------------------|
| SHMEM_MASTER_ADDR | 通信面IP |
| SHMEM_MASTER_PORT | 通信面端口 |
| MASTER_ADDR | 备用通信面IP |
| MASTER_PORT | 备用通信面端口 |
| SHMEM_HOME_PATH | shmem安装路径 |
| VERSION | 编译whl包默认版本号 |


## 加固须知

本文中列出的安全加固措施为基本的加固建议项。用户应根据自身业务，重新审视整个系统的网络安全加固措施。用户应按照所在组织的安全策略进行相关配置，包括并不局限于软件版本、口令复杂度要求、安全配置（协议、加密套件、密钥长度等）， 权限配置、防火墙设置等。必要时可参考业界优秀加固方案和安全专家的建议。

## 运行用户建议

root是Linux系统中的超级特权用户，具有所有Linux系统资源的访问权限。如果允许直接使用root账号登录Linux系统对系统进行操作，会带来很多潜在的安全风险。通常情况下，建议将在“/etc/ssh/sshd_config”文件中将“PermitRootLogin”参数设置为“no”，设置后root用户无法通过SSH登录到系统，增加了系统的安全性。如果需要使用root权限进行管理操作，可以通过其他普通用户登录系统后，再使用su或sudo命令切换到root用户进行操作。这样可以避免直接使用root用户登录系统，从而减少系统被攻击的风险。出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用shmem。

## 内存地址随机化机制安全加固

ASLR（address space layout randomization）开启后能增强漏洞攻击防护能力，建议用户将/proc/sys/kernel/randomize_va_space里面的值设置为2，开启该功能。


# 公网地址

| 类型   | 开源代码地址 | 文件名 | 公网IP地址/公网URL地址/域名/邮箱地址 | 用途说明 |
|------|-----------------|-------------|---------------------|-------------------|
| 自研   | 不涉及 | README.md/quickstart.md                   | https://gitee.com/ascend/catlass.git | 依赖的开源代码仓            |
| 自研   | 不涉及 | build.sh                    | https://gitee.com/ascend/memfabric_hybrid.git | 依赖的开源代码仓            |
| 文档   | 不涉及 | README.md                    | https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit | CANN商发文档            |
| 文档   | 不涉及 | shmemi_device_barrier.h                   | https://www.inf.ed.ac.uk/teaching/courses/ppls/BarrierPaper.pdf | 并行编程语言和系统            |
| 开源引入   | https://github.com/google/googletest | build.sh                   | https://github.com/google/googletest.git | gtest测试框架            |
| 开源引入   | https://github.com/doxygen | build.sh                   | https://github.com/doxygen/doxygen/releases/download/Release_1_9_3/doxygen-1.9.3.src.tar.gz | doxygen文档生成工具            |
