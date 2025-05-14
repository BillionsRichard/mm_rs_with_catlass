# SHMEM API
SHMEM包含host和device两类接口。host接口用SHMEM_HOST_API宏标识，device接口用SHMEM_DEVICE宏标识。

## Init API
SHMEM的初始化接口样例

### 初始化状态  
```c++
enum {
    SHMEM_STATUS_NOT_INITALIZED = 0,    // 未初始化
    SHMEM_STATUS_SHM_CREATED,           // 完成共享内存堆创建 
    SHMEM_STATUS_IS_INITALIZED,         // 初始化完成 
    SHMEM_STATUS_INVALID = INT_MAX,
};
```

### 初始化所需的attributes 
```c++
// 初始化属性
typedef struct {
    int version;                            // 版本
    int myRank;                             // 当前rank
    int nRanks;                             // 总rank数
    const char* ipPort;                     // ip端口
    uint64_t localMemSize;                  // 本地申请内存大小
    shmem_init_optional_attr_t optionAttr;  // 可选参数
} shmem_init_attr_t;

// 可选属性
typedef struct {
    data_op_engine_type_t dataOpEngineType; // 数据引擎
    // timeout
    uint32_t shmInitTimeout;
    uint32_t shmCreateTimeout;
    uint32_t controlOperationTimeout;
} shmem_init_optional_attr_t;
```

### 初始化样例
```c++
#include <iostream>
#include <unistd.h>
#include <acl/acl.h>
#include "shmem_api.h"
aclInit(nullptr);
status = aclrtSetDevice(deviceId);

shmem_init_attr_t* attributes;
shmem_set_attr(rankId, nRanks, localMemSize, testGlobalIpport, &attributes);
// shmem_init_attr_t* attributes = new shmem_init_attr_t{0, rankId, nRanks, testGlobalIpport, localMemSize, {SHMEM_DATA_OP_MTE, 120, 120, 120}}; // 自定义attr
shmem_init_attr(attributes);
// delete attributes; // 销毁自定义attr

status = shmem_init_status();
if (status == SHMEM_STATUS_IS_INITALIZED) {
    std::cout << "Init success!" << std::endl;
}
//################你的任务#################

//#########################################
status = shmem_finalize();
aclrtResetDevice(deviceId);
aclFinalize();

```
样例见[helloworld](../examples/helloworld)

目录下执行脚本即可运行

```sh
bash build.sh
```

## Team API
SHMEM的通信域管理接口样例

## Mem API
SHMEM的内存管理接口样例

## RMA API
SHMEM的远端内存访问接口样例

## Sync API
SHMEM的同步管理接口样例
