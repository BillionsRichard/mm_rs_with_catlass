# SHMEM API

## Init API
Init对应SHMEM所需要的初始化接口

### `int shmem_init_status()`
查询当前初始化状态。  
```c++
enum {
    SHMEM_STATUS_NOT_INITALIZED = 0,    // 未初始化
    SHMEM_STATUS_SHM_CREATED,           // 完成共享内存堆创建 
    SHMEM_STATUS_IS_INITALIZED,         // 初始化完成 
    SHMEM_STATUS_INVALID = INT_MAX,
};
```

### `int shmem_set_attr(int myRank, int nRanks, uint64_t localMemSize, const char* ipPort, shmem_init_attr_t **attributes)`
设置初始化接口使用的默认ATTR。
 - myRank 当前rank
 - nRanks 总rank数
 - localMemSize 当前rank占用的内存大小
 - ipPort sever的ip和端口号，tcp:://ip:port 例如 tcp://127.0.0.1:8666 
 - attributes 出参，获取到全局attributes的指针
 - retuen 如果成功返回SHMEM_SUCCESS
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

### `int shmem_set_data_op_engine_type(shmem_init_attr_t *attributes, data_op_engine_type_t value)`
修改attr的data operation engine type
 - attributes 要修改可选参数的attr
 - value 要修改的值
 - retuen 如果成功返回SHMEM_SUCCESS

### `int shmem_set_timeout(shmem_init_attr_t *attributes, uint32_t value)`
修改attr的timeout。
 - attributes 要修改可选参数的attr
 - value 要修改的值
 - retuen 如果成功返回SHMEM_SUCCESS

### `int shmem_init()`
根据默认attr的初始化接口。
 - retuen 如果成功返回SHMEM_SUCCESS

### `int shmem_init_attr(shmem_init_attr_t *attributes)`
根据指定的attr初始化。
 - attributes 自行构造的shmem_init_attr_t结构体
 - retuen 如果成功返回SHMEM_SUCCESS

### `int shmem_finalize()`
去初始化接口。
 - retuen 如果成功返回SHMEM_SUCCESS

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
status = shmem_init();
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