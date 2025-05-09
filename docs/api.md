# SHMEM API
SHMEM包含host和device两类接口。host接口用SHMEM_HOST_API宏标识，device接口用SHMEM_DEVICE宏标识。
## Init API
SHMEM的初始化接口

### `SHMEM_HOST_API int shmem_init_status()`
查询当前初始化状态。  
```c++
enum {
    SHMEM_STATUS_NOT_INITALIZED = 0,    // 未初始化
    SHMEM_STATUS_SHM_CREATED,           // 完成共享内存堆创建 
    SHMEM_STATUS_IS_INITALIZED,         // 初始化完成 
    SHMEM_STATUS_INVALID = INT_MAX,
};
```

### `SHMEM_HOST_API int shmem_set_attr(int myRank, int nRanks, uint64_t localMemSize, const char* ipPort, shmem_init_attr_t **attributes)`
设置初始化接口使用的默认ATTR。
 - myRank 当前rank
 - nRanks 总rank数
 - localMemSize 当前rank占用的内存大小
 - ipPort sever的ip和端口号，tcp://ip:port 例如 tcp://127.0.0.1:8666 
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

### `SHMEM_HOST_API int shmem_set_data_op_engine_type(shmem_init_attr_t *attributes, data_op_engine_type_t value)`
修改attr的data operation engine type
 - attributes 要修改可选参数的attr
 - value 要修改的值
 - retuen 如果成功返回SHMEM_SUCCESS

### `SHMEM_HOST_API int shmem_set_timeout(shmem_init_attr_t *attributes, uint32_t value)`
修改attr的timeout。
 - attributes 要修改可选参数的attr
 - value 要修改的值
 - retuen 如果成功返回SHMEM_SUCCESS

### `SHMEM_HOST_API int shmem_init()`
根据默认attr的初始化接口。
 - retuen 如果成功返回SHMEM_SUCCESS

### `SHMEM_HOST_API int shmem_init_attr(shmem_init_attr_t *attributes)`
根据指定的attr初始化。
 - attributes 自行构造的shmem_init_attr_t结构体
 - retuen 如果成功返回SHMEM_SUCCESS

### `SHMEM_HOST_API int shmem_finalize()`
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

## Team API
SHMEM的通信域管理接口

### `SHMEM_HOST_API int shmem_team_split_strided(shmem_team_t parentTeam, int peStart, int peStride, int peSize, shmem_team_t &newTeam)`
team的划分和创建接口。
 - parentTeam 父通信域
 - peStart 新通信域初始pe
 - peStride pe间隔
 - peSize pe数量
 - newTeam 新通信域
 - retuen 如果成功返回SHMEM_SUCCESS

### `SHMEM_HOST_API int shmem_team_translate_pe(shmem_team_t srcTeam, int srcPe, shmem_team_t destTeam)`
获取srcPe在目标team的编号。
 - srcTeam 
 - srcPe 
 - destTeam 
 - retuen 

### `SHMEM_HOST_API void shmem_team_destroy(shmem_team_t team)`
team销毁。
 - team 销毁的team

### `SHMEM_HOST_API int shmem_my_pe()`
获取当前pe。
 - retuen 当前pe

### `SHMEM_HOST_API int shmem_n_pes()`
获取pe总数。
 - retuen pe总数

### `SHMEM_HOST_API int shmem_team_my_pe(shmem_team_t team)`
获取当前pe在目标team内的编号。
 - team 目标team
 - retuen pe编号

### `SHMEM_HOST_API int shmem_team_n_pes(shmem_team_t team)`
获取目标team内的pe总数。
 - team 目标team
 - retuen pe总数

### `SHMEM_DEVICE int shmem_my_pe(void)`
获取当前pe。
 - retuen 当前pe

### `SHMEM_DEVICE int shmem_n_pes(void)`
获取pe总数。
 - retuen pe总数

### `SHMEM_DEVICE int shmem_team_my_pe(shmem_team_t team)`
获取当前pe在目标team内的编号。
 - team 目标team
 - retuen pe编号

### `SHMEM_DEVICE int shmem_team_n_pes(shmem_team_t team)`
获取目标team内的pe总数。
 - team 目标team
 - retuen pe总数

### `SHMEM_DEVICE int shmem_team_translate_pe(shmem_team_t srcTeam, int srcPe, shmem_team_t destTeam)`
获取srcPe在目标team的编号。
 - srcTeam 
 - srcPe 
 - destTeam 
 - retuen 

## Mem API
SHMEM的内存管理接口

### `SHMEM_HOST_API void *shmem_malloc(size_t size)`
分配size bytes内存并返回指向分配内存的指针。
 - size 需要分配的内存大小
 - retuen 指向分配内存的指针

### `SHMEM_HOST_API void *shmem_calloc(size_t nmemb, size_t size)`
为nmemb个元素每个分配size bytes内存，返回指向分配内存的指针。
 - nmemb 元素数量
 - size 每个元素需要分配的内存大小
 - retuen 指向分配内存的指针

### `SHMEM_HOST_API void *shmem_align(size_t alignment, size_t size)`
分配size大小内存并对齐地址为2的幂，返回指向分配内存的指针
 - alignment 内存地址对齐方式
 - size 每个元素需要分配的内存大小
 - retuen 指向分配内存的指针

### `SHMEM_HOST_API void shmem_free(void *ptr)`
释放分配的内存。
 - ptr 内存块的指针

## Rma API
SHMEM的远端内存访问接口

### `SHMEM_HOST_API void* shmem_ptr(void *ptr, int pe)`
计算ptr地址在目标pe上的对称地址。
 - ptr
 - pe

### `SHMEM_HOST_API int shmem_mte_set_ub_params(uint64_t offset, uint32_t ubSize, uint32_t eventID)`
mte ub参数设置。
 - offset 
 - ubSize 
 - eventID 

### `SHMEM_DEVICE __gm__ void* shmem_ptr(__gm__ void* ptr, int pe)`
计算ptr地址在目标pe上的对称地址。
 - ptr
 - pe

### 待补充

## Sync API
SHMEM的同步管理接口

### `SHMEM_HOST_API void shmem_barrier_on_stream(shmem_team_t tid, aclrtStream stream)`
stream team 同步。
 - tid
 - stream

### `SHMEM_HOST_API void shmem_barrier_all_on_stream(aclrtStream stream)`
stream同步。
 - stream

### `SHMEM_HOST_API void shmem_barrier(shmem_team_t tid)`
team同步
 - tid

### `SHMEM_HOST_API void shmem_barrier_all()`
全局同步。

### `SHMEM_DEVICE void shmem_barrier(shmem_team_t tid)`
team同步
 - tid

### `SHMEM_DEVICE void shmem_barrier_all()`
全局同步。

### `SHMEM_DEVICE void shmem_quiet()`
等待内存访问完成

### `SHMEM_DEVICE void shmem_fence()`
等待内存访问完成