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
team的划分和创建接口。从现有的父通信域中创建一个新的通信域。新的通信域由`(start,stride,size)`三元组定义
 - parentTeam 父通信域
 - peStart 父通信域组成新通信域PE子集中的最低PE编号
 - peStride 组成新通信域时父通信域间PE的跨度
 - peSize 组成新通信域中来自父通信域的PE数量
 - newTeam 新通信域
 - retuen 如果成功返回SHMEM_SUCCESS

### `SHMEM_HOST_API int shmem_team_translate_pe(shmem_team_t srcTeam, int srcPe, shmem_team_t destTeam)`
### `SHMEM_DEVICE int shmem_team_translate_pe(shmem_team_t srcTeam, int srcPe, shmem_team_t destTeam)`
将srcTeam中的PE编号转换成destTeam中相应的PE编号。
 - srcTeam 源team
 - srcPe 源PE
 - destTeam 目标team
 - retuen destTeam的PE编号

### `SHMEM_HOST_API int shmem_my_pe()`
### `SHMEM_DEVICE int shmem_my_pe(void)`
获取当前pe。
 - retuen 当前pe

### `SHMEM_HOST_API int shmem_n_pes()`
### `SHMEM_DEVICE int shmem_n_pes(void)`
获取pe总数。
 - retuen pe总数

### `SHMEM_HOST_API int shmem_team_my_pe(shmem_team_t team)`
### `SHMEM_DEVICE int shmem_team_my_pe(shmem_team_t team)`
获取当前pe在目标team内的编号。
 - team 目标team
 - retuen pe编号

### `SHMEM_HOST_API int shmem_team_n_pes(shmem_team_t team)`
### `SHMEM_DEVICE int shmem_team_n_pes(shmem_team_t team)`
获取目标team内的pe总数。
 - team 目标team
 - retuen pe总数

### `SHMEM_HOST_API void shmem_team_destroy(shmem_team_t team)`
team销毁。
 - team 销毁的team

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

### `SHMEM_HOST_API int shmem_mte_set_ub_params(uint64_t offset, uint32_t ubSize, uint32_t eventID)`
mte ub参数设置。
 - offset 偏移量
 - ubSize UB大小
 - eventID event id 

### `SHMEM_HOST_API void* shmem_ptr(void *ptr, int pe)`
### `SHMEM_DEVICE __gm__ void* shmem_ptr(__gm__ void* ptr, int pe)`
计算ptr地址在目标pe上的对称地址。
 - ptr 要引用的可远程访问数据对象的对称地址
 - pe 要访问ptr的PE编号
 - return 返回指向该对象的本地指针，不能远程访问时返回空指针

### `SHMEM_DEVICE void shmem_##NAME##_p(__gm__ TYPE* dst, const TYPE value, int pe)`
TYPE是标准RMA类型之一，有由表指定指定的对应的NAME。
为基本类型的单个元素提供了低延迟的put功能。
 - dst 目的数据对象的对称地址
 - value 要传递给dst的值
 - pe 远端PE编号

### `SHMEM_DEVICE TYPE shmem_##NAME##_g(__gm__ TYPE* src, int32_t pe)`
TYPE是标准RMA类型之一，有由表指定指定的对应的NAME。
为基本类型的单个元素提供了低延迟的get功能。
 - src 源数据对象的对称地址
 - pe 源所在对端PE编号
 - return 返回指定类型的单个元素

### `SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, uint32_t elemSize, int32_t pe)`
### `SHMEM_DEVICE void shmem_put_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, uint32_t elemSize, int pe)`
TYPE是标准RMA类型之一，有由表指定指定的对应的NAME。
将src上的数据拷贝到dst的数组中。启动操作后返回。后续调用`shmem_quiet()`后认为操作完成。
 - dst 目的数据对象的对称地址
 - src 源数据对象的对称地址
 - elemSize dst和src数组中的元素个数
 - pe 远端PE编号

### `SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(__gm__ TYPE* dst, __gm__ TYPE* src, uint32_t elemSize, int32_t pe)`
### `SHMEM_DEVICE void shmem_get_##NAME##_mem_nbi(AscendC::GlobalTensor<TYPE> dst, AscendC::GlobalTensor<TYPE> src, uint32_t elemSize, int pe)`
TYPE是标准RMA类型之一，有由表指定指定的对应的NAME。
将相邻对称数据从不同PE复制到本地PE上。启动操作后返回。后续调用`shmem_quiet()`后认为操作完成。
 - dst 目的数据对象的对称地址
 - src 源数据对象的对称地址
 - elemSize dst和src数组中的元素个数
 - pe 远端PE编号


## Sync API
SHMEM的同步管理接口

### `SHMEM_HOST_API void shmem_barrier_on_stream(shmem_team_t tid, aclrtStream stream)`
stream 上同步team 上所有的PE。
 - tid 需要同步的team对应的id
 - stream 对应的stream

### `SHMEM_HOST_API void shmem_barrier_all_on_stream(aclrtStream stream)`
stream上同步所有PE。
 - stream 对应的stream。

### `SHMEM_HOST_API void shmem_barrier(shmem_team_t tid)`
### `SHMEM_DEVICE void shmem_barrier(shmem_team_t tid)`
team的所有PE同步。当team的所有PE都调用shmem_barrier后返回。
 - tid 需要同步的team对应的id

### `SHMEM_HOST_API void shmem_barrier_all()`
### `SHMEM_DEVICE void shmem_barrier_all()`
同步所有PE。阻塞调用的PE直到所有PE都调用shmem_barrier_all。

### `SHMEM_DEVICE void shmem_quiet()`
确保PE上的操作执行完成

### `SHMEM_DEVICE void shmem_fence()`
确保PE上操作的传递顺序