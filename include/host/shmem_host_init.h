#ifndef SHMEM_HOST_INIT_H
#define SHMEM_HOST_INIT_H

#include "host_device/shmem_types.h"

/**
 * @brief 查询当前初始化状态
 * @param 
 * @return 返回初始化的状态，SHMEM_STATUS_IS_INITALIZED为初始化完成
 */
int ShmemInitStatus();

/**
 * @brief 设置初始化接口使用的默认ATTR
 * @param myRank 当前rank
 * @param nRanks 总rank数
 * @param localMemSize 当前rank占用的内存大小
 * @param ipPort sever的ip和端口号，例如 tcp:://ip:port
 * @param attributes 出参，获取到全局attributes的指针
 * @return 如果成功返回SHMEM_SUCCESS
 */
int ShmemSetAttr(int myRank, int nRanks, uint64_t localMemSize, const char* ipPort, ShmemInitAttrT **attributes);

/**
 * @brief 修改attr的data operation engine type
 * @param attributes 要修改可选参数的attr
 * @param value 要修改的值
 * @return 如果成功返回SHMEM_SUCCESS
 */
int SetDataOpEngineType(ShmemInitAttrT *attributes, DataOpEngineType value);

/**
 * @brief 修改attr的timeout
 * @param attributes 要修改可选参数的attr
 * @param value 要修改的值
 * @return 如果成功返回SHMEM_SUCCESS
 */
int SetTimeout(ShmemInitAttrT *attributes, uint32_t value);

/**
 * @brief 根据默认attr的初始化接口
 * @param 
 * @return 如果成功返回SHMEM_SUCCESS
 */
int ShmemInit();

/**
 * @brief 根据指定的attr初始化
 * @param attributes 指定的attr
 * @return 如果成功返回SHMEM_SUCCESS
 */
int ShmemInitAttr(ShmemInitAttrT *attributes);
/**
 * @brief 去初始化接口
 * @param 
 * @return 如果成功返回SHMEM_SUCCESS
 */
int ShmemFinalize();

#endif