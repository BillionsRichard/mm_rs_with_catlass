#ifndef LAUNCH_MAP_H
#define LAUNCH_MAP_H

#include <unordered_map>

enum CocCommType {
    MATMUL_ALLREDUCE = 0,
    ALLGATHER_MATMUL,
    MATMUL_REDUCE_SCATTER,
    TYPE_NUM
};

enum CocDataType {
    FP16 = 1,
    BF16 = 27
};

using KernelFuncPtr = void (*)(void *, uint64_t, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
    CocTilingParams &, uint32_t, uint32_t);

class KernelDispatcher {
private:
    static std::unordered_map<int, KernelFuncPtr> &GetKernelMap()
    {
        static std::unordered_map<int, KernelFuncPtr> kernelMap;
        return kernelMap;
    }

    static int Hash(CocCommType commType, CocDataType dataType)
    {
        return (commType << 8) | dataType;
    }

public:
    static KernelFuncPtr GetKernelFunc(CocCommType commType, CocDataType dataType)
    {
        auto &kernelMap = GetKernelMap();
        if (auto it = kernelMap.find(Hash(commType, dataType)); it != kernelMap.end()) {
            return it->second;
        }
        return nullptr;
    }

    static void RegisterKernelFunc(CocCommType commType, CocDataType dataType, KernelFuncPtr func)
    {
        auto &kernelMap = GetKernelMap();
        kernelMap.insert({Hash(commType, dataType), func});
    }
};

#define REGISTER_KERNEL_FUNC(kernelName, commType, dataType)                                                           \
    void Launch##kernelName##dataType(void *, uint64_t, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,         \
        uint8_t *, CocTilingParams &, uint32_t, uint32_t);                                                             \
    namespace {                                                                                                        \
        struct AutoRegister##kernelName##dataType {                                                                    \
            AutoRegister##kernelName##dataType() {                                                                     \
                KernelDispatcher::RegisterKernelFunc(commType, dataType, &Launch##kernelName##dataType);               \
            }                                                                                                          \
        } s_autoRegister##kernelName##dataType;                                                                        \
    }

REGISTER_KERNEL_FUNC(MatmulAllReduce, MATMUL_ALLREDUCE, FP16);
REGISTER_KERNEL_FUNC(AllGatherMatmul, ALLGATHER_MATMUL, FP16);
REGISTER_KERNEL_FUNC(MatmulReduceScatter, MATMUL_REDUCE_SCATTER, FP16);

REGISTER_KERNEL_FUNC(MatmulAllReduce, MATMUL_ALLREDUCE, BF16);
REGISTER_KERNEL_FUNC(AllGatherMatmul, ALLGATHER_MATMUL, BF16);
REGISTER_KERNEL_FUNC(MatmulReduceScatter, MATMUL_REDUCE_SCATTER, BF16);

#undef REGISTER_KERNEL_FUNC

#endif // LAUNCH_MAP_H