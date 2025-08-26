# AllGather矩阵乘法反量化算子设计文档

## 1. 算子概述

### 1.1 功能描述
AllGather矩阵乘法反量化算子 (AllGatherDequantMatmul) 是一个分布式矩阵乘法算子，其核心流程是先对激活矩阵A进行AllGather通信，再执行矩阵乘法和反量化。

该算子支持对输入进行量化，其中：
- **激活矩阵A**: 支持 **per-tensor** 量化（整个矩阵使用一个缩放因子）。
- **权重矩阵B**: 支持 **per-channel** 量化（矩阵的每一列使用一个独立的缩放因子）。

在调用算子前，Host侧需要将A的per-tensor scale和B的per-channel scale预先计算并融合成一个**fused_scale**向量。Kernel只需要接收这个融合后的scale向量即可完成反量化。

### 1.2 算子签名
```cpp
void AllGatherDequantMatmul(
    uint64_t fftsAddr,
    GM_ADDR aDevice,           // 输入矩阵A: [M, K], int8
    GM_ADDR bDevice,           // 输入矩阵B: [K, N], int8  
    GM_ADDR cDevice,           // 中间结果矩阵: [M*rankSize, N], int32
    GM_ADDR symmetricPtr,      // 用于Rank间通信的共享内存工作空间 (workspace)
    GM_ADDR dDevice,           // 输出矩阵: [M*rankSize, N], fp16
    GM_ADDR fused_scale,       // 融合后的量化缩放因子: [N], float32
    uint32_t m, 
    uint32_t n, 
    uint32_t k
);
```

### 1.3 输入输出规格
| 参数 | 形状 | 数据类型 | 描述 |
|------|------|----------|------|
| aDevice | [M, K] | int8 | 量化后的输入矩阵A (每个Rank一份) |
| bDevice | [K, N] | int8 | 量化后的输入矩阵B (所有Rank共享) |
| cDevice | [M*rankSize, N] | int32 | Matmul的int32输出 / Dequantize的输入 |
| dDevice | [M*rankSize, N] | fp16 | 最终输出矩阵 |
| fused_scale | [N] | float32 | 由A的per-tensor scale和B的per-channel scale在Host侧预先融合计算得出的scale向量 |
| symmetricPtr | - | GM_ADDR | 用于AllGather通信的共享内存工作区 |

## 2. 量化算法设计

### 2.1 核心计算流程
```mermaid
graph TD
    subgraph "Input (Per Rank)"
        A_i[int8 A_i]
        B[int8 B]
    end

    subgraph "Host Side"
        a_scale[A's per-tensor scale]
        b_scale[B's per-channel scale]
        a_scale --> Fuse{Fuse Scales}
        b_scale --> Fuse
        Fuse --> fused_scale[Fused Scale Vector]
    end

    subgraph "Step 1: AllGather on A (AIV)"
        A_i -->|All Ranks| Comm{AllGather}
        Comm --> A_full[Complete A_full]
    end

    subgraph "Step 2: Matmul (AIC)"
        A_full --> Matmul{INT8 Matmul}
        B --> Matmul
        Matmul --> C_int32[int32 Result]
    end

    subgraph "Step 3: Dequantize (AIV)"
        C_int32 --> Dequant{Dequantize}
        fused_scale --> Dequant
        Dequant --> D_fp16[fp16 Output]
    end
```

### 2.2 量化与反量化公式
```c++
    // --- Host Side ---
    // 伪代码: 量化
    a_int8 = round(a_fp32 / a_scale_scalar)
    b_int8 = round(b_fp32 / b_scale_vector)

    // 伪代码: 融合Scale
    fused_scale_vector = a_scale_scalar * b_scale_vector

    // --- Kernel Side ---
    // 伪代码：1. AllGather
    complete_a_int8 = allgather(a_int8_per_rank)

    // 伪代码: 2. 矩阵乘法
    result_int32 = matmul(complete_a_int8, b_int8)

    // 伪代码: 3. 反量化
    // fused_scale_vector被广播到C的每一行
    result_fp32 = result_int32 * fused_scale_vector
    output_fp16 = cast_to_fp16(result_fp32)
```

## 3. 核心实现架构

### 3.1 计算与通信分离
算子采用计算（AIC）和通信/后处理（AIV）分离的设计，实现了任务的流水线执行。
- **AIV (AI Vector Core)**: 负责两个阶段：
  1. **通信:** 执行 `AllGather` 操作，将所有Rank的 `INT8` 矩阵A收集到共享工作区。
  2. **后处理:** 在AIC计算完成后，对 `INT32` 结果执行反量化操作，得到最终的 `FP16` 输出。
- **AIC (AI Core)**: 负责核心的计算任务，即执行高密度的 `INT8 × INT8 → INT32` 矩阵乘法。它从共享工作区读取由AIV准备好的完整矩阵A进行计算。

### 3.2 主要模块
- **BlockMmad**: `catlass`库提供的矩阵乘法模块，通过`MmadAtlasA2Pingpong`调度策略，执行分块的INT8矩阵乘法。
- **CommBlockEpilogue**: `catcoc`库提供的通信Epilogue，用于执行 `AllGather` 操作。它将所有Rank的INT8矩阵A收集到每个Rank。
- **BlockEpilogueDequant**: `catlass`库提供的后处理Epilogue，用于在 `AllGather` 完成后，对INT32结果进行反量化，并转换为 `FP16`。

## 4. 内存布局设计

### 4.1 全局内存 (Global Memory)
- **输入布局**: `aDevice`, `bDevice`, `fused_scale` 均存储在GM中。
- **中间结果布局**: 每个Rank计算出的 `INT32` 累加器结果存储在各自的GM空间中，形状为 `[M*rankSize, N]`。
- **输出布局**: 最终的 `FP16` 输出也存储在GM中，形状为 `[M*rankSize, N]`。

### 4.2 共享内存 (Symmetric Memory)
- **用途**: `symmetricPtr` 指向的共享内存区域被用作 `AllGather` 操作的**临时工作空间（Workspace）**。
- **工作方式**: 在 `AllGather` 过程中，每个Rank需要将自己的INT8矩阵A写入到共享内存区域，然后所有Rank从这个区域读取完整的矩阵A。
- **布局**:
  ```cpp
  Catlass::layout::RowMajor layoutSymmetric{
      WORKSPACE_STAGES * rankSize * commSizeM,
      K,
      K
  };
  ```

## 5. 通信模式适配

### 5.1 AIV/AIC流水线工作模式
算子的核心是AIV和AIC之间的流水线（Pipeline）作业，实现了通信和计算的重叠。
1.  **AIV (通信先行):** AIV核首先启动，执行 `AllGather` 操作，将第一个数据块从所有Rank收集到共享工作区（Symmetric Workspace）的stage 0缓冲区。
2.  **AIV通知AIC:** 当stage 0的数据准备好后，AIV通过flag机制通知AIC可以开始计算。同时，AIV可以开始将下一个数据块收集到stage 1缓冲区。
3.  **AIC (计算):** AIC核被唤醒，从共享工作区的stage 0缓冲区读取完整的矩阵`A`数据块，并与本地矩阵`B`进行矩阵乘法。
4.  **循环执行:** AIC在计算stage 0时，AIV在准备stage 1的数据。这个过程循环往复，直到所有数据块处理完毕。
5.  **AIV (最终处理):** 在所有计算完成后，AIV负责对AIC产生的`INT32`结果进行最后的反量化。

### 5.2 流程图
```mermaid
sequenceDiagram
    participant AIV
    participant AIC
    participant Symmetric Memory
    
    AIV->>Symmetric Memory: AllGather(A_i) into stage 0
    AIV->>AIC: Signal (flag) data ready
    
    par
        AIC->>Symmetric Memory: Read complete A from stage 0
        AIC->>AIC: Matmul(A_complete, B) -> C_int32
    and
        AIV->>Symmetric Memory: AllGather(A_{i+1}) into stage 1
    end
    
    Note over AIV, AIC: Loop until all blocks are processed

    AIC-->>AIV: Signal Matmul complete
    AIV->>AIV: Dequantize(C_int32) -> D_fp16
```

## 6. 工作空间管理

### 6.1 多阶段流水线
算子使用 `WORKSPACE_STAGES = 2` 的多阶段流水线设计，通过 `commInterval = 3` 控制通信间隔，实现计算与通信的重叠。

### 6.2 内存复用策略
- **共享内存复用**: 使用共享内存作为AllGather操作的临时缓冲区，避免重复分配。
- **流水线缓冲**: 通过多阶段设计，实现计算与通信的并行执行。

## 7. 总结

该量化算子通过将高成本的通信操作（AllGather）在 `INT8` 数据上完成，避免了在 `FP16` 或 `FP32` 上进行通信，从而优化了性能。同时，通过精确的量化参数处理，确保了在多Rank环境下的计算精度。共享内存（Symmetric Memory）在此过程中扮演了关键的临时数据交换区的角色。

## 8. 使用指南

### 8.1 编译

```bash
# 切换到项目根目录
cd /path/to/project/root

# 运行总编译脚本，携带-examples选项
bash scripts/build.sh -examples
```

### 8.2 运行

```bash
# 在2个设备上运行（设备0和1）
bash examples/allgather_matmul_dequant/scripts/run.sh 0,1

# 在4个设备上运行（设备1, 3, 5, 7）
bash examples/allgather_matmul_dequant/scripts/run.sh 1,3,5,7
```

### 8.3 测试形状

测试形状定义在 `scripts/test_shapes.csv` 中：
```
M,K,N
16384,27392,4096
131072,8192,3072
64,16384,7168
```

### 8.4 数据文件

在 `output/` 目录中生成以下数据文件：
- `a_gm_rank_{i}.bin`: 每个Rank的量化输入矩阵A (int8)
- `b_gm.bin`: 量化输入矩阵B (int8)
- `scale_gm.bin`: 融合后的量化缩放因子 (float32)
- `golden.bin`: 验证用的期望输出 (fp16)
- `output.bin`: 算子的实际输出 (fp16)
- `c_gm.bin`, `d_gm.bin`: 算子执行所需的临时空矩阵

### 8.5 验证

脚本会自动验证输出结果与黄金参考的差异，使用现有的 `verify_result.py` 进行精度验证。该验证脚本支持 fp16 数据类型的精度检查。

### 8.6 调试模式

设置环境变量启用调试模式：
```bash
export debug=1
bash run.sh 0,1
```

调试模式下会使用全1矩阵和固定缩放因子，便于问题排查。

### 8.7 环境要求

- Ascend Toolkit 已正确安装
- SHMEM 环境已配置
- PyTorch 支持（用于数据生成和验证）
- 支持 fp16 数据类型的硬件环境

## 9. 与Reduce-Scatter算子的对比

| 特性 | AllGather算子 | Reduce-Scatter算子 |
|------|---------------|-------------------|
| 通信模式 | AllGather | Reduce-Scatter |
| 通信数据类型 | INT8 | INT32 |
| 输出形状 | [M*rankSize, N] | [M/rankSize, N] |
| 偏置支持 | 否 | 是 |
| 适用场景 | 需要完整矩阵A的场景 | 需要分片结果的场景 |
| 内存占用 | 较高（需要完整矩阵） | 较低（分片存储） |
