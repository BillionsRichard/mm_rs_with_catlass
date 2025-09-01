# Allgather + Matmul + Alltoall 融合算子设计文档

## 1. 算子功能描述

本算子旨在将 `Allgather`、`Matmul` 和 `Alltoall` 三个操作融合在单个核函数中，以减少核函数启动开销和中间数据传输，从而优化大模型推理性能。

算子的主要应用场景是分布式矩阵乘法，其中输入矩阵（激活值）首先需要在多个计算设备间进行汇集，然后与本地持有的权重矩阵分片进行矩阵乘法，最后将结果通过 `Alltoall` 操作分发回各个设备。

## 2. 设计细节

### 2.1. 通信模式

- **Allgather**: 在 `[rankSize]` 个设备间对输入张量进行 `Allgather` 操作。
- **Alltoall**: 在相同的 `[rankSize]` 个设备间对矩阵乘法后的结果进行 `Alltoall` 操作。
- 两个通信操作使用相同的HCCL通信域。

### 2.2. 数据流与Shape变换

假设 `rankSize` 是通信组中的设备数量，`M, K, N` 是逻辑上的矩阵维度。

1.  **输入 (Input)**:
    *   每个 rank `i` 拥有**独特**的输入激活张量 `A_i`，Shape 为 `[M, K]`。
    *   每个 rank `i` 持有**独特**的部分权重张量 `B_i`，Shape 为 `[K, N/rankSize]`。

2.  **Allgather**:
    *   对所有 ranks 的输入张量 `A_i` 进行 `Allgather` 操作。
    *   数据 Shape 变换: `[M, K]` (per rank) -> `[rankSize, M, K]` (on each rank)。
    *   `Allgather` 的结果是所有 `A_i` 的集合，存储在每个 rank 的本地内存中。

3.  **Matmul (Batched GEMM)**:
    *   在每个 rank 上，执行批处理矩阵乘法。
    *   运算描述: `[rankSize, M, K] @ [K, N/rankSize]`。
        *   这里需要注意的是，权重矩阵 `B` (`[K, N/rankSize]`) 会被广播（broadcast）以匹配批处理维度 `rankSize`。
    *   输出 Shape: `[rankSize, M, N/rankSize]`。
    *   本次设计不包含偏置（bias）和量化（dequantization）功能。

4.  **转置 (Transpose)**:
    *   对 Matmul 的结果进行转置，交换最后两个维度。
    *   数据 Shape 变换: `[rankSize, M, N/rankSize]` -> `[rankSize, N/rankSize, M]`。
    *   这一步是为了让数据布局满足 `Alltoall` 的要求。

5.  **Alltoall**:
    *   对转置后的张量进行 `Alltoall` 操作。
    *   每个 rank 将 `[rankSize, N/rankSize, M]` 的数据沿着第一个轴（`rankSize` 轴）切分成 `rankSize` 块，每块的 Shape 为 `[N/rankSize, M]`。
    *   第 `i` 个 rank 将第 `j` 块数据发送给第 `j` 个 rank。
    *   数据 Shape 变换: `[rankSize, N/rankSize, M]` -> `[rankSize, N/rankSize, M]`。
        *   虽然 Shape 保持不变，但张量内部的数据已经根据 rank 进行了重新分布。

6.  **视图变换与转置 (View & Transpose)**:
    *   `Alltoall` 的输出可以被重新解释（view）为一个更大的张量。
    *   数据 Shape 变换: `[rankSize, N/rankSize, M]` -> `[N, M]`。
    *   最后，为了得到最终的输出形式，再进行一次转置。
    *   数据 Shape 变换: `[N, M]` -> `[M, N]`。

7.  **最终输出 (Final Output)**:
    *   每个 rank 得到完整的最终结果矩阵的一部分。
    *   最终输出张量 `C` 的 Shape 为 `[M, N]`。

### 2.3. Host侧接口设计

```cpp
void allgather_matmul_alltoall(
    const half* A,         // 输入矩阵A, shape [M, K]
    const half* B,         // 输入矩阵B, shape [K, N/rank_size]
    half* C,               // 输出矩阵C, shape [M, N]
    int M,
    int K,
    int N,
    int rank,
    int rank_size,
    aclrtStream stream
);
```

### 2.4. Device侧核函数实现要点

- **内存管理**: 需要精确计算 `Allgather` 和 `Alltoall` 操作所需的共享内存（Shared Memory）或临时全局内存（Global Memory）大小。
- **批处理GEMM**: 利用 `cublas` 或自定义的 `GEMM` kernel 实现批处理矩阵乘法。
- **数据重排布**: `Transpose` 和 `View` 操作需要在核函数内部通过高效的内存拷贝和索引计算来实现。
- **同步**: 在通信和计算步骤之间需要适当的同步（e.g., `__syncthreads()`）来保证数据依赖的正确性。

## 3. 验证方案

- **Host侧验证**:
    1.  **数据生成**:
        *   在每个 rank `i` 上，生成其独特的输入激活 `A_i` (shape `[M, K]`) 和权重 `B_i` (shape `[K, N/rankSize]`)。
    2.  **Golden结果计算 (在Rank 0上集中计算)**:
        *   **构造全局矩阵**:
            *   Rank 0 收集所有 rank 的 `A_i`，并沿 M 维度拼接成一个大的 `A_full` 矩阵，shape 为 `[rankSize * M, K]`。
            *   Rank 0 收集所有 rank 的 `B_i`，并沿 N 维度拼接成一个大的 `B_full` 矩阵，shape 为 `[K, N]`。
        *   **计算Golden C**: 执行 `C_golden = A_full @ B_full`，得到基准结果，shape 为 `[rankSize * M, N]`。
    3.  **执行算子**:
        *   所有 rank 调用融合算子核函数，得到各自的输出分片 `C_npu_i`，shape 为 `[M, N]`。
    4.  **结果校验**:
        *   将 `C_golden` 矩阵按行切分成 `rankSize` 块，每块 `C_golden_i` 的 shape 为 `[M, N]`。
        *   在每个 rank `i` 上，比较其算子输出 `C_npu_i` 和对应的 `C_golden_i`，确保误差在允许范围内。
