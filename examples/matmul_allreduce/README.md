使用方式: 
1.在shmem/目录编译:
    bash scripts/build.sh

2.在shmem/examples/matmul_allreduce目录执行demo:
    # RANK、M、K、N等参数可自行输入
    # 从0卡开始，完成2卡的matmul_allreduce, matmul部分完成(M, K) @ (K, N)的矩阵乘
    bash scripts/run.sh -ranks 2 -M 1024 -K 2048 -N 8192