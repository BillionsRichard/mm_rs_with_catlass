使用方式: 
1.在3rdparty目录下, clone catlass代码仓：
    git clone https://gitee.com/ascend/catlass.git

2.在shmem/examples/matmul_allreduce目录下进行demo编译:
    bash scripts/build.sh

3.在shmem/examples/matmul_allreduce目录执行demo:
    # RANK、M、K、N等参数可自行输入
    # 从0卡开始，完成2卡的matmul_allreduce, matmul部分完成(M, K) @ (K, N)的矩阵乘
    bash scripts/run.sh -ranks 2 -M 1024 -K 2048 -N 8192