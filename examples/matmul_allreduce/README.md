使用方式: 
1.在3rdparty目录下, clone catlass代码仓：
    git clone https://gitee.com/ascend/catlass.git

2.在example/matmul_allreduce目录下进行demo编译:
    bash scripts/build.sh

3.在example/matmul_allreduce目录执行demo:
    # RANK、M、K、N可自行输入
    # 完成RANK卡下的matmul_allreduce, matmul部分完成(M, K) @ (K, N)的矩阵乘
    bash scripts/run.sh $RANK $M $K $N