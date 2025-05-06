使用方式: 
1.在3rdparty目录下, clone AscendC Templates代码仓：
    git clone https://gitee.com/ascend/ascendc-templates.git

2.在example/matmul_allreduce目录下进行demo编译:
    bash build.sh

3.在example/matmul_allreduce目录下生成golden数据：
    python3 utils/gen_data.py 1 2 1024 1024 16 0 0

4.在example/matmul_allreduce目录执行demo：
    bash run.sh

5.在example/matmul_allreduce目录验证算子精度：
    python3 utils/verify_result.py ./out/output.bin ./out/golden.bin 1 1024 1024 16