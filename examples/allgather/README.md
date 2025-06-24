使用方式: 
1.在shmem/examples/allgather目录下进行demo编译:
    bash build.sh

2.在shmem/examples/allgather目录执行demo:
    # 完成RANKS卡下的allgather，并打印各rank获取的值。
    bash run.sh -ranks ${RANKS} 