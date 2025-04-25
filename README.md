使用方式：

1.将稳定版本解压到3rdparty目录下

2.bash scripts/build.sh

3.bash scripts/run.sh

跑默认8卡用例

4.bash scripts/run.sh -ranks 8 -ipport tcp://127.0.0.1:8666 -gnpus 8

run.sh目前支持-ranks -ipport -gnpus三个入参分别设置总rank数，ip和端口，单机卡数。

不输入参数时各参数默认值分别为8，tcp://127.0.0.1:8666，8。

