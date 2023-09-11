# README

# 一、背景

## 模型简介

- 语义分割是计算机视觉中的基本任务，在语义分割中我们需要将视觉输入分为不同的语义类别。例如，我们可能需要区分图像中属于汽车的所有像素，并把这些像素涂成蓝色。

- Segformer模型主要用于语义分割，将Transformer与轻量级多层感知器（MLP）解码器统一起来。它的主要优势在于设计了一个新颖的分级结构Transformer编码器，这种结构可以输出多尺度特征，而且不需要位置编码，这就在测试分辨率与训练分辨率不同时，避免了性能下降的问题。另外，SegFormer还避免了复杂的解码器，而是采用MLP解码器从不同的层聚合信息，从而结合局部Attention和全局Attention来呈现强大的表示。这种简单和轻量级的设计使得SegFormer成为一种有效的语义分割Transformer。

- 论文地址：[Paper](https://arxiv.org/abs/2105.15203)

- 模型代码地址：https://github.com/NVlabs/SegFormer

  ![1](https://camo.githubusercontent.com/56105d95870dfb07f10f3c2dab225484bcbb758fc7e5252ecff0ee29f3f4b7e0/687474703a2f2f686970686f746f732e62616964752e636f6d2f666565642f7069632f6974656d2f653832346238393961393031346330383064343862323339303637623032303837626634663433662e6a70672363726f703d302663726f703d302663726f703d312663726f703d312669643d6a71796d72266f726967696e4865696768743d323233266f726967696e57696474683d343934266f726967696e616c547970653d62696e61727926726174696f3d3126726f746174696f6e3d302673686f775469746c653d66616c7365267374617475733d646f6e65267374796c653d6e6f6e65267469746c653d)

# 二、代码目录

脚本和样例代码

```
TRT-Segformer/
├── LICENSE
├── README.md
├── calib
│   └── segformer_calibration_test.cache
├── data
│   ├── calib_data
│   │   ├── frankfurt_000000_001236_leftImg8bit.png
│   │   └── frankfurt_000000_001751_leftImg8bit.png
│   ├── npy
│   │   └── 1.npy
│   ├── onnx_save
│   │   └── 1.png
│   ├── png
│   │   └── 1.png
│   └── predata_save.py
├── log
│   ├── segformer_test_int8_encoderScore.txt
│   ├── sim_fp16_segformer_b1_1024_1024_city_160k_encoderScore.txt
│   ├── sim_fp16_segformer_b1_1024_1024_city_160k_v1_encoderScore.txt
│   ├── sim_fp32_segformer_b1_1024_1024_city_160k_encoderScore.txt
│   ├── sim_fp32_segformer_b1_1024_1024_city_160k_v1_encoderScore.txt
│   ├── sim_fp32_segformer_b1_1024_1024_city_160k_v2_encoderScore.txt
│   ├── sim_plan_b1_1024_1024_fp16.log
│   ├── sim_plan_b1_1024_1024_fp16_v1.log
│   ├── sim_plan_b1_1024_1024_fp16_v2.log
│   ├── sim_plan_b1_1024_1024_fp32.log
│   ├── sim_plan_b1_1024_1024_fp32_v1.log
│   └── sim_plan_b1_1024_1024_fp32_v2.log
├── python
│   ├── ln_replace.py
│   ├── testSegFormer.py
│   └── trt_int8_quant.py
├── scripts
│   ├── sim_1024_1024_origin_build.sh
│   ├── sim_1024_1024_origin_build_fp16.sh
│   ├── sim_1024_1024_origin_build_fp16_v1.sh
│   ├── sim_1024_1024_origin_build_fp16_v2.sh
│   ├── sim_1024_1024_origin_build_v1.sh
│   └── sim_1024_1024_origin_build_v2.sh
└── soFile
    └── LayerNorm.so
```

# 三、环境配置

## 服务器环境确认：

服务器驱动配置为：470.199.02

Nvidia docker 文档： https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_21-10.html#rel_21-10

支持470驱动的，最新版本为**21.10** TRT版本为 **8.0.3.4**.

对应docker images名称为：nvcr.io/nvidia/tensorrt:21.10-py3

## docker的环境搭建:

Docker 安装

```Shell
sudo apt-get install docker
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L -O https://nvidia.github.io/nvidia-docker/gpgkey #大写欧，会在本地保存一个gpgkey文件
sudo apt-key add gpgkey#会输出OK

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

查看 cat /etc/apt/sources.list.d/nvidia-docker.list

```Prolog
deb https://nvidia.github.io/libnvidia-container/stable/ubuntu16.04/$(ARCH) /
#deb https://nvidia.github.io/libnvidia-container/experimental/ubuntu16.04/$(ARCH) /
deb https://nvidia.github.io/nvidia-container-runtime/stable/ubuntu16.04/$(ARCH) /
#deb https://nvidia.github.io/nvidia-container-runtime/experimental/ubuntu16.04/$(ARCH) /
deb https://nvidia.github.io/nvidia-docker/ubuntu16.04/$(ARCH) /
```

最后进行

```Shell
sudo apt-get update 
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 拉取镜像

```Bash
docker pull nvcr.io/nvidia/tensorrt:21.10-py3
```

## 初始化容器

```Bash
## origin
docker run -it -e NVIDIA_VISIBLE_DEVICES=0 --gpus "device=0" --name "trt" \
--shm-size 16G --ulimit memlock=-1 --ulimit stack=67108864 \
-v ~:/work \
nvcr.io/nvidia/tensorrt:21.10-py3 /bin/bash
```

先查看容器情况

```
docker ps -a
```

发现有trt的容器在跑，若没跑则用 `docker start trt`  开启。

## 进入容器

```Bash
docker exec -it trt /bin/bash
```

然后我们就在 `/workspace`目录下，`/work/trt` 能回到原始目录。

# 四、模型搭建

本项目基于Segformer-b1进行推理优化。

## 预训练权重下载

```Bash
git clone https://github.com/NVlabs/SegFormer.git
mkdir ckpt
## 下载权重
ls ckpt/
segformer.b1.1024x1024.city.160k.pth
```

## 预训练权重 ---> ONNX权重

```Bash
mkdir onnx/
python tools/pytorch2onnx.py local_configs/segformer/B1/segformer.b1.1024x1024.city.160k.py  --checkpoint ckpt/segformer.b1.1024x1024.city.160k.pth --output-file onnx/segformer.b1.1024.1024.city.160k.onnx
```

# 五、模型计算图优化

## ONNX Sim
我们使用了onnxsim优化onnx简化模型
```Bash
onnxsim trt/onnx/segformer.b1.1024.1024.city.160k.onnx trt/onnx/sim.segformer.b1.1024.1024.city.160k.onnx
```

## **FP32 & FP16 engine build**

```Bash
# Fp32
sh scripts/sim_1024_1024_origin_build_fp16.sh

# FP16
sh sim_1024_1024_origin_build.sh
```

### 精度测试

```shell
python python/testSegFormer.py
```

- **Baseline fp32**

![img](https://uvj4ui710rz.feishu.cn/space/api/box/stream/download/asynccode/?code=YWQzMzBlZDAxYjgzMzkwNzhjYTkxMzYwYmY2MjEyNmVfNFhUZlNDSUF3M0gzYXphNktOajlXbk9sVUtNQ05DRnhfVG9rZW46Sjd6WmJsSGR6b05TcXB4bjJuOWNna2xoblNoXzE2OTQ0NDU5MjE6MTY5NDQ0OTUyMV9WNA)

- **Baseline fp16**

![img](https://uvj4ui710rz.feishu.cn/space/api/box/stream/download/asynccode/?code=YzlhNTVkNjE2NjFlNDBhOTFjMWQ1MzQ3ZGNjNjk5ZmFfRnAwRXozUU9nZm4xRFBmaVVnR0xBYTNzdzc5dUNiZDRfVG9rZW46VjN2UmJoMGpzb01YRmh4N2pqVGNNTUN0bkxEXzE2OTQ0NDU5MjE6MTY5NDQ0OTUyMV9WNA)

## LayerNorm算子融合Plugin + **LayerNorm 算子替换**

![img](https://uvj4ui710rz.feishu.cn/space/api/box/stream/download/asynccode/?code=YjdmMjMxYTM4Y2M1YmNkNzliNjgxNTMxYTA2YzQ2MWNfczZkblZkcUw2dTRkUE52SFJTRHAwSXhZUlJmdnBSTWFfVG9rZW46SG00aGJSNXU2b1dCT1F4RGFSQmNjcVRZbkRnXzE2OTQ0NDU5MjE6MTY5NDQ0OTUyMV9WNA)

### V1：替换所有LayerNorm算子

共计30个算子，fp32+fp16 精度/性能有部分提升

fp32

![img](https://uvj4ui710rz.feishu.cn/space/api/box/stream/download/asynccode/?code=OWY1MTY5NjU2NjFjMTNjYzQzMTE2ZjY0NGIyZWYxNzZfbHY0YWp6TkNBMWk0V3Y3MlBoRUVjSmF4NUhkYVVhdTVfVG9rZW46WDRKeWJHb0hMbzR0WnR4cVJDUGNsangxbllkXzE2OTQ0NDU5MjE6MTY5NDQ0OTUyMV9WNA)

fp16

![img](https://uvj4ui710rz.feishu.cn/space/api/box/stream/download/asynccode/?code=NzY4MDI0YWM4NmUyZGY2YjY5YzUzOWQyNjJhNjRjNjhfWXkwdEFVV0hKN1RwMGZmZlZ1SjZEMWZFWEZlUGFyYk5fVG9rZW46STh2aGJMaDVlb25rZVB4NkZVSmNSbTN2bmNkXzE2OTQ0NDU5MjE6MTY5NDQ0OTUyMV9WNA)

### V2：替换部分LayerNorm算子

共计6个算子

fp32：性能提升较小

![img](https://uvj4ui710rz.feishu.cn/space/api/box/stream/download/asynccode/?code=NWQ2ZmI1NTZlYjVmM2QzNTYwNGRjYWZjZGE0ZWVhMDlfdU1reVZ5TFRrV2lBUTNTblRQckxNb2lza2Rpa2FxOTRfVG9rZW46SjhVRWJocjJzb3kzMUF4bkk4a2NwQmlXblpiXzE2OTQ0NDU5MjE6MTY5NDQ0OTUyMV9WNA)

fp16:性能无明显提升

![img](https://uvj4ui710rz.feishu.cn/space/api/box/stream/download/asynccode/?code=OWY0ZGU0ZmJiMTU4ZDlkNmU5NzU1ZWRmYmVmMTg5OGFfZ1cyeU1tRk1TWFhIVHZ4NmxDOVBPQjkxODRHWU90SGRfVG9rZW46S0FZaGJtTjZwb0FRRHl4eDRpWGNraWl3bmdiXzE2OTQ0NDU5MjE6MTY5NDQ0OTUyMV9WNA)

### Profiler

```Bash
nsys profile -o segformer-fp32-moPlugin --force-overwrite true  trtexec --loadEngine=/root/trt/plan/sim_fp32_segformer_b1_1024_1024_city_160k.plan --iterations=10 --idleTime=500 --duration=0 --useSpinWait 
```

## INT8 量化

```Python
python trt_int8_quant.py
```

- 构建Dataloader读取测试集数据，重写next_batch，作为stream传递给Calibrator。
- Calibrator读取每个batch，通过read & write对cache进行更新，得到最终量化表。
- 将重写的Calibrator加入config中，得到最后的engine。

![img](https://uvj4ui710rz.feishu.cn/space/api/box/stream/download/asynccode/?code=OTg4NTUyODhjNjYxY2FjNzk4YWNiMmIxZTI3MzNhOWJfMU04aHdTT3M1elhYZmdoYlkzS2F3UGhWbERVYzdhM1lfVG9rZW46UUhNNGJlbXVEb0pGZHB4RHU0emNRbVhvbnp2XzE2OTQ0NDU5MjE6MTY5NDQ0OTUyMV9WNA)

我们构建的int8 engine with partial LayerNormPlugin在batch_size=1，图片大小为1024x1024时24ms可以完成推理，但是相对误差（分类错误的像素点占总像素点的比例）在1e-2级别，这个误差对语义分割来说是一个不可用的状态（几乎全错）。
后续可以尝试的改进：
（1）方法一：找到使用低精度的层，手动调整为高精度实现，重新构建并测试生成的engine精度，直到找到问题层
（2）方法二：使用Polygraphy debug工具，详情见https://github.com/NVIDIA/TensorRT/tree/main/tools/Polygraphy/examples/cli/debug


# 补充

----

## 环境信息：

操作系统：Ubuntu x86_64 

GPU：Nvidia Tesla T4

CUDA：12.0

TensorRT：8.6.1

Python：3.8
