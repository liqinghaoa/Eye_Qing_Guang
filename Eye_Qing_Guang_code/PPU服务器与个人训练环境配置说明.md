# PPU 服务器与个人训练环境配置说明

> 用途：供 Codex 读取，用于后续修改 `Eye_Qing_Guang` 项目的训练配置、设备配置、路径配置与多卡训练设置。  
> 当前目标：在服务器既有 DeepSeek 推理服务继续运行的情况下，使用管理员分配的后四张 PPU（物理 PPU 4、5、6、7）进行训练实验。

---

## 1. 服务器基本信息

### 1.1 操作系统

```text
NAME="Alibaba Cloud Linux"
VERSION="3 (OpenAnolis Edition)"
PRETTY_NAME="Alibaba Cloud Linux 3.2104 U12.1 (OpenAnolis Edition)"
ID="alinux"
ID_LIKE="rhel fedora centos anolis"
VERSION_ID="3"
```

内核：

```text
Linux 6.6.102-5.600.al8.x86_64
```

架构：

```text
x86_64
```

---

## 2. CPU 配置

CPU 型号：

```text
AMD EPYC 9A75 64-Core Processor
```

拓扑：

```text
Socket(s):            2
Core(s) per socket:   64
Thread(s) per core:   2
CPU(s):               256
NUMA node(s):         2
```

因此：

```text
物理 CPU：2 颗
物理核心：128 核
逻辑线程：256 线程
```

CPU 最大频率：

```text
约 5.008 GHz
```

NUMA：

```text
NUMA node0 CPU(s): 0-63,128-191
NUMA node1 CPU(s): 64-127,192-255
```

---

## 3. 内存配置

```text
总内存：约 2.2 TiB
当前可用内存：约 2.0 TiB
Swap：0
```

示例状态：

```text
Mem: 2.2Ti total
     259Gi used
     1.4Ti free
     607Gi buff/cache
     2.0Ti available
```

---

## 4. 存储配置

### 4.1 系统盘

```text
挂载点：/
文件系统：ext4
总容量：177 GB
已用：约 73 GB
可用：约 97 GB
```

不建议把大型训练数据、checkpoint、实验输出长期写入系统盘。

### 4.2 数据盘

```text
挂载点：/data
文件系统：xfs
总容量：约 14 TB
已用：约 644 GB
```

训练项目和数据均应优先放在 `/data` 下。

---

## 5. PPU 加速卡配置

服务器共有：

```text
8 × ZW-M890P
```

每张卡显存：

```text
147456 MiB ≈ 144 GiB
```

总 PPU 显存：

```text
8 × 144 GiB ≈ 1152 GiB ≈ 1.125 TiB
```

PPU 管理工具：

```text
PPU-SMI 1.29
Driver Version: 2.2.2-r691b0
HGGC Version: 13.0
```

`nvidia-smi` 实际路径：

```text
/usr/local/PPU_SDK/CUDA_SDK/bin/nvidia-smi
```

注意：该服务器不是 NVIDIA GPU 环境。这里的 `nvidia-smi` 是 T-head PPU SDK 提供的兼容接口。

---

## 6. PPU SDK / CUDA 兼容环境

现有镜像内部包含 PPU SDK，并通过 CUDA 风格接口向 PyTorch 暴露 PPU。

关键环境变量包括：

```text
CUDA_TOOLKIT_ROOT=/usr/local/PPU_SDK/CUDA_SDK
CUDA_PATH=/usr/local/PPU_SDK/CUDA_SDK
CUDA_HOME=/usr/local/PPU_SDK/CUDA_SDK
CUDA_SDK=/usr/local/PPU_SDK/CUDA_SDK
CUDACXX=/usr/local/PPU_SDK/CUDA_SDK/bin/nvcc

PPU_SDK=/usr/local/PPU_SDK
PPU_HOME=/usr/local/PPU_SDK
PPU_PATH=/usr/local/PPU_SDK
PPU_VERSION=v2.1.1

CUDNN_HOME=/usr/local/PPU_SDK/CUDA_SDK
PJRT_DEVICE=CUDA
```

因此当前项目代码原则上仍使用：

```python
torch.cuda
cuda:0
cuda:1
...
```

而不是改成 `ppu:0` 之类的设备名。

---

## 7. 网络情况

服务器可以访问公网，但通过本地 HTTP/HTTPS 代理：

```text
HTTP_PROXY=http://127.0.0.1:10808
HTTPS_PROXY=http://127.0.0.1:10808
http_proxy=http://127.0.0.1:10808
https_proxy=http://127.0.0.1:10808
```

已经确认：

```text
https://pypi.org
https://pypi.org/simple/
```

均可正常访问并返回 HTTP 200。

因此后续在个人容器中安装 Python 包时，应保留这些代理环境变量。

---

## 8. 容器运行环境

宿主机使用：

```text
Podman 4.9.4-rhel
```

Docker 未安装。

已有基础镜像：

```text
localhost/asllm:2.1.0-pytorch2.11.0-ubuntu24.04-sail2.1.1-cuda13.0-sglang0.5.16-vllm0.26.0-py312
```

镜像大小：

```text
约 52.5 GB
```

已创建个人标签：

```text
localhost/kanglong/ppu-train-base:v1
```

该标签与原镜像指向同一个 IMAGE ID：

```text
28d12bf444d2
```

---

## 9. 基础镜像内 Python / PyTorch 环境

已确认：

```text
Python: 3.12.3
Python path: /usr/local/bin/python

PyTorch: 2.11.0
torch.version.cuda: 13.0
torch.cuda.is_available(): True
```

PyTorch 能正常识别全部 8 张 ZW-M890P。

示例：

```text
0 ZW-M890P
1 ZW-M890P
2 ZW-M890P
3 ZW-M890P
4 ZW-M890P
5 ZW-M890P
6 ZW-M890P
7 ZW-M890P
```

宿主机没有 Conda：

```text
conda: command not found
```

当前训练应优先基于现有 Podman 镜像，而不是在宿主机重新配置 Conda/PyTorch/CUDA。

---

## 10. 既有 DeepSeek 服务

服务器当前存在容器：

```text
ds41
```

其运行 DeepSeek 推理服务：

```text
DeepSeek-V4.1-Flash
```

使用：

```text
SGLang
Tensor Parallel size = 8
```

也就是物理 PPU 0～7 均有对应 SGLang scheduler 进程。

当前每张卡大约已经被 DeepSeek 占用：

```text
约 69～70 GiB 显存
```

但管理员已经明确确认：

```text
训练任务允许与当前 DeepSeek 服务共享 PPU
```

并且管理员明确分配：

```text
另一方使用前四张 PPU：物理 PPU 0、1、2、3
本人使用后四张 PPU：物理 PPU 4、5、6、7
```

因此本项目训练只能使用：

```text
物理 PPU 4、5、6、7
```

---

## 11. 个人训练容器

已创建个人训练容器：

```text
qinghao_ppu_train
```

基础镜像：

```text
localhost/kanglong/ppu-train-base:v1
```

容器当前状态：

```text
Up
```

创建时固定设置：

```text
CUDA_VISIBLE_DEVICES=4,5,6,7
```

因此容器内部只看到 4 张卡。

容器内部设备映射关系为：

```text
容器 cuda:0 -> 物理 PPU 4
容器 cuda:1 -> 物理 PPU 5
容器 cuda:2 -> 物理 PPU 6
容器 cuda:3 -> 物理 PPU 7
```

注意：

后续训练脚本内部不得继续使用：

```text
cuda:4
cuda:5
cuda:6
cuda:7
```

而应使用：

```text
cuda:0
cuda:1
cuda:2
cuda:3
```

单卡训练默认优先使用：

```text
cuda:0
```

即物理 PPU 4。

四卡训练使用：

```text
cuda:0 ～ cuda:3
```

---

## 12. 项目与数据目录

### 12.1 数据目录

宿主机：

```text
/data/workspace/KanglongTan/data
```

容器内保持完全相同路径：

```text
/data/workspace/KanglongTan/data
```

挂载方式：

```text
rw
```

### 12.2 项目目录

宿主机：

```text
/data/workspace/QinghaoLi/Eye_Qing_Guang
```

容器内保持完全相同路径：

```text
/data/workspace/QinghaoLi/Eye_Qing_Guang
```

挂载方式：

```text
rw
```

项目内部已经自行规划：

```text
训练程序
配置文件
实验目录
输出目录
```

因此后续无需额外创建统一 outputs 目录。

---

## 13. 目录挂载验证结果

两个目录均已在容器中成功识别：

```text
/data/workspace/KanglongTan/data
/data/workspace/QinghaoLi/Eye_Qing_Guang
```

并完成写入测试：

容器中写入：

```text
/data/workspace/QinghaoLi/Eye_Qing_Guang/container_test.txt
```

宿主机可以读取相同内容。

因此：

```text
宿主机项目目录 <-> qinghao_ppu_train 容器
```

读写映射正常。

---

## 14. PPU 计算验证结果

已在个人容器中测试 PyTorch 对 4 张 PPU 的访问。

结果：

```text
PyTorch: 2.11.0
CUDA: 13.0
available: True
count: 4
```

设备：

```text
0 ZW-M890P
1 ZW-M890P
2 ZW-M890P
3 ZW-M890P
```

单卡矩阵计算测试：

```text
device: ZW-M890P
shape: torch.Size([2048, 2048])
PPU COMPUTE PASS
```

四卡逐卡矩阵计算测试：

```text
cuda:0 -> ZW-M890P -> PASS
cuda:1 -> ZW-M890P -> PASS
cuda:2 -> ZW-M890P -> PASS
cuda:3 -> ZW-M890P -> PASS
```

因此当前已经确认：

```text
Podman
  ↓
PPU SDK
  ↓
CUDA compatibility layer
  ↓
PyTorch 2.11.0
  ↓
ZW-M890P
```

计算链路正常。

---

## 15. 后续修改训练配置时的约束

Codex 后续修改 `Eye_Qing_Guang` 项目时应遵守以下原则。

### 15.1 不重新安装官方 NVIDIA CUDA

当前不是 NVIDIA GPU 环境。

不要执行类似：

```text
apt/yum install nvidia-cuda
conda install cuda
pip 安装与当前 PPU 环境不兼容的 CUDA 运行时
```

除非明确确认厂商支持。

---

### 15.2 不覆盖当前 PyTorch

当前基础镜像已经提供：

```text
PyTorch 2.11.0
CUDA compatibility 13.0
PPU SDK v2.1.1
```

不要直接使用普通 PyPI/官方 NVIDIA wheel 覆盖 `torch`，否则可能破坏 PPU 兼容。

如项目 requirements 中固定了其他 `torch` / `torchvision` 版本，应先检查兼容性，再决定如何处理。

---

### 15.3 设备设置

项目代码应优先兼容标准 PyTorch CUDA 写法：

```python
device = torch.device("cuda:0")
```

单卡：

```text
cuda:0
```

多卡：

```text
cuda:0,cuda:1,cuda:2,cuda:3
```

容器外物理 PPU 4～7 已通过 `CUDA_VISIBLE_DEVICES=4,5,6,7` 屏蔽和重新编号。

---

### 15.4 多卡训练

如果项目使用 DDP，应按容器内 4 卡处理：

```text
world_size = 4
local_rank = 0,1,2,3
```

不要使用物理编号 4～7 作为 DDP local rank。

是否启用 4 卡训练，需要根据项目当前实现决定；不应未经检查直接强制改为 DDP。

---

### 15.5 数据路径

数据根目录固定为：

```text
/data/workspace/KanglongTan/data
```

项目根目录固定为：

```text
/data/workspace/QinghaoLi/Eye_Qing_Guang
```

如果项目存在 Windows 路径或旧服务器路径，应修改为以上 Linux 路径。

---

### 15.6 显存共享约束

每张物理 PPU 总显存约：

```text
144 GiB
```

DeepSeek 当前每张卡已占用约：

```text
69～70 GiB
```

因此训练任务单卡理论剩余约：

```text
约 74～76 GiB
```

但不应把剩余显存全部吃满，应为：

```text
DeepSeek 显存波动
PyTorch 临时张量
workspace
缓存
验证阶段额外开销
```

预留显存空间。

后续 `batch_size` 应根据实际训练峰值显存逐步测试，而不是直接按 144 GiB 单卡总显存估算。

---

## 16. 常用容器命令

查看个人容器：

```bash
podman ps --filter name=qinghao_ppu_train
```

进入容器：

```bash
podman exec -it qinghao_ppu_train bash
```

退出容器：

```bash
exit
```

停止容器：

```bash
podman stop qinghao_ppu_train
```

启动容器：

```bash
podman start qinghao_ppu_train
```

停止 shell 不会删除项目文件，因为项目和数据均存储在宿主机 `/data` 并通过 bind mount 挂载。

---

## 17. 当前阶段

目前已经完成：

- [x] 确认服务器系统配置
- [x] 确认 CPU / 内存 / 数据盘
- [x] 确认 8 张 ZW-M890P
- [x] 确认 PPU SDK / CUDA 兼容层
- [x] 确认服务器公网和 PyPI 可访问
- [x] 确认 Podman 基础镜像
- [x] 确认 PyTorch 2.11.0 可识别 PPU
- [x] 创建个人镜像标签
- [x] 创建个人训练容器 `qinghao_ppu_train`
- [x] 固定个人容器使用物理 PPU 4、5、6、7
- [x] 挂载数据目录
- [x] 挂载项目目录
- [x] 验证宿主机与容器读写
- [x] 验证单卡 PPU 计算
- [x] 验证四张 PPU 均可执行 PyTorch 计算

下一阶段：

```text
上传 / 检查 Eye_Qing_Guang 项目
→ 检查项目 requirements / 依赖
→ 检查训练入口脚本
→ 修改 Linux 路径
→ 修改设备配置
→ 处理 PPU/PyTorch 兼容问题
→ 单卡 smoke test
→ 正式训练
→ 如需要，再配置四卡 DDP
```

---

## 18. 给 Codex 的修改原则

读取本文件后，在修改项目训练配置之前应先：

1. 检查项目当前训练入口、配置文件和依赖文件。
2. 不假设该环境是标准 NVIDIA CUDA。
3. 保留当前 PPU 适配版 PyTorch 2.11.0。
4. 不直接覆盖 `torch`、CUDA 或 PPU SDK。
5. 将容器内 GPU 编号视为 `cuda:0～cuda:3`。
6. 默认优先实现单卡 `cuda:0` smoke test。
7. 单卡成功后再考虑多卡。
8. 所有数据路径使用 `/data/workspace/KanglongTan/data`。
9. 项目路径使用 `/data/workspace/QinghaoLi/Eye_Qing_Guang`。
10. 在修改依赖或训练配置前，先说明修改原因和可能的 PPU 兼容风险。
