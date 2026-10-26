# ConvNeXt-Tiny 512：冻结融合教师、标准蒸馏与校正蒸馏实验实现方案 v1

> 编写日期：2026-10-25。2026-10-26实现状态：E1/E2代码已实现，10项新增合成检查与19项相关回归检查通过；尚未执行真实GPU预检、正式训练或生成本方案的真实 submission。运行说明见代码项目 README 的“ConvNeXt 512 冻结三教师蒸馏 v1”章节。
>
> 本文用于交接代码实践 Thread。新增独立实验；已有 ConvNeXt、EfficientNet、B0、224/288 和 train+valid final-fit 的配置、权重及结果均保留。
>
> 2026-10-25 交接复查：补齐预检预算与入口、学生 checkpoint 字段、清单协议复用、严格配置约束及结果解释边界。保留原 v1 实验配方。

## 1. 实验目标与论文关系

本实验研究：能否将现有三个模型的互补预测迁移到一个 ConvNeXt-Tiny 学生，使最终推理仅加载一个模型。

论文来源为 Lanhu Wu 等的 **CNN-Transformer Rectified Collaborative Learning for Medical Image Segmentation**，正式发表于 IEEE TCSVT，2025，35(5)：4072–4086，DOI：`10.1109/TCSVT.2024.3523316`。

- 本地阅读版本：`../01_Background_and_Literature/key_papers/cnn_transformer.pdf`，2024 年 arXiv v2。
- 正式记录：<https://ieeexplore.ieee.org/document/10816601/>。
- 原作者代码：<https://github.com/LanhooNg/CTRCL>，只作方法参考，本方案不要求安装其完整分割工程。

论文采用 CNN 与 Transformer 双向协同，包含标签校正的输出层互学习 RLCL 和依赖像素掩码的特征层互学习 CFCL。本方案作如下改造：

| 项目 | 原论文 | 本方案 v1 |
| --- | --- | --- |
| 任务 | 像素级医学图像分割 | 图像级青光眼二分类 |
| 网络 | 两个同时更新的学生 | 三个冻结教师、一个更新的学生 |
| 信息传递 | 双向在线互学习 | 教师到学生的单向蒸馏 |
| 输出校正 | 对各学生错误像素修正 | 对融合教师错误的图像级软目标修正 |
| 蒸馏目标 | 原文式 (12) 的 KL 表达 | 标准软标签 BCE；其梯度等价于 `KL(教师 || 学生)` |
| 特征层 CFCL | 使用像素标签建立类别原型 | v1 不实现 |
| 推理 | 分别测试学生网络 | 只使用 ConvNeXt-Tiny 学生 |

因此，成果应称为“受 CTRCL 启发的标签校正蒸馏”，不能称为原论文完整复现，也不能将本方案等同于 CNN–Transformer 协同。现有三位教师均属于 CNN。

## 2. 现有证据与实验边界

### 2.1 官方 valid 上的离线融合结果

以下结果来自现有 `predictions/best/valid_predictions.csv`，按相同 `image_id` 与标签核对后，对未取整概率做等权算术平均。没有用测试标签或官网分数选融合权重。

| 方案 | AUROC | AP | 固定 0.5 阈值的 Balanced Accuracy |
| --- | ---: | ---: | ---: |
| ConvNeXt 1K 512 单模型 | 0.98762166 | 0.80671042 | 0.91859323 |
| ConvNeXt 1K + ConvNeXt 22K，等权 | 0.98922066 | 0.83951161 | 0.89957323 |
| 两个 ConvNeXt + EfficientNet-V2-S，等权 | 0.98989338 | 0.84933534 | 0.90769175 |

三模型融合相对主单模型的 valid AUROC 增加约 0.00227，但固定阈值表现下降。v1 冻结三模型等权教师，目的为迁移概率排序知识；不承诺学生一定超越教师，也不将上述 valid 指标换算成官网分数。

教师们已在官方 train 上拟合；它们对 train 的预测可能过度自信。valid 融合收益只是开展蒸馏的动机，不代表 train 软目标必然有用。需记录软目标熵和错误比例判断实际情况。

### 2.2 数据与选模

- 官方 train：51,396 张，阳性 1,657 张、阴性 49,739 张，仅这部分做学生梯度更新和教师软目标计算。
- 官方 valid：5,849 张，阳性 191 张、阴性 5,658 张，用于学生逐轮评估、选 best 和早停。
- 官方 test：5,855 张，学生配置与 best 选定后才进行最终推理，不进入训练或软目标生成。
- 不合并 train+valid，不重新划分数据，不使用 train+valid final-fit 权重作为教师。
- 教师在 valid 上已有预测可用于核对；不使用 valid 标签为训练目标做修正。本文校正公式仅作用于 train 样本。

依据《比赛参赛规则》第 3、5、6 条，模型选择和参数确定使用允许的开发数据；test 不用于 checkpoint、阈值、训练轮数或蒸馏系数选择。

官方规则：<https://ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20261023/aa80bececc2ee40e69cde5f8f2a0eaec.pdf>。

## 3. 三组对照与独立实验身份

代码根目录约定为 `Eye_Qing_Guang_code`；本文后续代码、配置、权重及实验路径均相对此目录。

| 编号 | 含义 | 学生监督损失 | 软目标 | 执行原则 |
| --- | --- | --- | --- | --- |
| E0 | 已有主实验 | BCE，`pos_weight=5` | 无 | 复用已有结果，不默认重训 |
| E1 | 标准冻结教师蒸馏 | 同 E0 | 三教师等权概率 | 新增配置与输出 |
| E2 | 标签校正蒸馏 | 同 E0 | 在融合概率上实施 ARM 风格校正 | 相对 E1 只改变校正开关 |

E0 目录：

```text
experiments/timm_convnext_tiny_fb_in1k_512_pw5_official_single3090_cache_s42_v1
```

E1 固定实验名与配置文件名：

```text
timm_convnext_tiny_fb_in1k_512_pw5_kd_ens3_std_official_single3090_s42_v1
configs/timm_convnext_tiny_fb_in1k_512_pw5_kd_ens3_std_official_single3090_s42_v1.yaml
```

E2 固定实验名与配置文件名：

```text
timm_convnext_tiny_fb_in1k_512_pw5_kd_ens3_rect_official_single3090_s42_v1
configs/timm_convnext_tiny_fb_in1k_512_pw5_kd_ens3_rect_official_single3090_s42_v1.yaml
```

两者分别写入 `experiments/<experiment.name>/`。计划执行顺序为 E1、E2；一次 run 只接受一个配置，不自动启动另一组或扩展超参数搜索。正式输出目录只要已经存在就拒绝启动，包括空目录，与现有管线 `exist_ok=False` 的语义保持一致。

## 4. 教师与学生的准确身份

### 4.1 教师固定清单

| ID | timm 模型 | checkpoint 路径 | 预期 epoch | 归一化 mean / std | 融合权重 |
| --- | --- | --- | ---: | --- | ---: |
| T1 | `convnext_tiny.fb_in1k` | `experiments/timm_convnext_tiny_fb_in1k_512_pw5_official_single3090_cache_s42_v1/checkpoints/best.pt` | 7 | ImageNet / ImageNet | 1/3 |
| T2 | `convnext_tiny.fb_in22k_ft_in1k` | `experiments/timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official_single3090_cache_s42_v1/checkpoints/best.pt` | 9 | ImageNet / ImageNet | 1/3 |
| T3 | `tf_efficientnetv2_s.in21k_ft_in1k` | `experiments/timm_v2s_in21k_ft_in1k_512_pw5_official_single3090_cache_s42_v1/checkpoints/best.pt` | 7 | `[0.5,0.5,0.5]` / `[0.5,0.5,0.5]` | 1/3 |

其中 ImageNet mean 为 `[0.485,0.456,0.406]`，std 为 `[0.229,0.224,0.225]`。

加载要求：

1. 使用已有严格 checkpoint loader；T1、T2 调用 `load_convnext_checkpoint`，T3 调用 `load_timm_checkpoint`。读取已训练的完整二分类模型，不只读取编码器。
2. 核对模型名称、单 logit 输出、特征维度（768/768/1280）、epoch、512 输入、`pos_weight=5`、官方 train/valid 协议、timm 版本 1.0.22。
3. 教师 `requires_grad_(False)`、`eval()`。不创建教师优化器，不更新其 BatchNorm 缓冲区、权重或其他状态。
4. 缺文件、错身份、误用 `last.pt` 或 final-fit 权重时明确失败；不得自动换教师、下载替代权重或随机初始化。
5. checkpoint 中记录的旧服务器数据绝对路径仅是来源信息。加载教师模型本身不需要访问旧数据路径；训练输入由本次学生数据配置统一提供。不要修改历史 checkpoint。
6. 同一教师参数若非有限，停止实验。训练开始前后抽查其参数和缓冲区未变化；无需每个 epoch 复制或另存教师权重。
7. 原 loader 只保证返回 `eval()` 模型，新教师封装必须另外执行 `requires_grad_(False)`，显式保留 FP32 参数，并核对传给优化器的参数集合仅属于学生。不能假定调用旧 loader 就已经冻结。

现有三份教师 best 文件合计约 304.27 MB；这是训练资源占用的一部分，最终学生推理不调用它们。

### 4.2 学生初始化

- 学生为 `convnext_tiny.fb_in1k`，特征维度 768，随机初始化一个输出 `[B,1]` 的二分类分类头。
- 加载本地通用预训练权重 `weights/convnext_tiny.fb_in1k.safetensors`；编码器与分类头都参与训练。
- 不用 T1 的任务 best 权重初始化学生；否则会同时改变蒸馏与续训起点。
- seed 固定为 42。建议先构建并加载教师，再重置全部训练随机种子，然后构建学生和训练 DataLoader。教师构建会消耗初始化随机数，必须防止它改变学生随机头、样本顺序和增强序列。
- E1、E2 的学生初始参数和 DataLoader 随机状态应一致。不能因为教师对象数或创建次序改变它们。

## 5. 同一图像视图与不同归一化

### 5.1 继承主实验的输入配方

学生保持 512 输入、`fov_presized_letterbox_v2`、BILINEAR、FOV、letterbox 及原增强：水平翻转 0.5，旋转 ±10°，亮度与对比度均为 `[0.9,1.1]`。

只从官方 train 缓存取训练样本，使用其“FOV 后长边缩放、增强后 letterbox”的训练路径。valid/test 使用各自确定性 evaluation 缓存。允许复用原 512 缓存与官方清单，不复制图像，也不把不同官方 split 的缓存改标为 train。

数据和缓存的实际绝对路径由运行环境设置，不能把某个历史配置快照中的 `E:/...` 或 `/data2/...` 当作所有机器通用路径。所有相对路径均锚定 `--project-root`，不依赖启动 shell 的当前目录。

### 5.2 教师与学生共享同一增强视图

每张 train 图像每次取样只随机增强一次，得到共同 RGB 视图 `u∈[0,1]`。教师和学生必须看到同一几何与颜色视图，只在归一化上不同。

```text
官方 train 图像/缓存
  → 一次随机增强与 letterbox → 同一 512 RGB 视图 u
      ├─ ImageNet 归一化 → 学生
      ├─ ImageNet 归一化 → T1、T2
      └─ (u - 0.5) / 0.5 → T3
```

现有 loader 输出的学生张量 `x_s=(u-mean_c)/std_c` 可直接给 T1、T2；T3 的输入在 FP32 中按下式转换，无需再次读取原图：

```python
u = x_s.float() * std_c + mean_c       # mean/std shape: [1,3,1,1]
x_t3 = (u - 0.5) / 0.5
```

- 不把 ImageNet 归一化后的 `x_s` 直接输入 T3。
- 不给教师分别调用随机增强或中心裁剪，不调用原始 timm 224/288 评测 transform。
- 不在转换中再次执行 FOV 或 letterbox。
- 默认不对反归一化结果额外 clamp；使用有限值与转换一致性检查定位问题。
- 保留 loader 返回的原始 FP32 `x_s` 作为共同视图来源。不要先将其 `.bfloat16()`，再为教师反归一化；学生的 BF16 由前向 autocast 实现，教师输入保持 FP32。
- 原有 valid/test 预测 CSV 不是 train 的软标签，禁止按行号或相似 ID 拼接成训练目标。

### 5.3 教师精度与显存

- 教师前向固定 FP32，在 `torch.no_grad()` 中执行，并显式关闭外层 autocast；所有教师保持 `eval()`。
- 学生前向 BF16；监督损失、软目标和蒸馏损失均 FP32。
- 默认 `teacher_forward_chunk=8`：一个学生微批次 48 张可拆成每次最多 8 张，依次计算三位教师，结果按原顺序拼回 `[48,1]`。
- 先完成教师前向并得到 detached 软目标，再做学生前向和反向，避免同时保留教师与学生的中间激活。教师无反向图。
- 训练时优先使用 `no_grad()`；若用 `inference_mode()`，须处理 inference tensor 进入需要保存 target 的反向算子时的兼容性，不能留下运行期错误。
- OOM 时先检查生命周期、分块和教师冻结，再减小教师 chunk。不得静默降低学生微批次或改教师精度。学生 batch 改变需登记新资源配置并保留有效 batch 及对照可比性的说明。
- 不预估“与基线一样快”。教师额外前向会增加训练时间，按有界预检中的实际测量记录。

## 6. 冻结融合教师与标准蒸馏 E1

### 6.1 v1 冻结参数

| 字段 | 固定值 | 原因 |
| --- | --- | --- |
| 教师数 | 3 | 使用已有互补模型 |
| 概率融合 | 等权算术平均 | 与已有 valid 融合分析一致 |
| 温度 `temperature` | 1.0 | 第一轮不额外改变软目标分布 |
| 蒸馏系数 `alpha` | 1.0 | 作为固定初始方案；不是已验证最优值 |
| 蒸馏 warmup | 0 | E1/E2 从第一轮使用相同系数 |
| 数值截断 `eps` | 1e-6 | 用于软目标与对数运算 |
| E1 修正 | `none` | 标准软目标 |
| E2 修正 | `arm_binary_v1` | 仅改变第 7 节的目标构造 |

不复制论文分割损失的 β=3，不加入 CFCL、特征 L2、MixUp/CutMix、重采样或新学习率分组。

### 6.2 融合顺序

教师输出正类 logit `z_1,z_2,z_3`，定义：

\[
p_j=\sigma(z_j),\qquad
q=\operatorname{clip}\left(\frac{p_1+p_2+p_3}{3},\epsilon,1-\epsilon\right).
\]

`q` 是 detached 的 `[B,1]` FP32 软目标。E1 取 `q_target=q`。

教师 logits、每位教师概率、融合概率、学生 logits 与标签在运算前均应检查形状和有限值，标签仅允许 0/1。遇到 NaN/Inf 要报错，不通过 clamp、填 0 或跳过样本掩盖错误。

固定先 sigmoid、后概率平均；不能改成 logit 平均、投票、择优教师或按样本置信度动态配权。温度 v1 固定为 1，训练式中不再额外乘温度平方。

### 6.3 损失

学生输出 `z_s`，真实标签 `y∈{0,1}`，两者 shape 都是 `[B,1]`。

\[
\ell_{sup}=-5y\log\sigma(z_s)-(1-y)\log(1-\sigma(z_s)),
\]

\[
\ell_{soft}=-q_{target}\log\sigma(z_s)
             -(1-q_{target})\log(1-\sigma(z_s)),
\]

\[
L=\operatorname{mean}(\ell_{sup})+
  \alpha\operatorname{mean}(\ell_{soft}),\qquad \alpha=1.
\]

使用稳定的 `binary_cross_entropy_with_logits`，不要先 sigmoid 学生再对概率求 log。

```python
sup_i = F.binary_cross_entropy_with_logits(
    student_logits.float(), labels.float(),
    pos_weight=torch.tensor([5.0], device=device), reduction="none")
soft_i = F.binary_cross_entropy_with_logits(
    student_logits.float(), q_target.detach().float(), reduction="none")
loss = sup_i.mean() + alpha * soft_i.mean()
```

`pos_weight=5` 只用于真实标签监督项，软标签项不使用 `pos_weight`、类别权重或重采样。教师概率来自加权 BCE 模型，不能自动解释为校准后的患病概率；本实验不做除 5、logit adjustment 或概率校准。

软 BCE 与 `KL(Bern(q_target) || Bern(sigmoid(z_s)))` 相差仅依赖软目标的熵，因此对学生梯度相同。可额外记录：

\[
H(q)=-q\log q-(1-q)\log(1-q),\quad
KL_{diag}=\operatorname{mean}(\ell_{soft}-H(q_{target})).
\]

`KL_diag` 是诊断字段，允许 FP32 舍入量级的微小负值，不据此悄悄更换损失。软 BCE 即使学生完全拟合教师也不必趋近 0；E1/E2 目标熵不同，总训练损失不适合直接比较优劣。

单输出二分类不能对 `[B,1]` 做单类别 softmax 后使用 categorical KL，这会得到恒为 1 的分布和无效蒸馏。

### 6.4 与 E0 比较时必须说明的类别权重变化

监督项的 pos_weight 虽然仍是 5，加入未加权软目标项后，整体目标的正负作用不再等同于单独 pw5。极限情况下若所有软目标都等于真实硬标签，alpha=1 时：

```text
正样本损失系数：5 + 1 = 6
负样本损失系数：1 + 1 = 2
等效正负相对系数：6 / 2 = 3
```

因此，E1 超过 E0 只能支持“本次联合训练目标改善了结果”，不能将收益全部归于教师知识；有效类别平衡和梯度作用也改变了。E2 在错误样本上更接近真实标签，同样需要注意这一点。

若 E1/E2 有收益且后续希望做严格方法归因，再单独安排硬目标控制：`L_control=BCE_pw5(z,y)+alpha*BCE_unweighted(z,y)`，其余保持一致，与 E1/E2 比较。alpha=1 时，该精确控制等于 `2*BCE_pw3(z,y)`；单独一个未乘 2 的 pw3 实验不能作为精确替代，因为整体损失尺度及裁剪影响也需匹配。该控制属于后续需登记的实验，v1 不自动增加训练任务。

## 7. 标签校正蒸馏 E2

E2 保持 E1 的教师、输入、温度、系数、初始化、优化器、数据和选模方式。唯一方法变化：`q_target` 改为下述校正目标。

### 7.1 在融合概率上校正

本方案是 `ARM(mean(p_j), y)`，不是 `mean(ARM(p_j,y))`；两种做法不等价，v1 只实现前者。

用固定 0.5 判断教师融合预测是否与 train 真实标签一致：

\[
\hat y=\mathbb{1}[q\ge0.5],\quad m=\mathbb{1}[\hat y\ne y].
\]

该 0.5 仅为本方法固定错误判定，不能用 test 调整。定义二类分布 `[1-q,q]`，则：

\[
p_{true}=yq+(1-y)(1-q),\qquad p_{pred}=\max(q,1-q),
\]

\[
\lambda_a=\frac{1}{1+p_{pred}-p_{true}},\qquad
\lambda_s=\exp(-CE)=p_{true},\qquad
\lambda_c=\exp[-H(q)],
\]

\[
\lambda=\lambda_a\lambda_s\lambda_c,
\]

\[
q_{target}=\operatorname{clip}\left(
 (1-m)q+m\left[\lambda q+(1-\lambda)y\right],
 \epsilon,1-\epsilon\right).
\]

只在错误样本上使用 `lambda`；正确样本原样保留。自然对数、FP32、逐样本计算。`CE` 与熵针对完整二类分布，不能对单个正类概率省略负类熵项。

### 7.2 参考伪代码

```python
with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
    q = fused_probability.float().clamp(eps, 1.0 - eps)
    y = labels.float()
    wrong = ((q >= 0.5) != (y >= 0.5))
    p_true = y * q + (1.0 - y) * (1.0 - q)
    p_pred = torch.maximum(q, 1.0 - q)
    entropy = -(q * q.log() + (1.0 - q) * (1.0 - q).log())
    lam_a = 1.0 / (1.0 + p_pred - p_true)
    lam = lam_a * p_true * torch.exp(-entropy)
    corrected = lam * q + (1.0 - lam) * y
    q_target = torch.where(wrong, corrected, q).clamp(eps, 1.0 - eps)
```

确定性算例：`y=0,q=0.8` 时 `q_target≈0.06063`；`y=1,q=0.2` 时约为 `0.93937`。正确预测 `y=1,q=0.8` 应保持 `0.8`。`q=0.5` 按正类判定，边界处理必须固定。

错误门控在 0.5 处存在预期的不连续：例如 y=0 时，q 刚低于 0.5 不修正，q=0.5 则修正到 0.125。v1 保留这一 ARM 风格定义，不为消除跳变而自行加平滑门控或更换阈值。

这是对论文 ARM 的二分类改造，不是已验证的青光眼方法。若标签存在噪声，依标签强行修正可能放大噪声；不得因教师不同意标签就删除样本、改标签或人工改测试预测。

## 8. 训练预算与选模规则

| 项目 | E1、E2 固定设置 |
| --- | --- |
| 学生 | `convnext_tiny.fb_in1k`，全量微调 |
| 输入/学生归一化 | 512；ImageNet mean/std |
| 训练监督 | BCEWithLogits，`pos_weight=5` |
| 优化器 | AdamW，all_trainable；仅学生参数 |
| lr / weight decay | 3e-4 / 0.01 |
| betas / eps | `[0.9,0.999]` / 1e-8 |
| 调度 | `warmup_cosine_success_v1` |
| epochs / schedule_total_epochs | 100 / 100 |
| warmup | 5 epoch，从 1e-4 到 3e-4 |
| min_lr | 1e-6 |
| 学生 precision | BF16；不用 FP16 GradScaler |
| 教师 / valid / test precision | FP32 |
| 学生 micro_batch / accumulation | 48 / 16 |
| 有效 batch | 768；最后不足组按真实样本计 |
| eval_batch / workers | 16 / 6，spawn |
| seed / clip | 42 / 5.0 |
| 设备 | 单卡、单进程；用 CUDA_VISIBLE_DEVICES 选物理卡 |
| best 选择 | 学生连续概率 valid AUROC 严格增加 |
| early stopping | min_epochs=30，min_delta=0.0001，patience=15 |
| 提交阈值 | 固定 0.5 |

梯度累积复用现有按真实样本数归一化的语义。每组有 `N_group` 个样本，微批次均值损失乘 `B_micro/N_group` 后反向；不要再除一次 accumulation。每组只做一次 clip、optimizer.step 和成功更新计数。教师分块数不进入 scheduler 更新数。

51,396 个训练样本、micro_batch=48、累积16，对应每轮 1,071 个微批次、67 个更新组；不丢尾批。监督项和软标签项都按相同真实样本计数累积。

学生 `train()` 不得递归切换教师到 train 模式。验证只调用学生 `eval()`，使用真实标签 BCE pw5 与原指标函数，不在 valid 上构造校正软目标或把蒸馏损失作为选模指标。

best_raw 与 early-stopping significant-best 分开，保留原规则：任何 AUROC 严格提升都更新 best；只有提升超过 min_delta 才清零早停 stale；完全相等保留更早轮次，不改用 AP 打破该 checkpoint 并列。

## 9. 日志、学生权重与推理产物

### 9.1 每轮最小记录

在已有 history/epoch report 中扩充字段即可，不额外建设独立审计流水线：

- `epoch, train_total_loss, train_supervised_bce_pw5, train_soft_bce, train_kd_kl_diagnostic`。
- `teacher_raw_entropy_mean, target_entropy_mean`，真实 train 教师融合错误数/比例、按正负标签拆分的错误数。
- E2 的 `corrected_samples, correction_lambda_mean_on_wrong`；当没有错误样本时，lambda 均值为 null。
- 学生 valid BCE pw5、连续概率 AUROC、AP、梯形 PR-AUC、Brier、Balanced Accuracy、Sensitivity、Specificity 及原有阈值指标。
- `lr, successful_updates, consumed_samples, seconds`；可在有界预检记录 CUDA 峰值显存。

各统计按真实样本数加权；每轮 train 消费数严格为 51,396。E1 的 corrected_samples=0，raw 教师错误统计仍要记录，以判断 E2 实际作用于多少样本。

run_manifest 标记 `official_valid_merged=false`、`teachers_frozen=true`、`teacher_target_source=official_train_same_augmented_view`、`student_initialization=ImageNet1K`、`distillation_variant`、`teacher_ensemble=probability_mean`、`temperature=1`、`alpha=1`、`inference_models=student_only`。记录三位教师的路径、模型名、epoch、大小和原预训练来源；教师参数不嵌入学生 checkpoint。

### 9.2 仅保留学生 best 与 last

```text
experiments/<experiment.name>/
├── config_snapshot.yaml
├── run_manifest.json
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── reports/
│   ├── history.csv
│   ├── epoch_XXXX.json
│   ├── metrics.json
│   └── valid_best_metrics.json
├── predictions/best/
│   ├── valid_predictions.csv
│   └── test_predictions.csv
└── submission/
    └── submission.csv
```

`epoch_XXXX.json` 是指标日志，不是逐轮权重。checkpoint 只有学生 `state_dict` 与必要元数据，不存教师 state_dict、optimizer、训练 RNG 或可恢复训练状态；本方案不提供中断续训。

本方案固定使用独立 format 4，`checkpoint_kind=kd_student_inference_weight`，新 loader 严格核对模型、variant、512/pw5、训练配置和元数据。旧 format 3 的含义保持不变；不能让旧 loader 通过跳过严格校验来读取 KD 配置。

最小 payload 字段如下，均沿用现有安全 plain 类型和 `torch.load(weights_only=True)` 的保存/加载方式：

```text
format_version: 4
checkpoint_kind: kd_student_inference_weight
model_name: convnext_tiny.fb_in1k
num_features: 768
timm_version: 1.0.22
pos_weight: 5.0
age_stats: null
epoch: 已完成的正整数轮次
config: 本次完整 KD 配置快照
distillation_variant: standard 或 rectified，与 config 一致
model_state: 仅学生的解包 state_dict
metrics: 本轮学生完整 valid 指标，含有限 roc_auc
model_metadata: 学生模型、初始化与通用预训练来源元数据
teacher_provenance: 三位教师的 ID、模型名、checkpoint 路径、epoch 和归一化信息
```

best 和 last 都保存上述结构；只允许由本次 run 原子替换自己创建的两个文件。加载时核对 payload 与 config 的模型/variant/pos_weight/timm_version、epoch 和 state_dict 严格一致。教师来源只做结构与身份核对，不访问其路径；若需要在运行时检查教师文件，放在训练准备函数，不能放在学生推理 loader 中。

学生 checkpoint 中可以保留教师来源的字符串元数据，但独立推理不得要求三份教师文件存在，也不得根据教师路径构建或加载教师。验证这一点是必要的部署检查。

### 9.3 自动完成训练后流程

1. 释放训练图、优化器和教师，重新严格加载学生 `best.pt`。
2. 对官方 valid 做 FP32 确定性预测，与选中轮的保存结果做既有一致性核对。
3. 学生配置和 best 已选定后，执行一次官方 test 推理，保存 `image_id,logit,probability`。
4. 输出 UTF-8 `submission.csv`，列为 `Image_name,Probability,Pred_Label`；正类概率不取整，标签由 `Probability>=0.5` 得到。
5. 核对 5,855 行、官方 ID 与顺序、无重复/缺失、有限概率且在 `[0,1]`；拒绝覆盖旧文件。

每组完整 run 支持上述训练、valid 回载、test、submission 串行流程；代码实现阶段仅用合成数据验证这些步骤，不据本文启动真实运行或上传官网。

最终推理仅有学生，架构与当前单 ConvNeXt 相同；模型大小预计接近已有约 111.35 MB，实际按文件统计。训练用三个教师不因参与训练而自动计入最终推理文件；必要学生配置等仍按官方第 8 条统计。T4 batch=1 端到端平均≤100 ms 需要真实测量，不因学生结构相同就填 passed。

## 10. 代码实践建议：隔离新增并复用已有接口

2026-10-25 检查时，工作区已有 224/pw10 等相关未提交修改。实现者应保留这些内容，不回滚、不将其混入本次蒸馏方法变化；以届时实际文件内容为接口依据。

建议新增下列文件；这是实现组织建议，不是当前已存在的实现：

| 建议路径 | 职责 |
| --- | --- |
| `src/eye_qing_guang/contracts/convnext_kd_config.py` | 独立 KD 协议及 E1/E2 固定字段校验 |
| `src/eye_qing_guang/contracts/convnext_kd_checkpoints.py` | 学生 format 4 保存与独立加载 |
| `src/eye_qing_guang/models/frozen_teacher_ensemble.py` | 三教师严格加载、冻结、分块、归一化适配 |
| `src/eye_qing_guang/trainers/convnext_kd.py` | 软目标、二分类 ARM、损失与累积训练 |
| `src/eye_qing_guang/pipelines/convnext_kd.py` | train/valid、选模、学生回载和提交 |
| `scripts/run/run_timm_convnext_kd.py` | 单组配置入口、合成预检/正式运行分派 |
| 两个独立 YAML | E1 和 E2 配置 |
| `tests/test_timm_convnext_kd.py` | 本节与第 12 节所需针对性检查 |

复用接口：

- `models/timm_convnext.py::build_convnext_model`：学生从本地 ImageNet 初始化。
- `contracts/convnext_checkpoints.py::load_convnext_checkpoint`、`contracts/timm_checkpoints.py::load_timm_checkpoint`：现有教师严格加载。
- `datasets/convnext_official.py`：官方清单、ID/标签与顺序核对。
- `datasets/loaders.py`、`datasets/transforms.py`、`datasets/offline_cache.py`：学生和数据输入。
- `trainers/convnext_official.py::build_optimizer_schedule`、`SelectionState` 和 `trainers/schedules.py`：优化器、成功更新调度、选模语义。
- `metrics/convnext_official.py`、`evaluators/binary.py`：只评估学生。
- `utils/timm_artifacts.py`：原子写入和 submission 格式检查。

现有 `convnext_config.py` 精确校验字段集合，`utils/timm_source.py::create_source_runtime` 和 loader 的 single_timm 分支识别显式协议集合。新 `protocol=timm_convnext_kd_official_v1` 必须有清晰注册，不能仅增加 YAML 的 distillation 字段就假定旧管线可用。

推荐让新校验器验证完整 KD 字段，再将去除 distillation 的深拷贝映射到已有官方 ConvNeXt 协议做基础配方校验；不要原地改用户配置或放宽旧协议。运行时、cache builder 如需要识别新协议，只添加显式分派并覆盖旧协议回归。

现有基础校验器允许 224/288/512 和 pw5/pw10，部分资源参数也可变化。通过它不等于本次方案已冻结。KD 校验器还须显式约束学生为 `convnext_tiny.fb_in1k`、512/pw5、单卡、BF16、48×16、seed42，以及本文规定的教师顺序、各自 loader、epoch、归一化、alpha/temperature 和训练预算；错误修改不能悄悄成为同名 v1。

官方清单可复用 E0 的 `data/manifests/glaucoma_first_convnext_in1k_official_single3090_cache_v1`，但须通过已有官方来源校验。在现有实现中，`summary.json` 的 protocol 固定为 `timm_convnext_official_trainvalid_v1`，它描述数据清单来源，不等同于新实验的 KD protocol。保留这个历史值，在新 run_manifest 中另记 `experiment.protocol=timm_convnext_kd_official_v1` 和 `manifest_source_protocol`；不要为匹配新实验身份而改写共享清单。缓存签名同样保持原数据配方含义。若运行环境路径不匹配，选择新清单目录，不覆盖原目录。

无需把教师放入 student 的 `nn.Module` 子树；否则容易把教师误传给优化器、调用 train() 或保存进学生权重。

## 11. YAML 增量与运行入口约定

以下是相对已有 512/pw5 主配置的新增字段，不是可直接执行的完整 YAML。代码实践 Thread 应生成完整 E1/E2 配置并通过新校验器。

```yaml
experiment:
  name: timm_convnext_tiny_fb_in1k_512_pw5_kd_ens3_std_official_single3090_s42_v1
  protocol: timm_convnext_kd_official_v1
  seed: 42
  output_root: experiments

distillation:
  variant: standard                 # E2 改为 rectified
  fusion: probability_mean
  temperature: 1.0
  alpha: 1.0
  warmup_epochs: 0
  loss: soft_bce
  correction: none                  # E2 改为 arm_binary_v1
  epsilon: 1.0e-6
  teacher_precision: fp32
  teacher_forward_chunk: 8
  shared_augmented_view: true
  teachers:
    - id: T1
      model_name: convnext_tiny.fb_in1k
      loader: convnext_format3
      checkpoint: experiments/timm_convnext_tiny_fb_in1k_512_pw5_official_single3090_cache_s42_v1/checkpoints/best.pt
      expected_epoch: 7
      weight: 0.3333333333333333
    - id: T2
      model_name: convnext_tiny.fb_in22k_ft_in1k
      loader: convnext_format3
      checkpoint: experiments/timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official_single3090_cache_s42_v1/checkpoints/best.pt
      expected_epoch: 9
      weight: 0.3333333333333333
    - id: T3
      model_name: tf_efficientnetv2_s.in21k_ft_in1k
      loader: timm_v2s_format3
      checkpoint: experiments/timm_v2s_in21k_ft_in1k_512_pw5_official_single3090_cache_s42_v1/checkpoints/best.pt
      expected_epoch: 7
      weight: 0.3333333333333333
```

三个权重检查与 1 的误差≤1e-12，明确等权；variant 与 correction 必须对应。alpha v1 正式配置固定 1；alpha=0 只在测试中用于证明监督路径等价。所有 teachers 字段都应在结构校验中约束，归一化从其已验证的 checkpoint 配置读取并与第 4 节预期值核对。

单组正式运行入口约定如下，命令在实现完成、真实预检通过且运行阶段获授权后执行：

```bash
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

python -u scripts/run/run_timm_convnext_kd.py \
  --project-root "$PWD" \
  --config configs/timm_convnext_tiny_fb_in1k_512_pw5_kd_ens3_std_official_single3090_s42_v1.yaml
```

E2 使用其 rect 配置与独立目录。正式 run 不默认执行完整性能矩阵、额外种子或官方上传。

入口还须支持 `--preflight-only --preflight-output <独立目录> --max-preflight-sample-instances <上限>`。只做第 12.2 节的固定检查并退出；此模式不创建正式实验目录，不进入正式训练或 test。预检目录已存在时同样拒绝覆盖。普通 run 不再隐式调用旧监督管线的 `execute_preflight()` 或执行第二次真实预检。

## 12. 最小验证与执行分阶段

### 12.1 代码实现阶段：离线与合成检查

这一阶段不需要下载权重、读取全量图像或启动真实训练。

1. 配置/导入/语法：两组完整 YAML 都通过；错误 teacher 身份、epoch、weight、variant/correction 组合明确拒绝；原 512/288/224 等受影响配置仍通过原有校验。
2. 纯数学：覆盖 E1/E2 软目标 `[B,1]`、有限范围、正负标签、正确预测、错误预测、0.5 边界、极端概率；核对第 7 节算例及 E1 不修正、E2 正确样本保持不变。
3. 损失/梯度：软 BCE 对学生梯度与 forward Bernoulli KL 一致；教师、软目标无梯度；固定人工 logits 下可核对监督项只有正类乘 5，软目标项没有正类加权。
4. 累积与随机性：人工不等长微批次的样本加权结果等价于完整组计算；alpha=0 的更新与原监督 update_group 在小型模型上等价；E1/E2 构建教师后学生初始化与采样 RNG 一致。
5. 冻结与归一化：玩具教师参数/BN 缓冲不变；学生有有限非零梯度；T3 反归一化再归一化与直接从共同 u 构造等价；教师 chunk 拼接保持样本顺序。
6. 合成端到端：小型模型、合成 records，覆盖若干轮选 best/last、学生独立回载、valid 一致性、test 形式预测与 submission；仅在隔离测试夹具中缩小数据数量和训练预算。
7. 学生独立性：教师文件不可访问时，已保存的学生 checkpoint 仍可加载和前向；加载学生不触发原预训练权重下载。
8. 旧行为：仅对实际触及的公共分派、checkpoint 加载和提交路径运行相关回归；`git diff --check` 通过。不以完整真实训练替代集成检查。

### 12.2 正式训练前：有界真实预检

这一步属于后续执行阶段，代码实现完成后单独启动，输出目录提前选定且独立于正式目录。

- 检查本地四种必要权重来源：三份任务教师 best 与学生 ImageNet 权重；验证官方样本数、标签合法、ID 唯一和 cache 签名。
- 固定取 8 张官方 valid 图像，核对三教师 FP32 前向与各自已保存预测；参考 CSV 按 image_id 对齐，不按行号猜测。概率 `atol=1e-5,rtol=1e-4`，logit `atol=1e-4,rtol=1e-4`；不一致先定位环境、归一化、checkpoint 或路径问题，不自动放宽容差。
- 固定取一个真实学生完整微批次 48 张进行三教师分块前向、BF16 学生前向和反向，验证峰值显存、模式冻结与有限梯度。包含正负样本，只用于运行检查，不以其指标选配方；不调用 optimizer.step。E1/E2 可在这一批共用已经 detached 的教师 q，分别以保留图的 autograd 检查或独立小型合成检查覆盖目标差异，不额外增加真实学生前向。
- 教师 chunk=8。48 张 train 对应 `3×48+48=192` 次按图像计的模型前向，8 张 valid 对应 `3×8=24` 次教师前向，总上限为 **216 model-forward sample instances**，其中仅 48 次学生反向；不同原图共 56 张。预算按模型实际消费的图像实例计，不把它误报为 56 次总前向。
- 基础配置的 `preflight.max_sample_instances=64` 保留旧默认含义，不能直接套用旧 preflight planner 来做此 216 实例检查。新入口必须读取显式 CLI 上限：不足 216 时在任何模型前向前拒绝并说明所需预算；不自动扩容或悄悄缩小学生微批次。实际用量写入该预检输出的简短结果。
- 不执行完整 epoch，不推理 test。预检模型/梯度全部丢弃；正式 run 从冻结 seed 和 ImageNet 权重重新开始。

实现完成后的预检调用示例（一次有界检查，不是本文编写阶段已经执行的命令）：

```bash
python -u scripts/run/run_timm_convnext_kd.py \
  --project-root "$PWD" \
  --config configs/timm_convnext_tiny_fb_in1k_512_pw5_kd_ens3_std_official_single3090_s42_v1.yaml \
  --preflight-only \
  --preflight-output experiments/preflight_convnext_kd_ens3_512_s42_v1 \
  --max-preflight-sample-instances 216
```

### 12.3 完整运行与异常

阶段进度报告包含开始、预检完成、完整训练开始、完成。若失败，立即说明正式训练是否开始、已处理轮次/样本、现存 partial outputs；保留结果并停止，不删除或悄悄在同目录重启。

E1/E2 只按既定 valid AUROC 选 best，teacher 冻结与目标公式在整个正式运行中保持一致。若需要更改 alpha、温度、模型组合或学生 batch，先登记另一个实验版本，不在中途覆盖配置。

## 13. 结果判读与后续决策

比较 E0、E1、E2 在同一官方 valid 上的学生 AUROC 为主，AP、PR-AUC、阈值指标、校准和成本为辅。不要以训练总损失下降或教师错误被修正多少直接认定泛化提升。

| 观察 | 可以得出的结论 | 不能直接得出的结论 |
| --- | --- | --- |
| E1 高于 E0 | 此次监督与软目标的联合配方有帮助 | 收益全部来自教师知识；还需排除第 6.4 节类别平衡变化 |
| E2 高于 E1 | 此次配方支持标签校正的额外价值 | 原论文在青光眼上已完整复现 |
| E1/E2 低于 E0 | 此配方未产生收益；检查目标熵、教师 train 拟合与损失作用 | 应立刻换 test 阈值或增加 epoch |
| 学生接近融合教师 | 有可能用单模型承接部分融合收益 | 已满足 T4 耗时或已超过官网历史最好成绩 |
| 仅 AP 提升 | 阳性排序相关表现有所改善 | 按主指标 AUROC 排名必然更高 |

小差值先报告探索性质。若需量化不确定性，可对保存的同 ID valid 预测做配对、按标签分层的图像级 bootstrap；因缺少可靠患者 ID，不能宣称校正了患者相关性。重复种子实验需要独立安排，不默认增加三倍训练。

v1 只回答“标准冻结教师蒸馏是否有效、标签校正是否额外有效”。CFCL、Transformer 新骨干、蒸馏温度扫描、软目标校准与多种子属于后续独立问题，不混入本轮。

## 14. 交接完成标准

- 两个独立实验配置、KD 训练入口与学生独立推理可用，E1/E2 仅校正设置不同。
- 真实训练范围保持 official train；valid/test 职责明确。
- 三教师身份、同视图归一化、冻结和精度处理正确。
- 监督项 pw5、软目标项不加权、alpha=1、temperature=1、E2 公式及累积语义与本文一致。
- 仅保存学生 best.pt/last.pt；训练后自动回载并生成规定提交文件。
- 代码实践 Thread 汇报修改文件、实际检查及结果、权重与数据准备状态、尚未执行的运行阶段；不把“合成检查通过”写成“真实训练已完成”。

