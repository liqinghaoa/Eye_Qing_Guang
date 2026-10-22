# timm EfficientNet-V2-S 两阶段预训练与联合推理提交实验实现方案 v1

> 2026-10-22 交接复核修订。本文为**新增实验的实现规格**，不是已有代码已经实现的证明。用户已经确定的实验身份与仍需从运行中实验导入的资源/训练参数分开列出；未冻结的必填参数不得用隐式默认值补齐。

> 同日补充：用户已提供 torchvision V2-S 预训练研究对照的 spec_version=0.2 真实快照，冻结时间为2026-10-21T23:15:16+08:00。已据此落实100轮、四卡每卡微批次8/累积4、BF16等已知项；原JSON未包含的优化器与调度实现细节仍明确待查，不再笼统称“缺少配置快照”。

## 1. 文档目的与实验边界

本实验是一个独立的新实验，不覆盖、不改写已有 B0-ResNet50 实验的配置、清单、代码路径和输出目录。可以复用数据加载、图像处理、指标和提交格式化等底层程序，但必须通过新的实验协议和新的输出目录运行。

本版固定回答以下问题：

1. timm 的 tf_efficientnetv2_s.in21k_ft_in1k 两阶段预训练权重，迁移到当前青光眼二分类任务后的表现如何；
2. 在 512 输入、官方 train/valid 划分和 pos_weight=5 条件下，模型是否优于当前 torchvision EfficientNet-V2-S 实验；
3. 训练完成后能否自动使用 best.pt 对官方 test 推理，并生成可直接检查和上传的 submission.csv；
4. 仅保留 best.pt 和 last.pt 两个训练权重文件，避免 B0 实验产生大量 epoch_XXXX.pt 文件。

本版不同时加入 Focal Loss、重采样、AUC Loss、MixStyle、AugMix、年龄融合或 ROI 分支。若本版表现有收益，后续实验再单独登记这些因素。

**比较边界：** 本版同时涉及模型实现/预训练来源、pos_weight，以及可能的划分、分辨率变化，是一个联合方案比较，不能把收益全部归因于 ImageNet-21K。`baseline_evaluation.md` 所记载的三架构实验是从头训练、384 输入，不是用户所说的当前 torchvision ImageNet-1K 预训练实验；两者不能混为同一个参照。若要比较当前运行的 torchvision 实验，须使用其真实配置快照及同一官方 valid 的逐图预测，明确列出不同项，不能拿 B0 的内部开发指标直接相减。

## 2. 最终实验身份

建议实验名称：

~~~text
timm_v2s_in21k_ft_in1k_512_pw5_official_trainvalid_s42_v1
~~~

实验身份必须写入配置快照和运行清单：

| 项目 | 固定值 |
|---|---|
| 实验协议 | timm_official_trainvalid_v1 |
| 模型库 | timm |
| timm 模型名 | tf_efficientnetv2_s.in21k_ft_in1k |
| 预训练来源 | ImageNet-21K 预训练后 ImageNet-1K 微调 |
| 输入尺寸 | 512 × 512 |
| 训练数据 | 官方 train 全部样本 |
| 验证数据 | 官方 valid 全部样本 |
| 测试数据 | 官方 test，只在训练完成后推理 |
| 损失 | BCEWithLogitsLoss |
| pos_weight | 5.0 |
| 主选模指标 | valid AUROC |
| 随机种子 | 42 |
| 使用年龄 | false |
| dropout / drop_path_rate | 本版均固定为 0.0；不得继承 timm 模型或旧配置中未声明的随机正则 |
| 训练预算 | epochs=100，warmup_epochs=5；采用B0已实现的30轮后、min_delta=0.0001、patience=15早停 |
| 四进程批量 | world_size=4，micro_batch=8，accumulation_steps=4，有效batch=128 |
| 训练精度与BN | BF16 autocast，不启用GradScaler；native_local BatchNorm |
| 梯度传输 | 采用本项目已在PPU服务器跑通的原生NCCL兼容DDP；不复现外部gloo_staged |

in21k_ft_in1k 才是两阶段权重。tf_efficientnetv2_s.in21k 只代表 ImageNet-21K 预训练，不能与本版实验名称混用。

## 3. 数据划分与清单

本版不再从官方 train 内部划分新的 train/valid。

~~~text
官方 train：51,396 张，全部用于训练
官方 valid：5,849 张，每轮用于验证和选模
官方 test：5,855 张，只用于训练结束后的无标签推理
~~~

配置必须使用官方三份清单：

~~~yaml
data:
  manifest_dir: data/manifests/glaucoma_first_official_v1
  splits:
    train:
      csv: train.csv
      images: train
      labeled: true
    valid:
      csv: valid.csv
      images: valid
      labeled: true
    test:
      csv: test.csv
      images: test
      labeled: false
~~~

上面的数量依据当前本地官网摘录；若实际官方新版本不同，必须停止核对并登记版本，不静默丢弃或补齐样本。

`data.splits.*.csv` 是 `raw_root` 下的**官方原始 CSV**，不是 `manifest_dir` 中标准化清单。后者必须由原始 CSV 一对一构建，不能从旧 B0 内部分割清单拼接。标准化列固定为 `image_id,image_path,label,age,split`；`image_id` 对应官方 `Image_name`，`split` 仍为 `train/valid/test`，来源 `official_train/official_valid/official_test` 写入 summary 和 run_manifest，不能混淆两个字段。

当前本地 `data/manifests/` 未发现可直接复用的官方版本清单。实现时须明确：不存在则新建本版清单；已存在则只读核对官方 ID、标签、来源与顺序后一致复用，不调用会拒绝已存在目录的 build_manifests 重建，也不覆盖旧清单。

运行前检查：

- train、valid、test 的图像标识没有交集；
- train 和 valid 都包含标签 0 和 1；
- test 没有读取标签列；
- 图像数量与官方清单一致；
- 推理输出的 test 标识集合和顺序与官方 test CSV 完全一致。

训练使用自然类别比例的 shuffle，不启用 WeightedRandomSampler，不复制阳性样本；`drop_last=false`，每轮所有训练样本恰好消费一次。坏图、缺图或标签非法要报错，不得静默跳过。清单排重检查用大小写规范化键，但对外输出保留官方原始文件名、扩展名与顺序，不自行去扩展名或重命名。test 的 label 在标准化清单中必须为空；即使原始表意外出现同名标签列也不读取为真值。

训练阶段可以加载 test 的无标签路径用于最终阶段配置检查，但不能读取 test 标签，也不能将 test 概率、文件名或任何预测结果用于训练、早停、超参数选择或阈值选择。

## 4. 新增代码边界

### 4.1 必须新增的文件或入口

建议新增以下文件：

~~~text
src/eye_qing_guang/models/timm_efficientnet.py
src/eye_qing_guang/pipelines/timm_official.py
scripts/run/run_timm_v2s_experiment.py
configs/timm_v2s_in21k_ft_in1k_512_pw5_official.yaml
~~~

如当前项目没有 scripts/run/ 目录，可以新建；本次交接统一采用 `scripts/run/run_timm_v2s_experiment.py`，后续命令以此为准。除这些入口外，为新协议新增配置校验、权重读写和测试文件属于正常实现范围；不要为了凑齐目录而复制整个 B0 管线。

### 4.2 可以复用的程序

可以复用以下程序的底层功能：

- datasets/metadata.py：官方清单读取和基础元数据校验；
- datasets/loaders.py：训练、验证、测试 DataLoader；
- datasets/transforms.py：FOV 检测、letterbox、确定性验证/推理变换；
- evaluators/binary.py：批量前向和概率输出；
- metrics/binary_classification.py：二分类指标；
- pipelines/inference.py 中的提交表格校验逻辑；
- utils/experiment_io.py：运行清单、CSV、JSON、YAML 输出；
- utils/reproducibility.py：随机种子和设备选择。

### 4.3 不得直接复用的 B0 编排路径

新版不得直接调用：

- pipelines/b0_development.py；
- pipelines/b0_distributed.py；
- datasets/development.py 的官方 train 内部划分逻辑；
- B0 专用的 is_b0 约束；
- b0_dev_v1 或 b0_dev_ddp_v1 协议。

可以复用低层训练函数，但官方 train/valid 的清单读取、选模和输出编排必须由新的 timm_official 流程负责。不得为了支持新版而改变 B0 配置的身份约束或既有输出目录。

### 4.4 当前工程的接口缺口，不能只新增 YAML

2026-10-22 本地核查发现：`contracts/config.py` 只接受旧 backbone 和 B0 协议，`models/factory.py` 只按 `model.backbone` 分派，`contracts/checkpoints.py` 只识别 format_version 1/2 且保存时强制包含 optimizer_state。故本版配置不能直接传给这些旧入口后声称已经兼容。

实现时须新增本协议自己的配置校验器与检查点读写器，或在公共入口增加显式协议分派且保持旧分支不变；不能放宽 B0 身份校验来容纳 timm。新增推理权重格式使用 `format_version=3`、`checkpoint_kind=inference_weight`，由新加载器识别，旧 format 1/2 的解释保持原样。恢复本版权重时使用 `weights_only=True`、CPU map_location、`strict=True`，载荷只保存普通字典、字符串、数值和 Tensor，不保存 nn.Module 或设备专属对象。

当前 `evaluators/binary.py::predict_batches` 调用 `model(images, ages)`，而非 `model(images)`。新模型 wrapper 必须支持 `forward(images, ages=None)` 并忽略 ages，保持 `use_age=false`；输出 `[B,1]`。`FundusDataset` 的 label 也为 `[B,1]` 浮点张量，不能让 `[B]` 与 `[B,1]` 广播。

当前指标函数未实现梯形 PR-AUC、高特异度召回或硬标签 AUROC，不能只在报告中列字段却不计算。仅为新协议补充指标，或保持原函数返回语义后显式扩展。

`pyproject.toml` 当前没有 timm 依赖。先在现有训练环境中核对 PyTorch/torchvision/PPU 适配版本，再安装并固定可用的 timm 版本；保存精确版本，不自动升级或替换服务器专用 torch、torchvision、SDK。预训练权重解析失败时明确报错，不回退到随机初始化、in1k 或其他名称，也不修改旧环境配置作为隐式修复。

实现路线更新（2026-10-22）：用户明确选择基于项目中已完成服务器验证的B0训练框架完成本实验。虽然另一份实验快照记有 `gradient_transport=gloo_staged`，当前项目未提供该引擎；本实验不以它为依赖，也不声称复现它。训练使用现有 `utils/distributed.py` 的四卡NCCL兼容DDP、DDP模型封装、`no_sync` 梯度累积、native local BN、spawn数据加载器和成功更新调度。该选择保留新模型、官方划分、pw5、BF16和独立产物。

## 5. 模型实现

### 5.1 timm 模型构造

~~~python
import timm
from torch import nn

encoder = timm.create_model(
    "tf_efficientnetv2_s.in21k_ft_in1k",
    pretrained=True,
    num_classes=0,
    global_pool="avg",
    drop_rate=0.0,
    drop_path_rate=0.0,
)
features = encoder.num_features
classifier = nn.Linear(features, 1)
~~~

以上是构造核心，不是完整 wrapper。可以将 encoder 及其 num_features 传给已有 `ImageClassifier(encoder, features, dropout=0.0)`，使 `forward(images, ages=None)` 兼容公共评估器。num_classes=0 用于移除 ImageNet 分类头，只保留图像编码器。该官方权重配置的特征维度为 1280；实现时读取并检查 `encoder.num_features`，不通过截断原 ImageNet 输出代替特征提取。最终输出必须是 [batch, 1] 的单个 logit，模型内不加 sigmoid；分类头随机初始化，**编码器与分类头均参与全量微调**，本版不冻结骨干、不额外增加 head-only warmup。

每次从头启动训练时从同一份预训练权重初始化编码器；预检模型不直接接着训练，正式模型须重新初始化，可复用本地权重缓存。重新加载 best.pt、last.pt 或执行测试推理时，必须使用 pretrained=False 构建同一结构，再加载检查点中的 model_state，不能再次下载或覆盖训练权重。DDP 多进程共享缓存需先准备就绪再同步加载，避免各 rank 同时下载和写同一文件。

运行清单必须记录：

- timm 版本；
- PyTorch 版本；
- 模型完整名称；
- 预训练配置中的 URL 或 Hugging Face 标识；
- 若通过本地文件加载，记录实际文件路径和大小；通过 timm/Hugging Face 缓存加载时记录 repo/tag/revision 或可解析的缓存标识，不虚构本地路径；
- num_classes=0、global_pool 和分类头特征维度。

### 5.2 输入与归一化

本版固定输入为 512 × 512。保留当前工程的 FOV、等比缩放、letterbox 和训练增强流程，不自动改成 timm 的 ImageNet 中心裁剪流程。

2026-10-22 读取该权重仓库的官方 config.json：`input_size=[3,300,300]`、`test_input_size=[3,384,384]`、`fixed_input_size=false`、`num_features=1280`、`mean/std=[0.5,0.5,0.5]`、`interpolation=bicubic`。因此“原本统一固定为384”不准确，300和384是该配置的不同推荐入口；本实验明确覆盖为512，不自动使用默认 data transform。正式运行仍需核对已安装 timm 解析出的 pretrained_cfg，记录后若 mean/std 或模型结构与本规格不符则报错核对，而非静默换值。

本版 mean/std 显式写入配置；图像解码为 RGB，ToTensor 将原始像素映射至 [0,1] 后执行 `(x-0.5)/0.5`。沿用工程现有 **BILINEAR** FOV 预览、旋转、等比缩放与黑色居中 letterbox，而不是偷偷改成权重仓库的 bicubic/中心裁剪。该差异须在报告中说明，不声称完全复刻原始 ImageNet 评测配方。完整 FOV 参数见第9节，缺少任一参数都应在加载配置时报错。

训练时使用随机增强；验证和测试推理使用确定性 FOV、letterbox 和归一化，关闭随机翻转、旋转、亮度和对比度增强。

依据：timm 官方权重仓库 `timm/tf_efficientnetv2_s.in21k_ft_in1k` 的 [config.json](https://huggingface.co/timm/tf_efficientnetv2_s.in21k_ft_in1k/blob/main/config.json)。此处只核对元数据，未下载模型权重。

## 6. 损失函数与选模

### 6.1 pos_weight=5

损失固定为：

~~~python
criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([5.0], dtype=torch.float32, device=device),
    reduction="mean",
)
~~~

不能使用自动按类别比例计算的 pos_weight，也不能在本版同时启用 Focal Loss 或重采样。

logits 和 labels 都为 `[B,1]`，labels 是 float32 的0/1；在 FP32 中计算 BCE。此处5.0表示**仅正类损失项乘5**，不等于正样本率5%、不等于每张样本权重都乘5；不再叠加另一个 class_weight。验证 loss 同样按加权 BCE 报告并明确命名为 `weighted_bce_pw5`，跨 batch 按实际样本数汇总，不能平均各 batch 均值而忽略尾批。

`probability=sigmoid(logit)` 不做除5或其他隐式反校准。加权 BCE 会改变工作点，其输出不应自动解释为已校准的临床患病概率；Brier 是描述性结果，不承诺一定随 AUROC 改善。提交阈值本版固定0.5，不能事后把高特异度探索阈值自动用于 submission。

训练保存的配置和运行清单必须同时记录：

~~~text
loss = bce
pos_weight = 5.0
auto_pos_weight = false
focal_gamma = unused_for_this_run
~~~

### 6.2 每轮评估

每个完成轮次都对官方 valid 做确定性推理，至少记录：

- validation loss；
- AUROC；
- Average Precision；
- 梯形 PR-AUC；
- Brier score；
- Accuracy；
- Balanced Accuracy；
- F1；
- Precision；
- Sensitivity；
- Specificity；
- TP、TN、FP、FN；
- 固定0.5生成标签后的硬标签 AUROC（本验证集同时有两类时应与 Balanced Accuracy 一致）；
- Recall@Specificity≥90%；
- Recall@Specificity≥95%。

模型选择使用 validation AUROC 最大的轮次。验证集阈值 0.5 只用于报告固定阈值指标，不作为唯一选模依据。高特异度工作点应通过验证集排序重新寻找阈值，并同时记录实际特异度和阈值。

计算口径固定如下：

- 先完整收集官方 valid 的 `(image_id,label,logit,probability)`，核对全量唯一 ID 和原始顺序，再计算指标；不能对各 batch 或各 rank 的 AUROC 求平均。
- valid 与 test 都采用 `eval()`、`torch.inference_mode()`、FP32 前向及 FP32 sigmoid。选模 AUROC 对连续 probability 计算，不对硬标签计算；保留原始 logit 供诊断，不中途切换选模列。
- AP 使用 `average_precision_score`；梯形 PR-AUC 对 `precision_recall_curve` 的实际点用 `auc(recall, precision)` 积分，字段命名 `pr_auc_trapezoid`，不得直接对递减 recall 调用有符号积分得到负值。二者均不冒充已确认的官方 AUPRC 算法。
- Recall@Spec 在全量 valid 的实际唯一分数阈值上寻找满足约束的最大 sensitivity，不插值；相同分数样本必须整体处理，预测条件仍为 `p>=threshold`。并列解优先实际 specificity 更高，其次阈值更高；允许“全部阴性”端点。报告 threshold、实际 sensitivity/specificity 和混淆计数；端点非有限阈值用明文状态或 null 表示，不在 JSON 写 Infinity。
- `roc_auc=None/NaN`、缺失样本或非有限预测属于验证失败，不保存其为 best，不假定为0分继续跑。

官方答疑曾称使用 Pred_Label，与连续分数 AUROC 及现有60.29案例仍有未解决矛盾。故同时报告硬标签与连续分数指标，不自造“官网模拟总分”，也不在本版改变已经确定的 AUROC 选模和0.5提交阈值。规则来源见 [提交作品文档第四章](../01_Background_and_Literature/大数据挑战赛%20-%20提交作品.md#四官方答疑要点与评分待确认问题)。

### 6.3 早停

本版沿用已确定的 `epochs=100`、`warmup_epochs=5`、`lr=0.0003`、`min_lr=1e-6`。缺失的早停细节按本项目已验证的B0实现明确选为启用：`min_epochs=30`、`min_delta=0.0001`、`patience=15`。这是本实验配方选择，不表示外部快照包含了这些值。

为消除 best 与 min_delta 的冲突，状态分开维护：

1. `best_raw=-inf`：第一轮有限 AUROC 必保存；之后只要 `score>best_raw` 就更新 best.pt，完全相等则保留较早轮次，不用 AP 打破 best 并列。
2. 仅启用早停时维护 `best_significant=-inf, stale=0`：`score>best_significant+min_delta` 才更新该参考并清零 stale，否则 stale 加1。stale 从首轮评估后累积，不在 min_epochs 重置；`epoch>=min_epochs and stale>=patience` 时停止。epoch 从1开始。

优化器、学习率、warmup/cosine、物理 batch、累积、精度、workers 也需显式落地。只以 `training.pos_weight=5` 声明而不把其传入 criterion，或沿用只识别 auto_pos_weight 的旧分支，均不算实现。

早停只使用官方 valid 的 AUROC。训练结束后记录 best_epoch、best_valid_auroc、最后一轮编号、实际完成轮数和触发停止的原因。

## 7. 检查点保存策略

### 7.1 最终文件要求

新实验的 checkpoints/ 目录最终只能保留：

~~~text
checkpoints/best.pt
checkpoints/last.pt
~~~

不得生成或保留：

~~~text
epoch_0001.pt
epoch_0002.pt
epoch_XXXX.pt
best_inference.pt
~~~

训练日志、逐轮 JSON、曲线和预测 CSV 可以保留；本要求只限制权重文件。

### 7.2 best.pt

当 validation AUROC 严格改善时，直接原子替换 best.pt，规则见6.3节。使用同目录临时文件写完、关闭后原子替换；只替换当前新 run 自己创建的这两个文件，不触碰其他实验。现有 save_checkpoint 使用 `open("xb")`，不能原样拿它反复写 best.pt，否则第二次更新即失败。

best.pt 至少包含：

~~~text
format_version = 3
checkpoint_kind = inference_weight
epoch
config
age_stats = null
pos_weight = 5.0
model_state
metrics
timm_version（普通字符串）
model_name = tf_efficientnetv2_s.in21k_ft_in1k
num_features = 1280
~~~

保存解包后的模型 state_dict，键不能带 DDP 额外的 `module.` 前缀；建议 Tensor 转为 CPU 后序列化。配置、版本、路径等元数据须转为安全的普通 Python 类型，例如 `str(torch.__version__)`，保证 `weights_only=True` 可读。

**本版明确选择仅保存推理权重，而不是恢复状态。** 因而不保存 optimizer、scheduler、AMP scaler 或 RNG，best.pt 和 last.pt 都不支持续训，CLI 应拒绝 `--resume-from`，不能只加载权重后称为恢复训练。用户“只留两个文件”的要求本身并不禁止完整训练状态；此处是不扩展续训功能的实现取舍。未来如需续训可另定协议，但本轮不新增此功能。

### 7.3 last.pt

每轮训练及其完整 valid 评估成功后，原子替换 last.pt；正常结束/早停时无需额外建立别名。其载荷格式与 best.pt 相同，epoch/metrics 对应该轮。last.pt 表示最后一个**完整完成训练与验证**的轮次，不一定是最好轮；中途未完成轮不覆盖它。

正常完成时必须只剩两个正式 `.pt`，临时文件处理完毕；失败时可能只有已成功落盘的一个文件或两个文件，不为满足数量伪造权重。best 与 last 同轮时允许两个文件内容等价，不需要另存第三份部署权重。

如果训练中途失败：

- 不伪造 completed 状态；
- 保留已写入的 best.pt 或 last.pt；
- 在 run_manifest.json 中记录失败阶段、已完成轮数和错误信息；
- 不生成不完整的权重别名。

## 8. 训练后自动推理与提交

### 8.1 run 程序的阶段顺序

run_timm_v2s_experiment.py 的单次运行顺序固定为：

~~~text
配置和环境校验
→ 官方 train/valid/test 清单校验
→ 模型与预训练权重加载
→ 代表性 batch 前向/反向预检
→ 重新构造正式模型、优化器和随机状态（预检更新不得带入正式训练）
→ 完整训练
→ 选择并保存 best.pt、last.pt
→ 用 best.pt 重新加载并评估官方 valid
→ 核对重载预测与该 best 轮保存的 valid 预测一致
→ 用 best.pt 推理官方 test
→ 生成 test_predictions.csv
→ 生成 submission.csv
→ 写入最终 run_manifest.json
~~~

训练阶段不能使用 test 预测。只有训练成功、best.pt 存在且能重新加载后，才能进入 test 推理和提交生成阶段。

配置与源清单读取的轻量检查可在此串联流程内自动运行。有限前后向预检见11.1节，独立输出且不能消耗正式目录；若 run 启动前已经有同一冻结配置的合格预检记录，可复用，不重复真实数据预检。重置随机种子、模型/BN、优化器、scheduler、scaler 与加载器 generator 后才算从头正式训练。

重新评估 best 时，全量有效 ID/labels/概率必须对齐，FP32 logits/probabilities 在预先规定的数值容差内与 best 轮结果相符；建议 `atol=1e-6, rtol=1e-5` 并记录最大差。若同一硬件下出现超差、指标变化或严格载入失败，停止 test 阶段并排查，不另选一个轮次绕过问题。

### 8.2 预测输出

建议输出目录：

~~~text
experiments/<experiment_name>/
├─ config_snapshot.yaml
├─ run_manifest.json
├─ checkpoints/
│  ├─ best.pt
│  └─ last.pt
├─ reports/
│  ├─ history.csv
│  ├─ metrics.json
│  ├─ valid_best_metrics.json
│  └─ epoch_XXXX.json
├─ predictions/
│  └─ best/
│     ├─ valid_predictions.csv
│     └─ test_predictions.csv
└─ submission/
   └─ submission.csv
~~~

test_predictions.csv 至少包含：

~~~text
image_id,logit,probability
~~~

`valid_predictions.csv` 另外包含 label，并按官方 valid 原始 CSV 排序。保存 best 更新当轮的 valid 结果以便回载核对；在本 run 内原子替换该 best 预测文件，不累计每轮完整预测文件。history/epoch 报告可以追加或独立写入，但不能复制 epoch 权重。

推理必须使用 best.pt，因为它是按官方 valid AUROC 选择的模型。last.pt 只作为最后轮权重保留，不默认用于提交。

### 8.3 submission.csv

最终文件必须固定为：

~~~text
Image_name,Probability,Pred_Label
~~~

要求：

- 文件名为 submission.csv；
- UTF-8 编码，表头严格三列且不写 DataFrame index；
- 行数与官方 test CSV 一致；
- 行顺序完全按照官方 test CSV；
- Image_name 集合完全一致；
- Probability 为有限值，范围为 [0, 1]；
- Pred_Label 使用配置阈值生成，默认 Probability >= 0.5 为 1；
- 不用 Pred_Label 替换 Probability；
- 如果目标文件已存在，拒绝覆盖并要求使用新的实验目录。

优先复用 `build_submission`：它通过 `image_id` 键映射回官方 test 原始行序，不能直接按两个表当前行号拼接。内部预测中的 probability 与最终 Probability 均保留足够精度（例如 float32 的9位有效数字），不先四舍五入为2位小数；序列化再读回后，按同一 `>=0.5` 规则生成的 Pred_Label 必须一致。还需检查 Pred_Label 确为整数0/1，而非字符串布尔值。

`test_predictions.csv` 生成前先核对无重复/缺失 ID，再按官方顺序重排；最终三列文件只是其确定性转换。禁止通过人工修改测试标签或改变 test BatchNorm 统计量提高分数；不启用 TTA、测试时训练或在线校准。

由于官网评分细则仍需以最新答疑和平台行为为准，本地必须保留完整概率列，不能只保存二值标签。

### 8.4 阶段状态与失败边界

run_manifest 至少记录 `training`、`best_validation`、`test_inference`、`submission` 的 `pending/running/completed/failed/not_started` 状态、epoch、样本数和错误摘要；只有四阶段均成功才写总状态 `completed`。训练失败时 test/submission 必须为 not_started；推理失败时保留已完成训练权重和报告，不自动重新训练、不改名覆盖旧文件。

写 submission 前先在内存校验，再写当前 run 自己的临时文件，读回校验通过后在**最终目标不存在**的前提下发布；校验失败不留下看起来可提交的最终文件。本版不自动恢复失败管线，也不自动上传。代码实践 Thread 如需要单独补跑已完成训练的推理入口，另行说明只写缺失的新产物且不重训的范围，不擅自扩大本轮接口。

## 9. 配置契约与源快照映射

### 9.1 已提供的真实源快照与继承边界

用户于2026-10-22在对话中提供 torchvision V2-S 的JSON，`spec_version=0.2`，`stage=phase3_pretrained_research`，`frozen=true`，源冻结时间为2026-10-21T23:15:16+08:00。这是ImageNet-1K预训练研究对照，不能与384从头训练三架构记录混淆。

| 参考快照字段 | 本版处理 | 状态/解释 |
|---|---|---|
| seed=42 | experiment.seed=42 | 继承 |
| epochs=100 | training.epochs=100 | 继承预算；早停是否启用仍需查源代码 |
| warmup_epochs=5 | training.warmup_epochs=5 | 继承时长，不代表已知warmup函数或起始学习率 |
| lr=0.0003 / min_lr=1e-6 | learning_rate=0.0003 / min_lr=1e-6 | 不按分辨率变化自动缩放学习率 |
| weight_decay=0.01 | training.weight_decay=0.01 | 继承；优化器及衰减参数组规则还需读取实现 |
| world_size=4 / micro_batch=8 | world_size=4 / batch_size=8 | batch_size是每卡物理微批次 |
| effective_batch=128 | accumulation_steps=4 | 由128/(4×8)推导；尾组不足128时按实际样本数处理 |
| precision=bf16 | precision=bf16 / amp=true | 不使用GradScaler，不静默回退FP16/FP32 |
| batchnorm=native_local | model.batchnorm=native_local | 各卡本地BN；不切换SyncBN或冻结BN |
| workers=4 | data.num_workers=4 | 每rank4个worker，共16个，不是全局4个 |
| cpus_per_rank=6 | distributed.cpus_per_rank=6 | 资源分配意图，共24核；不等于num_workers，不凭此认定CPU亲和性已设置 |
| gradient_transport=gloo_staged | distributed.gradient_transport=native_ddp | 用户随后选择复用已验证B0四卡框架；本版不复现该外部传输路线 |
| expected_valid_count=5849 | evaluation.expected_valid_count=5849 | 除数量外仍须核对ID和标签，不能只按数量认定同一验证集 |
| duplicate_logit_atol=1e-6 | evaluation.duplicate_logit_atol=1e-6 | 继承诊断容差；不允许利用容差静默去重验证样本 |
| image_size=384 | data.image_size=512 | 本版明确改变 |
| torchvision ImageNet-1K / ImageNet mean/std | timm in21k_ft_in1k / mean=std=[0.5,0.5,0.5] | 权重、模型实现与归一化同时改变 |

参考JSON未给出的优化器、beta/eps、裁剪、warmup端点、调度和早停参数，本版明确选用项目已实现并用于B0 PPU训练的配方：AdamW、betas=(0.9,0.999)、eps=1e-8、全参数统一参数组、裁剪5.0、successful-update warmup/cosine、warmup起点1e-4及30/0.0001/15早停。此处记录为本实验的新实现选择，不宣称这些数值来自参考JSON。损失、pw5、dropout=0等来自本版协议。通信改用既有B0 DDP路径。

为帮助代码实践 Thread 定位对应实现，保留用户提供的既有来源标识（**仅转录，未重新计算或核验**）：

~~~text
implementation_sha256 = c933c9128ac1b88781536948382eeffa1782bf3d386701bd98a66cfd1cc70b2b
pretrained_adapter_sha256 = 28dd7df20696c0047d12cc5599a76c8430e6e99abd8c0642d1fc4f308e5b3a18
source weight_enum = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
source weight_path = models/pretrained/efficientnet_v2_s-dd5fe13b.pth
~~~

上述torchvision权重路径只是**对照来源**，不得加载进timm新模型。JSON列出的archive/technical_checks及reports/acceptance路径是源实验证据，本地本轮未检索到对应引擎记录，不能声称已复核，更不能把其BF16/通信验收直接继承为新timm-512模型已验收。

`max_benchmark_updates=200`、`max_correction_runs=2`、`baseline_runs=3`是源实验管理字段，不是本版自动开展200次更新预检、两次修正或三组实验的授权。forecast约180.29秒/轮的四卡V2-S值来自源方案预测，不用于宣称本版512训练耗时。源JSON的competition_eligibility=pending_official_clarification是历史状态，原样记录；不自动覆盖为新实验已合规，也不据此抹掉提交作品文档4.4节记录的官方允许预训练答复，最终材料分别注明规则依据和待验收事项。

### 9.2 实现规格模板

以下仍**不是可直接启动的最终 YAML**：已知值已落实，`null`只表示源JSON未提供且仍需从对应引擎读取的项，不允许隐式默认。`recipe.source_config`在本例是来源标识而非磁盘文件路径；代码实践 Thread 找到实际源文件后可补记其路径，不要求用户重复寻找整个快照。无法取得缺项时可完成代码和合成测试，但要精确报告哪些实现参数阻止正式启动。

`raw_root` 的本地有效路径已核对为 `data/raw/glaucoma_first`（相对代码项目根）；服务器如为 `/data/workspace/KanglongTan/data/glaucoma_first`，在正式配置中显式改写并记录，不把服务器绝对路径当成本地通用路径。

~~~yaml
experiment:
  name: timm_v2s_in21k_ft_in1k_512_pw5_official_trainvalid_s42_v1
  protocol: timm_official_trainvalid_v1
  seed: 42
  output_root: experiments

recipe:
  source_experiment: 'phase3_pretrained_research:efficientnet_v2_s'
  source_config: user_supplied_spec_v0.2_2026-10-22
  source_frozen_at: '2026-10-21T23:15:16+08:00'

data:
  raw_root: data/raw/glaucoma_first
  manifest_dir: data/manifests/glaucoma_first_official_v1
  image_column: Image_name
  label_column: True_Label
  age_column: Age
  splits:
    train: {csv: train.csv, images: train, labeled: true}
    valid: {csv: valid.csv, images: valid, labeled: true}
    test: {csv: test.csv, images: test, labeled: false}
  image_size: 512
  num_workers: 4
  mean: [0.5, 0.5, 0.5]
  std: [0.5, 0.5, 0.5]
  preprocessing:
    version: fov_letterbox_v1
    preview_size: 512
    threshold: 8
    min_line_fraction: 0.01
    margin: 0.02
    min_area: 0.5
    min_coverage: 0.995
  augmentation:
    horizontal_flip: 0.5
    rotation_degrees: 10
    brightness: [0.9, 1.1]
    contrast: [0.9, 1.1]

model:
  library: timm
  name: tf_efficientnetv2_s.in21k_ft_in1k
  pretrained: true
  dropout: 0.0
  drop_path_rate: 0.0
  batchnorm: native_local
  use_age: false

training:
  optimizer: adamw
  optimizer_parameter_groups: all_trainable
  epochs: 100
  batch_size: 8
  eval_batch_size: 128
  accumulation_steps: 4
  learning_rate: 0.0003
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1.0e-8
  loss: bce
  pos_weight: 5.0
  auto_pos_weight: false
  device: cuda
  precision: bf16
  amp: true
  grad_clip: 5.0
  max_consecutive_skips: null
  scheduler: warmup_cosine_success_v1
  schedule_total_epochs: 100
  warmup_epochs: 5
  warmup_start_lr: 0.0001
  min_lr: 1.0e-6
  monitor: roc_auc
  early_stopping_enabled: true
  min_epochs: 30
  min_delta: 0.0001
  patience: 15

distributed:
  enabled: true
  world_size: 4
  gradient_transport: native_ddp
  cpus_per_rank: 6
  backend: nccl
  timeout_seconds: 900
  multiprocessing_context: spawn
  log_every_n_groups: 10

evaluation:
  threshold: 0.5
  precision: fp32
  expected_valid_count: 5849
  duplicate_logit_atol: 1.0e-6

submission:
  mode: competition
  id_column: Image_name
  value_column: Probability
  label_column: Pred_Label
~~~

配置解析器必须拒绝 backbone、name、library、pos_weight 等字段的歧义组合。新版不应把 tf_efficientnetv2_s.in21k_ft_in1k 映射成旧的 efficientnet_b0 或 torchvision V2-S 名称。

本协议只使用 `model.library/name`，不再额外传入旧 `model.backbone`。配置校验还须落实：

- AdamW + warmup/cosine继承本项目B0已经实现的优化器和成功更新调度；显式使用betas=(0.9,0.999)、eps=1e-8、统一可训练参数组、warmup起点1e-4、weight decay=0.01。此处为本版冻结的新配方选择，不归称为外部gloo_staged源快照参数。
- 梯度裁剪按本项目B0配置设为5.0；BF16不使用GradScaler或溢出跳步，max_consecutive_skips为null/不适用。
- batch_size 为**每卡物理微批次**，`B_effective=world_size*batch_size*accumulation_steps`；eval_batch_size 也为每卡批次，workers 为每个训练进程的 worker 数，不能把全局 batch 复制为每卡 batch。512 相对384的资源需求不同，如需减小微批次/增大累积，须先登记新配方而非 OOM 后自动更改；BN 统计未必与原物理 batch 等价。
- 尾批/尾累积组按实际样本数归一化；B0 DDP在累积组前向期间采用no_sync，仅最终微批触发一次DDP梯度平均，不再另做all_reduce或重复除以world_size。四rank每轮样本与尾批必须保持一致，日志计数使用真实消费样本数。
- `precision` 只能显式为 fp32/fp16/bf16；fp32 对应 amp=false，fp16/bf16 对应 amp=true，只有 fp16 使用 GradScaler。目标设备不支持即失败，不自动降级。验证和提交推理始终 fp32。
- 使用 `warmup_cosine_success_v1`；scheduler 仅在成功optimizer更新后推进。BF16无GradScaler且无溢出跳步；非有限梯度直接使当前运行失败。
- `early_stopping_enabled=false` 时将 min_epochs/min_delta/patience 明确设为 null 并记录不使用；启用时三项必须为有效值，满足 `1<=min_epochs<=epochs`、`min_delta>=0`、`patience>=1`。除这种明确不适用的情况外，必填项为 null 必须在模型加载/训练前拒绝。
- 本版正式配置固定四rank；不要因目标机器暂时缺卡自动改成单卡。复用B0的`backend=nccl`、900秒timeout和spawn数据进程；启动器仍为外部torchrun。当前不补齐训练要求 `51396 % 4 == 0`，每rank12,849张，不允许悄悄复制/丢弃样本。

在最终 config_snapshot 中，以上有效字段都必须已解析为具体值。该快照是重载模型的来源，不在推理时读取一个后来改过的 YAML。

## 10. 命令接口建议

下面命令是**代码实现完成、训练配方冻结后**的调用接口，不表示本轮授权执行。PowerShell 先将工作目录切换到代码项目根，再运行脚本；`--project-root` 不会让 Python 自动找到当前目录之外的脚本。

~~~powershell
Set-Location -LiteralPath 'E:\projects\Eye_Qing_Guang\Eye_Qing_Guang_code'
~~~

脚本至少支持以下两个入口：

~~~powershell
# 只做有限预检，不启动完整训练
python scripts/run/run_timm_v2s_experiment.py --project-root E:\projects\Eye_Qing_Guang\Eye_Qing_Guang_code --config configs/timm_v2s_in21k_ft_in1k_512_pw5_official.yaml --preflight-only
~~~

~~~powershell
# 完整训练、best/last 保存、官方 test 推理和 submission.csv 生成
python scripts/run/run_timm_v2s_experiment.py --project-root E:\projects\Eye_Qing_Guang\Eye_Qing_Guang_code --config configs/timm_v2s_in21k_ft_in1k_512_pw5_official.yaml
~~~

完整 run 必须在新的实验目录不存在时才启动。不得自动覆盖已有实验目录或已有 submission.csv。

路径解析统一锚定 `--project-root`：config 相对路径、output_root、manifest_dir、raw_root 和清单中的相对 image_path 均按该根解析，不能依赖调用 shell 的偶然工作目录。现有版本名已存在时，先选定新的显式 experiment.name（如后缀v2）并冻结后再启动，不自动追加时间戳改变输出目录。

`--preflight-only` 不创建正式 `<experiment_name>/`，而使用独立且预先确定的 `experiments/_preflight/<experiment_name>_pf_v1/`；若已存在则停止，显式选择新预检版本。默认完整 run 可先复用同一冻结配置已有的合格预检记录；没有时在该独立预检目录执行后再创建正式 run。

本版复用项目B0的原生NCCL兼容DDP训练路线，不使用`gloo_staged`。torchrun读取RANK/LOCAL_RANK/WORLD_SIZE；DDP负责参数初始化和梯度平均，累积时使用no_sync，验证以无补齐分片调用裸模型。外层进程组通信与DDP实现均复用项目已经验证的代码。预检仍需验证当前PPU镜像、timm模型算子和本实验batch。

四卡接口示例（Linux服务器、已进入代码项目根，权重/路径已准备且目标设备预检完成）：

~~~bash
torchrun --standalone --nproc_per_node=4 scripts/run/run_timm_v2s_experiment.py --project-root /实际代码项目绝对路径 --config configs/timm_v2s_in21k_ft_in1k_512_pw5_official.yaml
~~~

不能套用 B0-dev 的内部数据划分、身份校验或输出协议；四rank训练通信与基础设施明确复用B0原生DDP实现。训练须满足：

- 训练 DistributedSampler 每轮 set_epoch；每 rank 消费 N/W 样本且不补齐/不丢弃，全局一轮为51,396。
- 训练保持native_local BN，不转SyncBN或将各rank局部BN方差简单平均。验证使用无补齐索引分片（如rank::W），各rank分片长度可不同。验证前将rank0的参数与BN缓冲区广播到各rank；本协议最终导出rank0模型与BN缓冲区，用该同一状态对全量valid计算选模指标并保存。若分片验证，先将这一状态同步到各rank再用非通信模型预测，不对不等长分片执行DDP forward；汇总后由rank0恢复全量ID顺序，并在验证后恢复各rank训练时的本地BN缓冲区。跨rank同步仅用于统一评估状态，不能偷偷改变既定native_local训练语义。
- rank0 将 AUROC、best/早停判定广播给所有 rank，统一继续或停止；所有 rank 的集合通信顺序一致。只有 rank0 写检查点、清单、报告和提交，I/O失败须通知其他 rank 退出，不能让其他进程无限等待。
- 训练结束统一完成同步并退出训练进程组。其余 rank 正常结束，只有 rank0 通过独立的**非DDP模型和无分布式采样器**重载 best、全量验证回测和 test 推理；这些阶段不再调用已销毁的进程组。
- CPU/Gloo合成测试只用于验证DDP封装下的梯度平均数学、验证分片和错误路径；它不证明PPU/NCCL可用、BF16算子通过、512目标batch显存足够或T4延迟已通过。

## 11. 实现前预检与验收

### 11.1 小规模预检

先完成不训练、无网络下载的语法/导入与合成单元测试，再做已授权的真实模型轻量预检。完整训练前的预检应覆盖：

1. 解析已冻结配置；检查 timm 名称、版本、预训练缓存/来源和 num_features。真实权重未准备时报告“待授权下载/准备”，不能把 pretrained=False 的合成测试记成预训练载入通过。
2. 在目标512输入上做有限前向/反向，输出和 labels 同为 `[B,1]`，pos_weight 实际等于5，logits、loss、梯度均有限；至少一个正常更新确实改变参数。
3. 保存并通过新协议加载器重载 best.pt/last.pt，在同一确定性输入和 FP32 eval 条件下比较 logit/probability；不把尚未训练的 smoke 权重放入正式 run。
4. 取一个有正负两类的有限 valid 子集验证指标实现；样本数和选择方式写明，子集 AUROC 只证明接口工作，不作为模型效果或选模依据。
5. 无标签 Dataset/DataLoader 和 submission 转换用**人工合成的8条测试记录/图像**验证：包括打乱输入顺序、重复/缺失 ID、无效概率和0.5边界。不读取真实官方 test 图像、不做官方 test 概率预测，不生成任何可误认为正式提交的5855行文件。
6. 训练保存控制逻辑用三轮合成分数验证最优轮、最后轮、min_delta、并列值和失败分支；无需真实跑三个 epoch。

默认真实图像消费预算不超过**64个样本实例（各 rank 与重复前向/反向合计）**、最多1次目标优化器更新、最多1次 FP32 梯度参考；预先分配训练/验证数量并登记，预算不足则停止，不增加重试。验证子集应包含两类；合成数据测试不冒充真实模型预检。若目标微批次/累积组本身超过预算，可先用小配置验证控制逻辑，但必须标注 `target_batch_memory=not_verified`；扩大真实预检预算或目标batch测试需另行授权。

采用混合精度时，同一初始模型/BN状态和缓存增强张量先做一次 FP32 梯度参考，再做一次目标精度更新；参考不执行 optimizer.step，样本仍计入预算。记录数值差异和显存；设备/精度不支持、FP16溢出或超差时停止，不自动重试。若因预算无法完成参考，标记混合精度未验收，不能宣称精度对照通过。

预检失败不启动完整训练。低成本预检通过不等于512目标batch、长程DDP稳定性或T4限时已经验证；这些结论各自注明。正式启动前仍需确认目标资源配置可以运行，不能用一轮完整数据训练发现普通集成错误。

### 11.2 必须新增的回归测试

至少增加以下测试：

- 新协议不会读取 B0 内部划分清单；
- 新协议使用官方 train 和 official valid 的完整清单；
- 缺少源训练参数、FOV参数、timm依赖或预训练权重时明确失败，不静默回退；
- tf_efficientnetv2_s.in21k_ft_in1k 权重加载后分类头输出一个 logit；
- pos_weight 运行时严格为 5.0；
- 三轮模拟训练后 checkpoints/ 只有 best.pt 和 last.pt；
- best.pt 保存的 epoch 是验证 AUROC 最优 epoch；
- last.pt 保存的 epoch 是最后完成 epoch；
- 首轮有限AUROC必存best；小于min_delta的真实提升仍更新best；完全并列保留较早轮；
- 预检不会更新正式模型/BN或消耗正式目录，且不会预测官方test；
- 新格式权重可 weights_only=True、strict=True 载入，旧format 1/2分支含义不变；
- 不生成 epoch_XXXX.pt 或 best_inference.pt；
- best checkpoint 推理输出与 test CSV 的 ID 集合和顺序一致；
- submission 概率有限且位于 [0, 1]；
- submission无索引、UTF-8、ID与概率按键绑定，读取后Pred_Label与0.5规则一致；
- 训练失败不触发test，推理/格式检查失败不写总completed；
- DDP全量指标不是rank均值，验证分片不补齐，输出只有rank0写入；
- B0 既有测试仍然通过，且 B0 输出目录不发生变化。

## 12. 分阶段完成判定

### 12.1 代码交接实现完成

这一层不要求真实跑完完整训练。须满足：

- 新协议和新配置可独立加载；
- B0 原实验文件和输出未被覆盖；
- 模型、固定pos_weight和512预处理接入，合成测试覆盖上述接口/文件/状态规则；
- best/last覆盖保存及新格式回载、无标签推理、提交排序转换在有限测试中通过；
- 新增完整run已编排训练→best回载→valid核对→test推理→submission，但明确标注尚未实际执行的阶段；
- 用户源快照已提供且已知参数已映射；未核实的源引擎细节/真实权重/目标设备须准确列为待完成，不能标记正式实验可启动。

### 12.2 获得授权后的真实实验完成

只有实际运行后才能判断：

- 预训练权重真实载入且来源、版本和预处理已记录；官方train/valid/test来源和样本数已核对；
- 正式训练/验证完成，每轮实际使用pw5；最终只保留best.pt/last.pt两个正式训练权重；
- best.pt可严格重载，全量valid回测与best轮一致；5855张test无标签推理完整；
- 生成的submission.csv通过字段、概率、整数标签、ID集合/绑定/顺序和编码检查；
- run_manifest四阶段均实际完成，记录best_epoch、最后轮、耗时、资源与部署待验事项；
- 不把本地valid指标解释为已知官方test真值成绩，不自动消耗线上提交次数。

### 12.3 模型部署约束单独验收

复赛部署只需要 best.pt 及其推理代码/配置，不将 last.pt、预训练缓存、优化器或训练日志一并当作必要部署模型。对实际提交的全部推理模型文件统计压缩前字节数，同时报告 MB/MiB；不据参数数目直接保证≤500MB，不为了导出再留下第三个best_inference.pt。

T4 16GB、batch_size=1 的100ms是独立验收项：计时从单图读取/解码前开始，包含RGB转换、FOV/letterbox/归一化、传输、模型前向、sigmoid和预测结果可供CPU使用；CUDA计时前后同步，不含首次模型加载/编译。报告预热、样本数、计时边界及均值/P95/最大值；官方尚未明确最终采用哪个统计量，不能仅凭均值宣布官方验收通过。

没有真实T4时写 `t4_latency_status=not_measured`，允许本地训练和CSV生成流程独立完成，但不能宣称部署合规；PPU或其他GPU的耗时不能替代T4证据。推理FP16/BF16、TTA或新增加速策略都不包含在本版既定FP32提交路径中，后续需要另行验证输出与延迟。

本版文档只定义实现边界和验收要求，不启动训练、不下载权重、不修改已有 B0 实验产物。

## 13. 可转交代码实践 Thread 的实施边界

以下是建议交接语，只有用户在目标 Thread 实际发送或明确确认后才构成该阶段授权；读取本文本身不授权代码修改、预检或训练。

> 请实现 timm_official_trainvalid_v1，保持旧B0实现、清单和输出不变；复用B0已验证四卡训练框架（native DDP/NCCL兼容栈、no_sync梯度累积、successful-update调度和spawn worker），但使用独立timm模型、配置、官方train/valid编排、pw5损失、best/last权重格式与提交工作流。不要依赖或声称复现gloo_staged。B0配方中被明确选用的优化器、调度、裁剪和早停参数写成本实验冻结配置。执行相关回归和离线合成测试，不下载权重、不启动真实图像预检、正式训练、官方test预测或平台上传；真实预检和训练以后续授权为准。报告代码实现、配置状态、目标PPU真实预检、正式训练及T4验收边界。

