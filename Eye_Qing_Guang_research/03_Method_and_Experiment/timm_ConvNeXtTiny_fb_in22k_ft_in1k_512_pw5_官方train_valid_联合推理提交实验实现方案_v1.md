# timm ConvNeXt-Tiny 两阶段预训练与联合推理提交实验实现方案 v1

> 编写日期：2026-10-23。本文件是新增实验的实现规格和代码交接文档，不表示代码、权重或目标设备已经验收。本轮只写入并复核文档，不下载权重、不启动真实预检、训练、test 推理或官网提交。

## 1. 实验目的与边界

本实验以当前 timm EfficientNet-V2-S 方案为参照，建立独立的 ConvNeXt-Tiny 实验。不得覆盖 B0、torchvision V2-S 或 timm V2-S 的配置、清单、代码入口和输出目录。

建议实验名：

    timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official_trainvalid_s42_v1

第一轮回答：在官方 train/valid 划分、512 输入、pos_weight=5、统一训练预算和同一提交流程下，ConvNeXt-Tiny 是否产生更好的验证结果和线上候选结果。此比较同时改变骨干、预训练来源、特征维度和输入归一化，不能把差异全部归因于架构。

本轮不加入 ConvNeXt V2、Focal Loss、重采样、分层学习率、layer-wise LR decay、EMA、Mixup、CutMix、TTA、ROI 分支或年龄融合。模型专属调参另立实验。

## 2. 与当前 timm EfficientNet-V2-S 的逐项决策

| 项目 | 当前 timm V2-S | 本实验 | 决策 |
|---|---|---|---|
| 模型 | tf_efficientnetv2_s.in21k_ft_in1k | convnext_tiny.fb_in22k_ft_in1k | 必须替换 |
| 预训练 | ImageNet-21K 后 ImageNet-1K | ImageNet-22K 后 ImageNet-1K | 必须替换并记录 |
| 编码器特征 | 1280 | 768 | 必须替换 |
| 分类输出 | 单个 logit | 单个 logit | 保持 |
| 输入尺寸 | 512 | 512 | 保持 |
| 图像流程 | FOV、BILINEAR、letterbox、增强 | 完全相同 | 保持 |
| 数据 | 官方 train 全量训练，valid 全量验证，test 仅最终推理 | 完全相同 | 保持 |
| 损失 | BCEWithLogitsLoss，pos_weight=5 | 完全相同 | 保持 |
| 优化器及训练预算 | AdamW、3e-4、WD=0.01、100轮、warmup 5轮 | 完全相同 | 保持 |
| 分布式批量 | 4卡，每卡8，累积4，有效batch 128 | 目标保持 | 资源不足时先登记新配置 |
| 精度与通信 | BF16训练、FP32评估、原生NCCL兼容DDP | 完全相同 | 保持 |
| dropout/drop-path | 0/0 | 显式设为0/0 | 保持 |
| 输入归一化 | 当前 V2-S 配方 | ConvNeXt 权重对应的 mean/std | 必须调整 |
| 归一化层 | BatchNorm | 原生 LayerNorm | 必须按模型调整 |
| 选模与输出 | valid AUROC、best/last、训练后提交 | 完全相同 | 保持 |

第一轮不因模型变化自动调整学习率、weight decay、增强或分辨率。若训练不稳定，另建 ConvNeXt 专属调参实验。

## 3. 模型和权重

### 3.1 固定模型身份

- 模型库：timm。
- 完整名称：convnext_tiny.fb_in22k_ft_in1k。
- 架构：ConvNeXt V1 Tiny，不是 Small、Tiny-HNF 或 ConvNeXt V2 Tiny。
- stage 深度：[3,3,9,3]；通道：[96,192,384,768]。
- 特征维度：768；新分类头：Linear(768,1)；全模型微调。
- dropout=0；drop_path_rate=0。
- 预训练：ImageNet-22K 预训练后 ImageNet-1K 微调。

不能替换为 fb_in1k、in12k_ft_in1k、未做1K微调的权重或 torchvision 权重。

### 3.2 官方元数据核对结果

已读取 timm 官方 config.json：

| 字段 | 值 |
|---|---|
| input_size | 3 × 224 × 224 |
| test_input_size | 3 × 288 × 288 |
| fixed_input_size | false |
| interpolation/crop_mode | bicubic/center |
| crop_pct/test_crop_pct | 0.875/1.0 |
| mean | [0.485,0.456,0.406] |
| std | [0.229,0.224,0.225] |
| num_features | 768 |

224是默认入口，288是推荐测试入口。加载权重不会自动修改输入尺寸。本实验显式使用512以保持眼底任务管线一致；512不代表一定优于288，分辨率消融另做。

来源：[ConvNeXt-Tiny 权重 config.json](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k/blob/main/config.json) 和 [timm ConvNeXt 实现](https://github.com/huggingface/pytorch-image-models/blob/v1.0.22/timm/models/convnext.py)。本轮只核对元数据，不下载模型文件。

### 3.3 构造和前向约束

模型构造的核心语义为：

    encoder = timm.create_model(
        "convnext_tiny.fb_in22k_ft_in1k",
        pretrained=initialize_pretrained,
        num_classes=0,
        global_pool="avg",
        drop_rate=0.0,
        drop_path_rate=0.0,
    )
    assert encoder.num_features == 768
    model = ImageClassifier(encoder, 768, dropout=0.0)

正式代码必须通过已准备的本地权重或已有缓存加载，禁止因 pretrained=True 隐式联网。num_classes=0只移除原1000类输出，必须保留完整编码器前向中的池化和分类头前归一化；不能绕过预训练头归一化后直接手工平均空间特征。

ImageClassifier 的 forward(image, age=None) 忽略 age，输出形状必须为 [B,1]，模型内不加 sigmoid。加载 best/last 时用 pretrained=False 构建相同结构，再 strict=True 载入检查点，不能重新载入ImageNet权重覆盖微调结果。

### 3.4 LayerNorm 处理

ConvNeXt-Tiny使用 LayerNorm/LayerNorm2d、GELU、LayerScale 等结构，没有 EfficientNet 的 BatchNorm running statistics。保留这些原生层，不转换 SyncBN，不添加本地BN逻辑。配置中的 batchnorm 设为 not_applicable，模型元数据记录 normalization=layernorm。

DDP仍需同步参数和梯度。通用评估上下文若遍历 named_buffers，可以允许空集合正常执行，但不能记录“BN已校正”。

## 4. 数据和预处理

官方协议保持不变：

- train 51,396张，全部训练；
- valid 5,849张，每轮验证和选模；
- test 5,855张，只在训练成功且 best 回载后无标签推理；
- 不从官方 train 再划分；
- 不读取test标签，不让test参与训练、早停、调参、阈值或增强设计。

先核对官方 CSV 和 manifest 一对一对应、ID唯一、三划分无交集、标签合法、文件存在、顺序一致。坏图、缺图和数量不符都报错，不静默跳过或补齐。

沿用现有 FOVLetterbox：RGB → FOV裁剪 → 训练增强 → 等比缩放和黑色居中letterbox到512 → ToTensor → ConvNeXt mean/std归一化。

FOV参数保持 preview_size=512、threshold=8、min_line_fraction=0.01、margin=0.02、min_area=0.5、min_coverage=0.995。训练增强保持水平翻转p=0.5、旋转[-10°,10°]、亮度[0.9,1.1]、对比度[0.9,1.1]；验证/test关闭随机增强。FOV预览、旋转、缩放继续使用BILINEAR，不改为权重仓库的bicubic中心裁剪。

输入 mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]。输入像素归一化与模型内部LayerNorm是两个不同环节。512和FOV/letterbox偏离原始ImageNet评测配方，运行清单必须注明。

## 5. 训练和分布式配置

固定值：

- seed=42；
- AdamW，所有可训练参数统一参数组；
- learning_rate=0.0003，weight_decay=0.01，betas=(0.9,0.999)，eps=1e-8；
- gradient clipping=5.0；
- epochs=100，warmup_epochs=5，warmup起点1e-4，cosine最低学习率1e-6；
- 使用既有 successful-update warmup/cosine 调度；
- BF16 autocast训练，不启用GradScaler；BCE在FP32计算；验证/test为FP32；
- 四rank，每卡micro_batch=8、累积4，目标有效batch=128；
- 每进程 workers=4、cpus_per_rank=6；
- early stopping：min_epochs=30、min_delta=0.0001、patience=15；
- 自然类别比例，禁止重采样，drop_last=false。

每rank每轮消费 51,396/4=12,849张；每rank有1607个微批次，累积组数402；最后一组每rank17张，全局68张。尾组按真实样本数归一化，不当作128张，不丢弃。无早停完整100轮最多40,200次成功更新，实际以日志为准。

若512目标微批量或FP32评估批量无法容纳，先报告并登记资源版配置；不自动改batch、累积、activation checkpointing、channels-last、compile或算子。

## 6. 损失、指标、选模和早停

损失固定 BCEWithLogitsLoss，pos_weight=tensor([5.0])，reduction=mean。logits和labels均为 [B,1]，labels为float32的0/1。5只放大正类损失项，不自动重算，不叠加class_weight、Focal或重采样。

每轮完整valid记录 validation loss、连续概率AUROC、Average Precision、梯形PR-AUC、Brier、Accuracy、Balanced Accuracy、F1、Precision、Sensitivity、Specificity、TP/TN/FP/FN、固定0.5硬标签指标以及 Recall@Specificity≥90% 和 ≥95%。连续概率AUROC用于选模；阈值0.5只用于固定工作点和提交标签。

valid预测先按官方ID收集并核对唯一性，再由rank0计算全量指标，不能平均各batch或各rank AUROC。AP使用average_precision_score；梯形PR-AUC使用precision_recall_curve后由auc(recall,precision)计算。高特异度召回在实际唯一阈值上整组处理同分样本，不插值，不把探索阈值用于提交。

best与早停状态分开：best_raw初始负无穷，第一轮有限AUROC必保存，之后严格提升即保存，完全相等保留较早轮；early stopping 的 best_significant 只有超过0.0001才清零stale，epoch达到30且stale达到15时停止，最多100轮。

NaN、缺失预测、重复ID或缺失ID属于验证失败，不把分数置0继续。官网评分字段的最终解释仍以提交规则和答疑材料为准，不在本实验伪造官网总分。

## 7. 代码边界和适配项

现有项目已有 timm V2-S 的模型、配置、检查点、训练和管线文件，可复用数据、指标、DDP、原子写入和提交校验的底层能力。建议新增：

    src/eye_qing_guang/models/timm_convnext.py
    src/eye_qing_guang/contracts/convnext_config.py
    scripts/run/run_timm_convnext_experiment.py
    configs/timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official.yaml

不能只复制V2-S YAML，因为当前代码有 V2-S 专属硬编码：

- timm_config.py 的 PROTOCOL、MODEL_NAME、mean/std和native_local身份检查；
- timm_efficientnet.py 的权重 repo、pretrained metadata和num_features=1280；
- timm_checkpoints.py 的模型名、维度和V2-S工厂调用；
- timm_official.py 的配置加载、模型构造和回载路径；
- evaluation_state 对缓冲区同步的假设。

新增协议名为 timm_convnext_official_trainvalid_v1。可以共用 format_version=3，但检查点必须同时验证协议、模型名、num_features=768、timm版本和配置身份。旧B0和V2-S分支的含义、配置、输出和已有测试保持不变。错权重、错版本、错特征维度或错归一化配置要直接失败。

## 8. 检查点、自动推理和提交

checkpoints最终仅保留 best.pt 和 last.pt。best在valid AUROC严格改善时原子替换；last在一轮训练和完整valid成功后原子替换。禁止 epoch_XXXX.pt 和 best_inference.pt。只保存解包后的state_dict，去除DDP module.前缀，Tensor转CPU，元数据使用普通安全类型。

format_version=3载荷至少包含：

    checkpoint_kind=inference_weight
    epoch
    config
    age_stats=null
    pos_weight=5.0
    model_state
    metrics
    timm_version
    model_name=convnext_tiny.fb_in22k_ft_in1k
    num_features=768
    model_metadata

本版不保存 optimizer、scheduler、scaler或RNG，不支持resume。加载使用weights_only=True、CPU map_location、strict=True；模型名、版本、配置、特征维度必须匹配。

运行顺序：

    配置和环境校验
    -> 官方清单校验
    -> 本地预训练权重核对
    -> 有限预检
    -> 重置随机状态并重建正式模型
    -> 训练和逐轮valid
    -> best回载及全量valid一致性核对
    -> best推理官方test
    -> 生成submission.csv
    -> 写最终run_manifest

建议正式目录：

    experiments/timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official_trainvalid_s42_v1/
      config_snapshot.yaml
      run_manifest.json
      checkpoints/best.pt
      checkpoints/last.pt
      reports/history.csv
      reports/metrics.json
      reports/valid_best_metrics.json
      predictions/best/valid_predictions.csv
      predictions/best/test_predictions.csv
      submission/submission.csv

submission字段固定 Image_name,Probability,Pred_Label，5855行；概率有限且在[0,1]，Pred_Label由原始概率按p>=0.5得到，整数0/1；UTF-8、无索引、ID不重复，按官方test顺序输出。生成文件不等于上传，不自动消耗官网次数。

## 9. 目标配置字段

新配置需由ConvNeXt专用校验器读取，不能直接交给现有V2-S校验器：

    experiment:
      name: timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official_trainvalid_s42_v1
      protocol: timm_convnext_official_trainvalid_v1
      seed: 42
      output_root: experiments
    recipe:
      implementation: b0_native_ddp_v1
      implementation_path: src/eye_qing_guang/utils/distributed.py
      parameter_notes: controlled backbone comparison; no gloo_staged
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
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
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
      name: convnext_tiny.fb_in22k_ft_in1k
      pretrained: true
      timm_version: 1.0.22
      pretrained_file: null
      pretrained_revision: main
      dropout: 0.0
      drop_path_rate: 0.0
      batchnorm: not_applicable
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

注意：第9节的 preprocessing 和 augmentation 已展开为具体字典，正式 YAML必须保持这些字段，不能让代码读取不存在的默认值。raw_root、manifest_dir和权重文件/revision需在实际服务器上核实。现有 timm V2-S 协议名应保持不变；ConvNeXt正式协议必须使用 timm_convnext_official_trainvalid_v1。

## 10. 预检、验收和交接边界

授权实现后先做语法、导入、配置分派和合成检查，至少覆盖模型输出形状、LayerNorm路径、mean/std、pw5、尾累积组、best/last覆盖、strict回载、submission ID和0.5边界，以及旧B0/V2-S回归。

目标设备预检默认最多64个真实样本实例、最多一次优化器更新和一次不step的FP32梯度参考，独立写入 experiments/_preflight/<experiment>_pf_v1。64个样本不覆盖完整有效batch或每卡eval batch，通过后仍把目标batch显存记为 not_verified。检查失败不启动正式训练，不自动降精度或重试。权重加载、PPU BF16算子、四卡DDP和目标输入显存需分别记录。

后续授权后的命令接口：

    torchrun --standalone --nproc_per_node=4 scripts/run/run_timm_convnext_experiment.py --project-root /实际代码项目绝对路径 --config configs/timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official.yaml --preflight-only

    torchrun --standalone --nproc_per_node=4 scripts/run/run_timm_convnext_experiment.py --project-root /实际代码项目绝对路径 --config configs/timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official.yaml

T4 batch=1 100ms和模型压缩前大小≤500MB是独立验收项，必须实际测量；没有实测就记录not_measured。参数量不能替代文件大小统计。

## 11. 结果比较方法

先核对两实验的配置快照、清单版本、实际权重来源、停止轮次、物理/有效batch和输入流程。按相同官方valid的逐图ID比较AUROC、AP、固定0.5工作点、高特异度召回、Brier、耗时、峰值显存、参数量和实际文件大小。

AUROC是本版主选模指标；其他指标用于解释工作点和校准差异。单个seed的小幅差异只记为观察结果，不宣称统计显著。test预测只作为候选submission和格式检查依据，不能由无标签分布推断线上成绩。

## 12. 可转交代码实践 Thread 的交接语

> 请按本文新增 ConvNeXt-Tiny 实验：模型为 convnext_tiny.fb_in22k_ft_in1k，使用512输入、官方train/valid、pw5、seed42、四卡BF16与既定AdamW和成功更新调度。复用已有timm/B0底层，但按新协议显式分派模型、配置和检查点身份，保留旧实验行为与输出。ConvNeXt保留LayerNorm及预训练特征头归一化，使用ImageNet mean/std，不能沿用V2-S的1280维、0.5归一化或native-local BN身份。新增独立入口、配置和输出，编排best/last、best回载、全量valid核对、test推理及submission生成。先做实现和离线合成回归；不下载权重、不跑真实预检、正式训练、test预测或官网上传，后续分别授权。

## 13. 文档复核结果

已复核：实验名与模型名一致；ConvNeXt特征维度为768；224默认入口与288测试入口未混淆；512输入已明确为任务覆盖；mean/std与权重匹配；LayerNorm与BatchNorm已区分；官方清单数量、pw5、seed、训练预算、DDP路线、尾组计算、best/last规则、提交字段和路径一致；当前代码中的V2-S硬编码已列为必须适配项。

文档复核只证明规格自洽，不证明代码已有ConvNeXt分派、权重已缓存、512目标batch可运行或T4部署达标。
