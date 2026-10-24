# 青光眼二分类实验工程

本项目使用 PyTorch 完成眼底图像二分类，支持图像基线和可选的年龄特征融合。工程参考 `skin_optics` 的组织方式：配置描述实验、脚本解析命令、Python 包实现可复用功能，每次实验独立保存结果。

已接通数据审计、样本清单生成、训练、检查点评估、无标签测试集预测、提交文件转换和可视化。项目搭建只使用合成样本验证流程，尚未运行正式比赛数据训练，未产生模型性能结论。

新增的 **B0-dev ResNet50-512** 使用独立内部开发协议，服务器步骤见本文末尾“B0-dev 服务器运行”。下方原始 ResNet18 快速流程继续保留，两者的数据划分和优化预算不同。

四卡扩展使用独立 `b0_dev_ddp_v1` 协议，见文末“PPU 后四卡 DDP 运行”。单卡配置继续保留，四卡配置的 batch/worker 均按每个进程解释；PPU 上的通信、目标模型 AMP 和实际加速仍须在服务器验证。

## 目录

```text
Eye_Qing_Guang_code/
├── configs/
│   ├── resnet18.yaml                 # 初始基线，兼容旧实验
│   ├── b0_resnet50_512_dev_v1.yaml    # B0 单卡内部开发实验
│   └── b0_resnet50_512_dev_ddp4_ppu_v1.yaml # PPU 后四卡 DDP
├── data/
│   ├── raw/glaucoma_first/           # 本地原始布局；PPU服务器数据在 /data/workspace/KanglongTan/data/glaucoma_first
│   ├── manifests/                    # 版本化的标准样本清单
│   ├── processed/                    # 后续预处理生成的数据
│   ├── contracts/dataset.yaml        # 数据字段、标签值及处理约定
│   └── README.md
├── experiments/
│   └── <experiment_name>/            # 运行时创建，同名目录拒绝覆盖
│       ├── checkpoints/
│       ├── reports/
│       ├── predictions/
│       ├── figures/
│       ├── config_snapshot.yaml
│       ├── run_started.json
│       └── run_manifest.json
├── scripts/
│   ├── preprocess/                   # audit_dataset.py / build_manifest.py
│   ├── train/                        # run_binary_classification.py
│   ├── evaluate/                     # evaluate_checkpoint.py / summarize_experiment.py
│   ├── infer/                        # predict_test.py
│   ├── submit/                       # build_submission.py
│   └── visualize/                    # 曲线、混淆矩阵、Grad-CAM
├── src/eye_qing_guang/
│   ├── contracts/                    # 配置、样本清单和检查点校验
│   ├── datasets/                     # 图像、数据增强、加载器、年龄处理
│   ├── models/                       # 图像分类、年龄融合及模型工厂
│   ├── losses/                       # BCE / Focal BCE
│   ├── metrics/                      # 二分类指标和实验汇总
│   ├── trainers/                     # 单轮训练
│   ├── evaluators/                   # 批量预测和验证
│   ├── pipelines/                    # 训练、评估、推理完整流程
│   ├── visualization/                # 图表与解释性可视化
│   ├── utils/                        # 路径、随机种子和实验文件读写
│   └── cli.py                        # 所有入口共用的参数解析
├── tests/                            # 数据边界和合成数据端到端验证
├── .gitignore
├── pyproject.toml
└── README.md
```

`configs/` 不预设阶段子目录。新增实验时复制并修改一份 YAML，待实验数量增长后再按实际需求分组。`data/processed/` 当前只预留目录；基线在加载时缩放和归一化，不提前复制全量图像。

## 环境安装

要求 Python 3.10+。当前服务器使用已配置的 Podman 个人容器 `qinghao_ppu_train`，项目位于 `/home/KanglongTan/QinghaoLi/Eye_Qing_Guang`。环境以 [PPU服务器与个人训练环境配置说明](PPU服务器与个人训练环境配置说明.md) 为依据：保留镜像内的 Python3.12、PPU适配版PyTorch2.11.0及SDK，不迁入旧3090的Conda环境，也不通过普通PyPI/NVIDIA wheel替换torch、torchvision或CUDA运行库。如需登记或重新安装源码，在已有依赖环境中执行：

```bash
# 在宿主机进入已有个人容器；若已在容器内，跳过这一行。
podman exec -it qinghao_ppu_train bash
cd /home/KanglongTan/QinghaoLi/Eye_Qing_Guang
python -m pip install --no-index --no-deps --no-build-isolation -e .
```

这里的 `--no-deps --no-build-isolation` 用于保留现有厂商框架；若有缺少的包，先确认具体缺项及兼容版本再补齐，不重新解析并覆盖整套环境。容器已有HTTP/HTTPS代理，本次路径配置不修改代理、镜像、设备映射或现有DeepSeek服务。

开发时也可以将 `src` 加入 `PYTHONPATH`。Windows本机默认Anaconda导入PyTorch曾存在OpenMP冲突；不在项目代码中设置绕过冲突的环境变量。服务器PPU环境应以容器内实际兼容性检查为准。

安装后，脚本与统一命令等价：

```powershell
python scripts/train/run_binary_classification.py --help
eye-qing-guang train --help
python -m eye_qing_guang.cli train --help
```

## 数据与评估约定

两份默认配置的 `data.raw_root` 均为 `/data/workspace/KanglongTan/data/glaucoma_first`，其下直接包含官方 `train.csv`、`valid.csv`、`test.csv` 和 `train/`、`valid/`、`test/` 图像目录，对应 51,396、5,849、5,855 张图像。环境说明给出的 `/data/workspace/KanglongTan/data` 是上级挂载根，本次按用户明确指定的 `glaucoma_first` 子目录读取数据。训练和验证表包含 `Image_name, Age, True_Label`，测试表只有 `Image_name, Age`。Windows本地若要读取比赛数据，需另用配置指定本地 `raw_root`；合成测试会显式改用临时目录，不访问服务器原始数据。

- 清单保留官方 `Image_name` 作为 `image_id`；实际文件路径写入 `image_path`，解决 CSV 的 `.jpg` 和磁盘 `.JPG` 大小写差异。
- 标签只接受 `0/1`。模型输出一个 logit，`sigmoid(logit)` 表示 `True_Label=1` 的概率，临床语义以比赛说明为准。
- 基线仅使用图像。设置 `model.use_age: true` 可启用年龄融合；缺失年龄以训练集年龄中位数填充，再用训练集均值、标准差归一化，统计量随检查点保存。
- 初始 ResNet18 流程用官方 `valid` 早停；B0-dev 只在官方 train 内划分和选模，官方 valid/test 不进入开发训练。两套流程不自动互换。
- 当前模型选择指标是 ROC AUC，并报告 AP、Accuracy、Balanced Accuracy、F1、Precision、Sensitivity、Specificity 和混淆计数。这是可修改的基线设置，尚不代表已核实的比赛排名规则。
- 分类阈值来自配置，默认 `0.5`。只含单一标签的评估集，其 ROC AUC/AP 返回 JSON `null`；使用 ROC AUC 选模时要求验证集包含两类。

## 使用流程

项目根目录由安装位置自动识别；服务器宿主机和容器中均为 `/home/KanglongTan/QinghaoLi/Eye_Qing_Guang`，无需将Linux路径写死到Python源码。该目录应直接包含 `configs/`、`scripts/`、`src/` 和 `pyproject.toml`。配置和命令中的相对路径均相对于项目根目录解析，可用 `--project-root` 显式指定；绝对路径按原值使用，`--project-root` 不会重定向绝对的 `data.raw_root`。

原始数据保留在共享的数据挂载目录。生成的清单仍放在项目内的 `data/manifests/`：B0使用 `/home/KanglongTan/QinghaoLi/Eye_Qing_Guang/data/manifests/glaucoma_b0_dev_ppu_v1`，ResNet18使用同级的 `glaucoma_first_ppu_v1`，独立测试清单示例为 `glaucoma_official_test_ppu_v1`。实验输出根为 `/home/KanglongTan/QinghaoLi/Eye_Qing_Guang/experiments`，无需额外创建统一outputs目录。外部原始图像写入清单时使用实际绝对路径；已有3090清单及检查点不自动改写，后续获准生成时使用新的PPU清单版本，核对来源和相同的内部样本划分。每条命令均支持 `--help`；以下命令供相应实验阶段使用，不会自动执行。

### 1. 审计数据并生成清单

```powershell
python scripts/preprocess/audit_dataset.py --output data/manifests/audit_v1.json
python scripts/preprocess/build_manifest.py
```

审计检查必要列、标签、年龄、标识唯一性和图像路径；增加 `--decode-images` 可逐张验证图像解码。发现错误时立即停止并报告首个错误。清单生成后，`data/manifests/glaucoma_first_ppu_v1/` 包含 `train.csv`、`valid.csv`、`test.csv` 和 `summary.json`。已有清单版本拒绝覆盖；改版需同时修改 `data.manifest_dir`，或通过 `--output-dir` 指定新目录并同步配置。

### 2. 训练基线

```powershell
python scripts/train/run_binary_classification.py --config configs/resnet18.yaml --name resnet18_image_ppu_v1
```

支持 `resnet18`、`resnet50`、`efficientnet_b0`。`tiny_cnn` 供离线流程验证，必须设置 `pretrained: false`。默认 ResNet18 使用 ImageNet 预训练，首次正式训练可能下载权重；离线运行可在配置中关闭。当前未做模型优劣比较。

初始 ResNet18 流程使用 AdamW、BCE 或 Focal BCE，可选按训练集比例计算 `pos_weight`，采用单设备 FP32 和普通 patience。B0 使用后文的专用 AMP/累积/调度/早停逻辑。当前均不支持断点续训、DDP、自动调参或集成。

每次运行在启动前确定唯一实验目录，同名运行拒绝覆盖。改进轮保存 `epoch_XXXX.pt`，完成后额外保存 `best.pt` 和 `last.pt`。检查点包含模型/优化器状态、配置、年龄统计量、损失权重和轮次。逐轮进度写入 `reports/epoch_XXXX.json`，完成后生成 `history.csv` 和最佳指标 `metrics.json`。失败时保存 `run_manifest.json`，记录已完成的验证轮数和错误；进程被强制结束时可参考 `run_started.json` 和已有逐轮文件。

### 3. 评估与汇总

```powershell
python scripts/evaluate/evaluate_checkpoint.py --checkpoint experiments/resnet18_image_ppu_v1/checkpoints/best.pt --output-dir experiments/resnet18_image_ppu_v1/reports/valid_best
python scripts/evaluate/summarize_experiment.py --experiments experiments/resnet18_image_ppu_v1 --output experiments/summary_v1.csv
```

评估读取检查点内的模型和预处理配置，避免外部 YAML 改动导致不一致；支持 `--split train` 或默认 `valid`，用 `--manifest` 指定另一份同格式清单。输出指标、逐样本预测及运行记录。`best.pt` 的官方验证集成绩是选模成绩，不应解释为独立测试性能。

### 4. 测试集预测与提交转换

```powershell
python scripts/infer/predict_test.py --checkpoint experiments/resnet18_image_ppu_v1/checkpoints/best.pt --output-dir experiments/resnet18_image_ppu_v1/predictions/test_best
python scripts/submit/build_submission.py --predictions experiments/resnet18_image_ppu_v1/predictions/test_best/test_predictions.csv --output experiments/resnet18_image_ppu_v1/predictions/submission.csv
```

内部无标签预测字段为 `image_id,logit,probability`。当前配置采用 `submission.mode: competition`，按本地《大数据挑战赛 - 提交作品》摘录输出 UTF-8 CSV，固定表头为 `Image_name,Probability,Pred_Label`，文件名必须为 `submission.csv`。`Pred_Label` 由未取整的概率按配置阈值计算，默认 `Probability >= 0.5` 为1；保留概率数值，不用类别标签替代排序分数。转换严格匹配官方测试 CSV 的标识和行序，拒绝缺失、多余、重复、空标识以及非有限或越界概率，拒绝覆盖已有文件。样本数以官方测试 CSV 为准，本次数据应为5855行（不含表头）。

旧二列导出仅在显式设置 `--mode probability|label` 或加载旧版二列 `submission` 配置时保留，不作为当前比赛格式。`--id-column`、`--value-column`、`--label-column` 可覆盖配置，但比赛模式拒绝不符合上述固定表头的列名。旧B0配置/检查点没有 `submission` 节时，默认使用比赛三列模式；新增节只影响提交格式，不改变训练协议或预处理。

初始ResNet18配置启用ImageNet预训练；本地提交页未明确授权外部预训练，不能据此认定其合规。比赛默认研究路径使用随机初始化的B0。初赛只提交结果CSV，Conda环境压缩包、内部开发预测和整个实验目录均不属于初赛提交内容。

### 4.1 B0 的独立测试清单与提交路径

B0开发清单仍只包含官方train划出的内部拟合/开发部分，不自动读取官方test。后续明确进入测试预测阶段时，在工程根目录使用独立命令生成新版本测试清单；它只读取官方test表和图像路径，不读取train/valid，不修改内部划分，不运行模型。可选 `--decode-images` 完整解码测试图片；默认只检查元数据和路径。

```bash
python scripts/preprocess/build_test_manifest.py --config configs/b0_resnet50_512_dev_v1.yaml --output-dir data/manifests/glaucoma_official_test_ppu_v1
python scripts/infer/predict_test.py --checkpoint experiments/b0_resnet50_512_dev_ppu_s42_v1/checkpoints/best_inference.pt --manifest data/manifests/glaucoma_official_test_ppu_v1/test.csv --output-dir experiments/b0_resnet50_512_dev_ppu_s42_v1/predictions/test_best_v1
python scripts/submit/build_submission.py --config configs/b0_resnet50_512_dev_v1.yaml --predictions experiments/b0_resnet50_512_dev_ppu_s42_v1/predictions/test_best_v1/test_predictions.csv --output experiments/b0_resnet50_512_dev_ppu_s42_v1/predictions/test_best_v1/submission.csv
```

以上为未来阶段命令示例，不会随训练自动执行。`test-manifest` 的输出目录必须与开发清单分开，已有目录拒绝覆盖。B0预测必须显式提供 `--manifest`，避免误将内部开发集当作测试集。B0-dev检查点仍是内部选模结果；此入口不替代B0-fit，也不自动训练、提交或将内部成绩标记为官方成绩。

复赛还需实际核对精简模型文件及所有必需模型文件压缩前总大小≤500MB，并在文档指定的T4 16GB、Python3.12、PyTorch2.10.0/torchvision0.25.0、CUDA12.8/cuDNN9环境验证加载、预测一致性及单图端到端延迟≤100ms。3090训练环境与预检中的短程未预热耗时均不能代替该验收，当前未完成这些实测。官方AUPRC的积分定义未在可读文字中说明，工程AP仍按average precision单列，不声称二者完全等同。

按本地提交页末尾细则：每天最多一次成功出分；待评分期间不可继续提交，报错可重试但当天最多两次。截止时间记载为2026年11月6日23:59；实际提交前应复核页面最新要求。本项目命令只生成本地文件，不调用平台上传。

### 5. 可视化

```powershell
python scripts/visualize/plot_training_curves.py --history experiments/resnet18_image_ppu_v1/reports/history.csv --output experiments/resnet18_image_ppu_v1/figures/training_v1.png
python scripts/visualize/plot_confusion_matrix.py --metrics experiments/resnet18_image_ppu_v1/reports/valid_best/metrics.json --output experiments/resnet18_image_ppu_v1/figures/confusion_v1.png
python scripts/visualize/generate_gradcam.py --checkpoint experiments/resnet18_image_ppu_v1/checkpoints/best.pt --image /data/workspace/KanglongTan/data/glaucoma_first/valid/valid_00000.JPG --age 61 --output experiments/resnet18_image_ppu_v1/figures/gradcam_v1.png
```

Grad-CAM 默认解释标签 1；`--class-index 0` 解释标签 0。年龄融合模型的 `--age` 应使用样本实际年龄，省略时按训练集缺失值规则处理。输出为缩放后图像上的 PNG 热图，用于模型检查。

## 验证和版本控制

```powershell
python -m unittest discover -s tests -v
```

测试在临时目录创建合成图像，覆盖文件名大小写、无标签测试数据、年龄统计、指标边界、图像/年龄模型，以及训练→检查点加载→评估→推理→提交→可视化。测试使用 CPU、极小模型和一轮训练，不下载预训练权重。

`.gitignore` 排除原始数据、派生数据、实验结果、权重、环境、缓存和本机凭据；保留源码、配置、测试、说明和 `data/contracts/`。生成的清单默认不提交。已被 Git 跟踪的数据不受新增忽略规则影响。正式实验应使用独立版本目录保存结果。

## B0-dev 服务器运行

实现依据为研究目录中的 `03_Method_and_Experiment/B0_ResNet50_512_基线实验实现方案_v1.md`。PPU配置仍为 seed=42 的单卡B0开发协议。本次路径与配置更新没有生成清单、运行PPU预检、训练或推理；环境说明记录的矩阵计算通过不等于项目的ResNet50、FP16 AMP或吞吐验收通过。

配置 `configs/b0_resnet50_512_dev_v1.yaml` 固定：随机初始化 torchvision ResNet50，全网络训练，2048→1 单头，dropout=0，图像单输入；普通 BCE、pos_weight=1、自然比例随机遍历、不丢尾批。训练 physical batch=16、累积=2，评估 batch=16；AdamW 的峰值 lr=1e-3、weight_decay=1e-4、betas=(0.9,0.999)、eps=1e-8。最多80轮，最短30轮允许早停，patience=15，min_delta=1e-4。配置中的关键字段会被校验，拼错或未实现的 B0 字段会报错。

当前配置适配个人容器 `qinghao_ppu_train`：源码为 `/home/KanglongTan/QinghaoLi/Eye_Qing_Guang`，原始图像与CSV为 `/data/workspace/KanglongTan/data/glaucoma_first`。容器已固定 `CUDA_VISIBLE_DEVICES=4,5,6,7`，内部逻辑 `cuda:0/1/2/3` 分别对应物理PPU4/5/6/7。默认 `training.device: cuda:0` 使用物理PPU4；不要在进入容器后重新覆盖 `CUDA_VISIBLE_DEVICES`，也不要填写 `cuda:4～7` 或用逗号设备列表代替当前单设备参数。本项目未接入DDP，本次不启动多卡，也不操作 `ds41` 服务。

以下仅列出后续相应阶段的命令示例，本次不执行：

```bash
cd /home/KanglongTan/QinghaoLi/Eye_Qing_Guang
python -m pip install --no-index --no-deps --no-build-isolation -e .
python -m unittest discover -s tests -v
python scripts/preprocess/audit_dataset.py --config configs/b0_resnet50_512_dev_v1.yaml --decode-images --output data/manifests/b0_train_audit_ppu_v1.json
python scripts/preprocess/build_development_manifest.py --config configs/b0_resnet50_512_dev_v1.yaml
python scripts/train/preflight_b0.py --config configs/b0_resnet50_512_dev_v1.yaml --name b0_resnet50_512_preflight_ppu_s42_v1
```

以上最后一条是有界预检，**不是正式训练**：仅内部拟合子集，训练与FP32梯度对照合计最多64个样本实例、最多4个累积组；FP32对照即使不执行优化器step也保守计入一个组。默认先用16×2=32个实例完成AMP累积更新，再从相同初始权重和BatchNorm状态对同一组增强后的微批次执行32个实例的FP32样本加权前反向。重复喂入的对照样本也计数。两条路径使用相同物理微批次，不把不同BatchNorm批次的梯度要求为逐位一致。

预检比较未缩放BCE与unscale/裁剪后的梯度：损失允许绝对差 `0.005 + 0.02×|FP32 loss|`，梯度差的L2范数允许 `0.005 + 0.1×FP32梯度L2范数`。这些是启动前公开的数值冒烟容差，不是调参得出的精度保证；实际差值、容差与通过状态写入 `reports/preflight.json`。超差会失败，不自动调整容差、切换精度或改变配方。FP32路径独立按微批样本数/组总样本数加权；对照前后恢复AMP模型状态，不改变其检查点。

正常的GradScaler溢出跳步保留scale退避；预算允许时读取下一组新的拟合样本继续检查，不重喂跳步组、不推进学习率成功更新索引。若预算已用尽而未完成成功更新或FP32对照，状态写为 `inconclusive`，不是数值失败，更不是预检通过；命令以退出码2结束，避免 `preflight && train` 自动开始训练。例如默认配置首组跳步、第二组成功后已消费64个实例，就会明确报告“尚无预算执行FP32对照”。此时应报告现有记录并另定预检安排，不自行换名字追加运行。失败/重复运行的成本均需累计。

预检还检查参数确实更新、保存加载、配置评估batch的确定性FP32推理和BN不更新；保存最多12张拟合部分的原图/FOV对照，以及AMP峰值显存、短程吞吐和一次未充分预热的端到端路径耗时。若选取样本不足以填满评估batch，会记录实际检查数量。它不提供正式时延认证。AMP组的日志位于 `successful_amp_group` 和 `update_attempts`，总消费/组数以 `progress` 为准；FP32参考样本和组数另有分类字段。

审计在 B0 配置下只检查官方 train；`--decode-images` 先验证文件结构，再重新打开、完整加载像素并转换RGB，拒绝能通过结构验证但像素已截断的JPEG，可能较慢。内部划分先按 `image_id.casefold()` 排序，无关联组时按 seed=20261018 进行80/20分层划分。有可信关联时采用所有患者/检查/已确认重复关联的并集连通分量，使用 `StratifiedGroupKFold(5)` 的第0折作为内部开发集；不择优选折，不拆组回退。开发比例在[0.1,0.3]之外或单类时停止，该可用性范围也写入清单说明。清单 `train.csv`/`valid.csv` 额外保存 `source_split=official_train` 与 `group_id`，`summary.json` 保存计数、版本和限制。训练时重新核对来源、关联、版本和实际划分；不读取官方 valid/test。

若有可信患者/检查或确认重复信息，先提供关联 CSV，并在 `development.associations_csv` 中填入相对路径，再生成开发清单。格式示例：

```csv
image_id,association_type,group_id
train_00000.jpg,patient,P001
train_00001.jpg,patient,P001
train_00001.jpg,confirmed_duplicate,D001
train_00002.jpg,confirmed_duplicate,D001
```

支持 `patient`、`examination`、`confirmed_duplicate`。上例三张图会在同一个连通组中；同一图可有多条关联。没有可信患者ID时保留“未能排除同一患者双眼跨划分”的限制，不推断患者身份，也不自动以视觉相似度合并样本。新增关联必须生成新清单版本。

检查 `experiments/b0_resnet50_512_preflight_ppu_s42_v1/figures/` 后，再冻结资源配置。对照图只是候选，需确认覆盖正常曝光、极暗、强反光、宽黑边、偏心和非方形案例；不足时先补充有界人工检查，不据少量例子宣称 FOV 已全面通过。原方案最多24张人工候选。若显存不足，预检失败并保留进度，不切CPU、不自动减batch、不更换损失；在正式开始前显式修订并记录物理batch、累积、eval batch与workers。若FOV规则不可靠，应另立“原图等比填充”变体，不能静默修改 `fov_letterbox_v1`。

预检数值和图像检查通过后，由你在服务器启动正式训练：

```bash
python scripts/train/run_binary_classification.py --config configs/b0_resnet50_512_dev_v1.yaml
```

按上述显式 `--name` 运行时，PPU预检与正式目录分别是 `b0_resnet50_512_preflight_ppu_s42_v1` 和 `b0_resnet50_512_dev_ppu_s42_v1`。已有目录拒绝覆盖，失败运行保留原产物，新版本需要新名称；当前无 `--resume`，不承诺中断后无缝恢复。请保留训练开始时记录的源码版本与环境，特别是 Pillow、torchvision、scikit-learn 版本；不同 scikit-learn 版本导致来源记录变化时不会静默接受旧划分，应保留原环境或另定新清单版本。

### PPU 单卡资源参数调整

YAML已标注资源参数的位置，本次仍保留workers=0、physical batch=16、累积=2、eval batch=16、FP16 AMP开启作为未测速起点。单卡总显存144GiB并非本任务独占；按环境说明，DeepSeek也在所分配的物理PPU4～7上运行，显存余量及计算负载会变化，不能把历史剩余显存当作固定额度。

| 参数 | 后续可比较的设置 | 对实验的影响 |
| --- | --- | --- |
| `data.num_workers` | 先比较0与8，再视CPU/I/O测试16 | 可能改变增强随机序列；不是越多越快 |
| `training.eval_batch_size` | 16→32，有余量时另测更大值 | FP32验证显存需单独确认，不能按AMP训练估算 |
| `training.batch_size` + `accumulation_steps` | 16×2；资源允许时另测32×1 | 名义有效batch相同，但BatchNorm行为不同，需记录物理batch |
| `training.device` | 逻辑cuda:0～3中的一个 | 仅选一张获准的物理PPU，不启用DDP |
| `training.amp` | 保持true | 切换BF16/FP32不是当前B0吞吐参数，应另立变体 |
| `experiment.name` | 每个资源组合使用新名称 | 防止覆盖并保留配置快照，不复用旧3090实验目录 |

已有预检预算是AMP及FP32参考合计64个样本实例，因此一次完整对照要求有效batch≤32，且仍可能因溢出跳步而预算不足。64×1会没有FP32对照预算，64×2超过单组预算；不要通过改名反复预检或静默扩大预算。后续扩大目标batch需要另定验证安排。

速度比较保持图像尺寸、FOV/增强、学习率、scheduler和轮数预算一致，一次只改变一个资源因素；改变有效batch还会改变每轮更新组数及调度步数。可读取已有 `reports/epoch_XXXX.json` 的 `train_seconds`、`consumed_samples_per_second`、`epoch_seconds` 和显存峰值；`epoch_seconds`包含训练与内部验证，不含随后保存检查点及最终分析。预检会预先缓存用于AMP/FP32对照的增强张量，它的短程吞吐不能代替持续在线数据管线的吞吐。共享PPU上还应记录同时段服务负载；本进程峰值显存不是整卡总占用，不根据一组数字承诺提速。

B0 输入顺序是确定性 FOV → 训练时水平翻转/双线性 expand 旋转/固定先亮度后对比度 → Pillow双线性等比缩放/黑色居中填充 → Tensor/0.5归一化。检测预览和最终画布都采用 round-half-up；FOV面积与前景覆盖不足时回退原图。旧配置/旧检查点缺少新版本字段时继续使用旧直接 Resize，绝不重新解释。训练、评估和加载后的推理复用同一变换。

CUDA训练使用FP16 autocast，BCE在autocast外使用FP32 logits计算。累积组按实际样本数归一化，组末unscale一次，有限梯度按全局范数5裁剪，溢出由标准GradScaler跳步并降scale。FP32非有限梯度立即失败；连续8组跳步终止。跳步组仍计入已消费样本和train BCE，但不计入成功组样本、不推进成功更新索引。`reports/updates.csv`逐组记录lr、scale前后值、范数、消费/成功样本和跳步；逐轮报告实际计数、吞吐和峰值显存。

以每轮预期累积组数G冻结调度：U=80G，Uw=5G。每组step前按j=此前成功更新数+1设置lr；第1次为1e-4，第Uw次为1e-3，第U次为1e-6，warmup后按文档余弦函数衰减。跳步复用同一索引；早停不压缩周期，轮数到预算即结束，不补跑跳步。

每轮内部开发用 `eval()` 和FP32，不额外完整评估训练集。未四舍五入的概率AUROC严格升高才保存best，相等保留较早轮次；patience参照仅在累计改善严格超过1e-4时重置，第30轮之前只禁止停止、不重置stale。logit的AUROC/AP单列用于饱和检查，不切换选模依据。AP保持average precision定义，不冒充其他积分方式的官方AUPRC。

正式运行产物继续使用现有目录结构：

- `reports/updates.csv`、`epoch_XXXX.json`、完成后的 `history.csv` 和 `metrics.json`：优化计数、曲线与内部开发指标。
- `checkpoints/epoch_XXXX.pt`：发生严格best改善时保存，包含优化器、scheduler/scaler及早停状态；完成后复制为 `best.pt`，另保存 `last.pt`。每份ResNet50训练检查点较大，需要预留磁盘；失败中途可保留已完成的 `epoch_XXXX.pt`，但未必已有最终别名。
- `checkpoints/best_inference.pt`：剔除优化器/训练状态的推理检查点，保留模型、配置和预处理规则；实际文件大小与参数张量字节数应分别看待。
- `predictions/dev_best.csv`：顺序与内部开发清单一致的 `image_id,label,logit,probability`。
- `reports/prior_reference.json`、`subgroup_rules.json`、`subgroups.json`、`image_proxies.csv`、`fov_summary.json`、`errors.csv`：训练先验参照、尺寸/曝光/清晰度代理子组、裁剪统计和假阴性/假阳性索引。子组边界只在稳定排序后均匀抽取的最多256个拟合样本上取中位数；它们是工程代理，不是临床质量标签，单类组AUROC为null。
- `config_snapshot.yaml`、`run_started.json`、`run_manifest.json`：配置、数据来源、代码/依赖、状态及失败位置/消费计数。`best`成绩属于内部开发选模，不能写成官方确认性能。

本次不实现/运行多种子复验或B0-fit训练入口。仅提供文档所述fit长度取整函数及边界测试，B0-fit仍需独立阶段实现，不能将 `epochs` 改短并同时压缩80轮调度。正式单图计时认证、训练收敛、真实FOV安全性和服务器资源适配均未由本次合成验证证明。

独立复算B0检查点时，`evaluate_checkpoint.py` 默认读取保存的内部开发清单并标记 `evaluation_source=internal_development`。若后续获准评估外部清单，必须同时显式指定 `--manifest` 和 `--evaluation-source`，避免把官方确认成绩与内部开发成绩混名；此处不执行该后续评估。旧检查点仍沿用原配置的评估来源。

## PPU 后四卡 DDP 运行

**2026-10-20 精度修复：服务器后续操作优先使用文末“PPU 四卡 FP32/BF16 精度路线”。本节的旧 FP16 配置和预算说明保留用于兼容、解释已有记录；更新后的混合精度预检已改为先执行 FP32 对照、再执行一次目标精度更新。**

本节对应研究方案第十三章。复用 `qinghao_ppu_train` 的 PPU 适配环境，保留其 `CUDA_VISIBLE_DEVICES=4,5,6,7` 和代理设置，不覆盖 PyTorch/torchvision/SDK。容器内逻辑 `cuda:0..3` 对应物理 PPU 4..7，每张卡一个训练进程；DeepSeek 服务继续运行，显存和计算资源属于共享资源。

进入容器、项目目录，并确保导入的是本次更新的源码：

```bash
# 宿主机执行
podman exec -it qinghao_ppu_train bash

# 以下在容器内执行；不要重新设置 CUDA_VISIBLE_DEVICES
cd /home/KanglongTan/QinghaoLi/Eye_Qing_Guang
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import eye_qing_guang; print(eye_qing_guang.__file__)"
```

独立配置为 `configs/b0_resnet50_512_dev_ddp4_ppu_v1.yaml`：每卡 batch=16、累积=2，全局有效 batch=128；每卡 workers=4，总计16；验证每卡batch=16、FP32。`training.device: cuda` 的索引由 `LOCAL_RANK` 绑定，不能改成固定 `cuda:0`。普通 BatchNorm 仍使用每卡微批次统计，与此前单卡 batch=64 有差异。峰值学习率保持1e-3，不自动乘四。默认每10个全局更新组输出一次终端进度，`updates.csv` 仍逐组记录。

第一步只检查四卡通信，不打开真实图像或创建训练目录：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m eye_qing_guang ddp-check \
  --config configs/b0_resnet50_512_dev_ddp4_ppu_v1.yaml
```

仅使用现有 PPU 的 `nccl` 兼容后端；后端不可用、设备限制不符或进程数不是4时直接失败，不回退CPU或单卡。此命令覆盖本流程需要的设备张量通信类型，但不替代目标模型/精度预检。若环境要求另一厂商后端，先核实后修订支持，不安装普通NVIDIA包覆盖现有栈。

开发清单沿用 `data/manifests/glaucoma_b0_dev_ppu_v1/`。已存在且来源核对通过时直接使用，不能覆盖重建。若尚未生成，仅单进程执行：

```bash
python -m eye_qing_guang development \
  --config configs/b0_resnet50_512_dev_ddp4_ppu_v1.yaml
```

训练清单数量必须能被4整除，否则停止，不通过填充重复样本或丢弃尾部维持运行。验证采用无重复分片，汇总完整开发集后计算 AUROC/AP；不平均四张卡的局部指标。只有rank 0写公共输出，权重保存时移除DDP包装，仍支持原单卡评估/测试预测入口及显式清单约束。

四卡数值预检默认仍限制为全局64个样本实例、最多4个全局组，FP32反向参照也计入。当前目标配置单个组已需128个实例，因此直接使用默认预算会在训练前拒绝。若使用独立每卡batch=4、累积=2的小配置，AMP32+FP32参照32可以落在原预算内，但不能证明目标batch=16已通过。

目标配置的一个AMP组及一个FP32参照最少共256个实例。**仅在明确采用该扩展预检预算时**执行以下命令；本次实现没有执行它：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m eye_qing_guang preflight \
  --config configs/b0_resnet50_512_dev_ddp4_ppu_v1.yaml \
  --name b0_resnet50_512_preflight_ddp4_ppu_s42_v1 \
  --max-sample-instances 256 --max-groups 2
```

此预算在首个AMP组成功时足够；若发生溢出跳步而缺少FP32参照预算，则报告 `inconclusive` 并以非零退出码结束，不自动扩大预算、降scale重喂同组、切换精度或用新目录反复追加尝试。AMP与FP32使用同一增强张量、相同初始化和微批次划分，并复用既有公开数值容差；目标配置显存、参数更新一致性及单进程加载也在预检范围内。预检吞吐来自缓存的增强张量，不是在线训练速度测量。

完成必要预检、人工输入检查并确定正式运行后，再执行下面的正式训练命令，不能将文档中的命令示例视为已运行：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m eye_qing_guang train \
  --config configs/b0_resnet50_512_dev_ddp4_ppu_v1.yaml
```

正式目录为 `experiments/b0_resnet50_512_dev_ddp4_ppu_s42_v1/`，拒绝覆盖已有目录。80轮调度、最短30轮及patience=15规则沿用B0；更新数按全局成功优化计数，AMP跳步不推进调度。日志中的消费/成功组样本数是四卡总量，更新数不会乘四。每轮的 `train_seconds` 为最慢rank训练时间，吞吐为全局消费数/该耗时；记录验证及汇总、检查点保存、各rank峰值显存。`epoch_XXXX.json` 的 `epoch_seconds` 截至公共轮次报告写入之前；`history.csv` 每轮追加并刷新，另含 `report_write_seconds` 和包含报告写入及随后同步的 `complete_epoch_seconds`，取最慢rank的轮次耗时，不含该行计时记录自身的写入。中途暂停或失败时，已完成轮次的计时仍保留在该CSV中，不等待正常训练结束。四卡配置的安全worker启动方式也适用于不分片的单进程权重回测加载器，默认使用spawn。

长时间的最终图像代理/错误分析在释放训练进程组后由rank 0执行，避免其他rank因等待CPU分析超过通信超时。失败时保留已有输出，并尽力写 `reports/failure_rank_XXXX.json`；强制杀进程可能来不及写最终清单，应结合 `run_started.json`、逐组CSV、启动器错误及rank失败记录判断完成程度。旧版未提供断点恢复；完整epoch检查点续训现已通过文末的 `--resume-from` 支持，始终使用新目录。

服务器通信、ResNet50/PPU AMP和同全局batch下的单卡/四卡速度比较，均不能由本地CPU/Gloo合成测试代替。四卡提速幅度以共享服务器上的实测为准，不预先承诺四倍加速。

### 每卡 batch=32、累积=1 的独立候选

新增 `configs/b0_resnet50_512_dev_ddp4_ppu_b32a1_w4_v1.yaml`，仅相对四卡起点修改实验名称、每卡batch（16→32）和累积次数（2→1）；workers仍为每卡4，FP32验证batch仍为每卡16。全局有效batch保持128，普通BatchNorm的微批次统计范围会变化，因此独立报告结果。该候选尚未在PPU运行，不承诺提速。

设备状态中的低利用率和少量显存占用需要结合日志是否持续推进判断。先确认当前运行属于预检还是正式训练、有无新更新或阶段变化；若进程卡在加载/通信，先定位原因。不要在同四张卡的原任务仍运行时直接叠加新训练。CPU总占用偏低也不排除少量加载进程或存储等待成为瓶颈。

当前运行正常推进且已结束，或用户明确暂停并保留已有产物后，再将候选同步至服务器。目标配置的一次AMP组加FP32反向参照仍至少需要256个全局样本实例、2个全局组；预检使用独立名称 `b0_resnet50_512_preflight_ddp4_ppu_b32a1_w4_s42_v1`。原预检结果和输出目录不覆盖。预检通过不代表在线吞吐已改善，后续按相同数据、精度、共享服务负载及全局batch比较 `history.csv` 中训练吞吐和完整轮次耗时。

<a id="ppu-precision-v2"></a>

## PPU 四卡 FP32/BF16 精度路线（2026-10-20）

原预检出现一次 FP16 梯度溢出后成功更新，但用完256实例/2组，未执行FP32梯度对照。本次改为显式精度和固定工作量，不依赖增加预算或重复尝试。旧配置缺少 `training.precision` 时仍按 `amp=true → fp16`、`amp=false → fp32` 解释；新字段目前仅用于四卡B0配置，旧单卡协议保持不变。

| 新配置 | 精度字段 | GradScaler | 每卡batch / 累积 / workers | 预检实际工作量 |
| --- | --- | --- | --- | --- |
| `configs/b0_resnet50_512_dev_ddp4_ppu_fp32_v2.yaml` | `precision: fp32`、`amp: false` | 禁用 | 16 / 2 / 4 | 128实例、1组FP32更新 |
| `configs/b0_resnet50_512_dev_ddp4_ppu_bf16_v2.yaml` | `precision: bf16`、`amp: true` | 禁用，BF16 autocast仍启用 | 16 / 2 / 4 | 128实例FP32对照 + 128实例BF16更新，共256实例/2组 |

两配置保持同一内部划分、ResNet50-512、全局有效batch128、普通每卡BatchNorm、学习率和成功更新调度。精度变体使用独立正式目录，不与原FP16结果混名。BF16先执行小型卷积/反向算子检查，再由实际ResNet50预检验证；不单凭设备能力标志判断支持，不支持时不会静默切换精度。

混合精度预检先保存初始化状态，对一组缓存的增强张量执行FP32前反向，然后恢复相同权重/BatchNorm状态，在相同物理微批次上执行一次目标精度更新。损失/梯度容差沿用原公开值，不为通过检查而放宽。FP32基线不做与自身的混合精度对照，报告 `precision_comparison.status=not_applicable`；参数更新、一致性和权重加载均通过后，总体仍为 `status=completed`。混合精度还必须 `precision_comparison.status=passed`。

预算不足会在初始化设备和模型前拒绝；即使提供更大上限，也只执行上述固定工作量。FP16更新溢出时保留已完成的FP32对照，报告 `fp16_gradient_overflow_after_completed_fp32_reference`、退出码2，不重试、不增加预算。BF16/FP32非有限梯度、算子不支持、精度超差或权重加载异常均直接失败并保留阶段记录；正式训练不会自动开始。

预检终端逐阶段输出 started/done 和耗时，最终 `reports/preflight.json` 包含 `training_precision`、`stage_seconds`、FP32对照及目标训练峰值显存、更新记录和检查点加载结果。用于检查权重的小型确定性FP32推理采用workers=0，避免每次预测重建spawn池；训练数据加载仍使用配置中的每卡4个workers。缓存张量吞吐不能代表在线训练速度。

将更新后的 `src/` 和 `configs/` 同步到已挂载的项目，无需重建容器或重装训练依赖。以下命令在原个人容器内执行，保留 `CUDA_VISIBLE_DEVICES=4,5,6,7`：

```bash
cd /home/KanglongTan/QinghaoLi/Eye_Qing_Guang
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# 1. 先确认四卡FP32更新、参数一致性与检查点加载。
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m eye_qing_guang preflight \
  --config configs/b0_resnet50_512_dev_ddp4_ppu_fp32_v2.yaml \
  --name b0_resnet50_512_preflight_ddp4_ppu_fp32_s42_v2 \
  --max-sample-instances 128 --max-groups 1
```

仅在上一条退出码为0、报告总体 `status=completed` 时，执行BF16检查：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m eye_qing_guang preflight \
  --config configs/b0_resnet50_512_dev_ddp4_ppu_bf16_v2.yaml \
  --name b0_resnet50_512_preflight_ddp4_ppu_bf16_s42_v2 \
  --max-sample-instances 256 --max-groups 2
```

BF16预检通过后，正式训练命令如下。若BF16不支持或精度对照失败、而FP32预检已经通过，可直接采用下面列出的FP32路线，不继续调整预算。两条正式训练命令二选一，不同时运行：

```bash
# BF16路线：仅在对应BF16预检通过后使用。
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m eye_qing_guang train \
  --config configs/b0_resnet50_512_dev_ddp4_ppu_bf16_v2.yaml

# FP32路线：对应FP32预检通过后可使用。
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m eye_qing_guang train \
  --config configs/b0_resnet50_512_dev_ddp4_ppu_fp32_v2.yaml
```

FP32预检本身异常时，不继续启动训练，应根据失败阶段处理。目录已存在时命令会拒绝覆盖，先阅读该次报告，不通过改名反复运行。以上为服务器操作说明，不表示本次已执行PPU预检或正式训练；本地合成测试不能保证PPU自定义算子兼容或长期收敛。

<a id="ppu-resume-diagnostics-v3"></a>

## 四卡从第17轮续训及卡点诊断（2026-10-20）

使用 `configs/b0_resnet50_512_dev_ddp4_ppu_fp32_resume17_diag_v3.yaml`。按用户最终要求，它完整保留已运行实验快照中的训练、数据和分布式参数：每卡batch128、累积1、验证batch128、每卡workers8、FP32、全局有效batch512、80轮调度及原学习率。只增加诊断选项并采用新的实验名称；原实验、权重和日志不覆盖。

续训入口只接受完整epoch的四卡format-2训练检查点，恢复模型/BN、AdamW一二阶矩与step、scaler、成功更新调度和早停状态；核对数据身份、每轮组数及累计样本数。v3配置保持原全局batch；下文v4支持显式改变batch并迁移剩余调度。不接受只有权重的推理检查点，不自动扩展轮数或重置warmup。旧检查点中的 `resume_supported=false` 表示旧入口没有实现续训，不代表训练状态不可用；新入口会核验实际内容，禁止手工修改标志绕过核验。

`epoch_0017.pt` 应恢复1377次成功更新、698972个全局消费样本，从epoch18重新开始；第一次新更新索引1378，学习率约 `0.0009380906869793779`。旧epoch18未保存的64次更新会重做。旧检查点没有各rank随机数和DataLoader状态，因此明确记录 `exact_replay=false`，不承诺逐位复现原轨迹。

先将本次更新的 `src/` 和 `configs/` 同步到服务器原项目，保留原检查点文件；无需重建容器或重装PyTorch。以下在个人容器中执行，不覆盖 `CUDA_VISIBLE_DEVICES`：

```bash
cd /home/KanglongTan/QinghaoLi/Eye_Qing_Guang
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m eye_qing_guang train \
  --config configs/b0_resnet50_512_dev_ddp4_ppu_fp32_resume17_diag_v3.yaml \
  --resume-from experiments/b0_resnet50_512_dev_ddp4_ppu_fp32_s42_v2/checkpoints/epoch_0017.pt
```

新目录固定为 `experiments/b0_resnet50_512_dev_ddp4_ppu_fp32_resume17_diag_s42_v3/`。启动会核验并恢复训练状态，再对继承的best权重做一次内部开发集预测，确保没有新best时仍可生成正确的最终产物；这是验证前向，不是重新训练前17轮。随后终端应出现 `Resumed epoch 17; next epoch=18; successful_updates=1377` 和 `Epoch 18` 进度。

新目录的history/updates只记录新执行的epoch18及以后，累计更新/样本计数承接旧检查点；run_started/run_manifest记录来源与新增完成轮数。AUROC选best与patience规则不变。为使后续完整轮次均可恢复，四卡训练现在每轮保存独立 `checkpoints/epoch_XXXX.pt`，即使没有改善；`best.pt`仍按AUROC选择，不等同于最新epoch。继承的best文件会复制到新目录，原始文件内容保留。每份训练权重约270MiB，需为后续轮次保留磁盘空间。

诊断产物如下：

| 文件 | 内容 |
| --- | --- |
| `reports/diagnostics_rank_0000.jsonl` 至 `0003` | 各rank的epoch、累计组号、微批次、取数/传输/前向/反向/更新阶段；取数前预期ID、取数后实际ID；关键点的本进程allocator显存计数 |
| `reports/stacks_rank_0000.log` 至 `0003` | 任一阶段连续60秒未推进时自动输出Python线程调用栈；发生Python异常时追加异常栈。输出栈本身不会终止训练 |
| `reports/device_snapshots.jsonl` | rank0后台每10秒执行一次现有 `nvidia-smi` 或 `ppu-smi`，保存整卡及DeepSeek等进程原始占用；单次命令最多5秒，失败记录错误而不阻塞训练 |
| `reports/failure_rank_XXXX.json` | 可捕获的异常与该rank进度；强制abort可能来不及写此文件 |

启用诊断时，多进程DataLoader等待上限为90秒，小于180秒通信超时，尽量先暴露实际取数失败；workers0时不设置DataLoader超时。诊断不主动同步每个GPU算子，不静默改变训练算法；Python调用栈能指出停留的调用，但不能保证展开厂商C++内核内部。首次算子初始化等长操作也可能触发栈输出，应结合阶段日志判断。

若再次出错，保留整份新目录及完整终端日志。先比较各rank最后阶段和堆栈，再查看同时间的整卡显存：可区分DataLoader等待、设备操作/同步等待与外部显存变化，不能仅凭NCCL报错认定DeepSeek导致OOM。此实现不替代PPU实机长期稳定性验证，也不会自动重启失败任务。

## 第32轮后改为每卡104续训（v4）

使用 `configs/b0_resnet50_512_dev_ddp4_ppu_fp32_resume32_b104_v4.yaml`：每卡训练/验证batch均为104、累积1、workers8、四卡FP32，其余训练参数与v3相同。训练集每卡10279张，每轮99个更新组，全局完整组batch416；每卡最后一组87张，四卡共348张，仍按实际组样本数归一化。原配置和产物保留，新目录固定为 `experiments/b0_resnet50_512_dev_ddp4_ppu_fp32_resume32_b104_s42_v4/`。

变batch续训保持模型/BN、AdamW moments/step、早停、累计成功更新/跳过更新及样本计数。学习率不按batch比例缩放、不重置warmup：保留旧计划的下一次成功更新学习率，将旧计划尚未执行的曲线映射到新剩余成功更新预算。第32轮检查点若已成功更新2592次，则从第33轮开始计划48×99=4752次新更新，累计调度终点7344；历史2592次更新不改写成32×99。最后一个计划更新达到原 `min_lr`；跳过更新不推进调度，因此后续若有跳过，在第80轮结束时不保证已达到最低学习率。提前停止仍按原规则执行。

检查点保存迁移前曲线和更新计数起点，支持再次按相同或不同batch续训；再次改变每轮组数时重新映射尚未执行的曲线。原方案尾部和新尾部都需至少两个计划更新，否则拒绝变组数迁移。新运行的 `run_started.json` / `run_manifest.json` 中 `resume.batch_transition` 记录源/目标全局batch、每轮组数、调度终点与下一步学习率。原始RNG/DataLoader状态缺失，且batch/普通BatchNorm统计发生变化，此续训属于改变训练配方后的延续，不是原128配置的等价重放。

同步更新的 `src/` 和新配置后，在原容器中执行；源权重必须是服务器上已保存的完整第32轮训练检查点：

```bash
cd /home/KanglongTan/QinghaoLi/Eye_Qing_Guang
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m eye_qing_guang train \
  --config configs/b0_resnet50_512_dev_ddp4_ppu_fp32_resume32_b104_v4.yaml \
  --resume-from experiments/b0_resnet50_512_dev_ddp4_ppu_fp32_resume17_diag_s42_v3/checkpoints/epoch_0032.pt
```

启动应显示 `Resumed epoch 32; next epoch=33; successful_updates=2592` 和 `Resume batch/schedule` 信息。每轮继续保存完整检查点与诊断记录。已有目标目录会拒绝覆盖。减小batch用于尝试降低资源压力，不保证消除共享PPU设备或厂商兼容栈中的执行/通信停顿。

## 独占四卡从头重训候选配置（v6）

用户选择独占物理卡4～7、继续四进程DDP并从第1轮重训。使用 `configs/b0_resnet50_512_dev_ddp4_ppu_fp32_retrain_b256_w16_v6.yaml`，每卡训练/验证batch256、累积1、workers16（总计64），FP32、原始冻结划分和seed42不变，学习率峰值仍为0.001。通信超时900秒，诊断栈60秒、取数超时90秒；延长通信等待只是容错，不证明底层卡顿已修复。

41116张训练图像在四卡上各10279张：每轮41组，全局完整batch1024，尾组每卡39张；从零开始的调度共3280次成功更新、前205次warmup。早停与每轮检查点规则不变，绝不加载第17/32轮或其他旧权重。该配置是新的大batch训练配方，与旧batch128训练并不等价。

batch128旧运行的第32轮训练耗时约281.18秒，rank0阶段日志中取数等待约221.55秒；因此workers16作为改善供数的候选，而不是已证实的提速参数。batch256能否放入144GiB、吞吐是否提高，均需在PPU上验证。先执行一个全局完整更新的FP32预检；1024为全局训练样本实例预算，不含验证前向，预检不作为稳态吞吐或长期稳定性证据：

```bash
cd /home/KanglongTan/QinghaoLi/Eye_Qing_Guang
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m eye_qing_guang preflight \
  --config configs/b0_resnet50_512_dev_ddp4_ppu_fp32_retrain_b256_w16_v6.yaml \
  --name b0_resnet50_512_preflight_ddp4_ppu_fp32_b256_w16_s42_v6 \
  --max-sample-instances 1024 \
  --max-groups 1
```

仅在预检正常完成且无数值/显存错误后，启动从头重训；不要传 `--resume-from`：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  -m eye_qing_guang train \
  --config configs/b0_resnet50_512_dev_ddp4_ppu_fp32_retrain_b256_w16_v6.yaml
```

正式输出固定为 `experiments/b0_resnet50_512_dev_ddp4_ppu_fp32_retrain_b256_w16_s42_v6/`，预检目录与其分离，旧运行保留。首轮和后续轮次的取数等待、吞吐和显存是决定是否继续使用该候选的依据；不能把更多空闲显存或更大batch直接当作已验证的训练加速。本地只检查配置和调度，不自动启动服务器训练。

## timm V2-S 官方 train/valid 实验（2026-10-22）

入口为 `scripts/run/run_timm_v2s_experiment.py`，配置为
`configs/timm_v2s_in21k_ft_in1k_512_pw5_official.yaml`，协议独立为
`timm_official_trainvalid_v1`。它不调用 B0 开发划分、B0 编排或 B0 配置校验器。
官方 train/valid/test 数量分别固定为 51396/5849/5855；无填充、无丢样本，原始 ID/标签/顺序须逐项匹配。
清单不存在时创建，存在时只读校验来源后复用。预检只解码有限 train/valid 图像，不预测真实 test。

实现包含 `tf_efficientnetv2_s.in21k_ft_in1k`、1280 维编码器、单 logit 分类头和全参数微调。
RGB/FOV/BILINEAR/letterbox 输入为 512，mean/std 都是 0.5；训练增强沿用方案的翻转、旋转、亮度与对比度。
固定正类权重 5 的 FP32 BCE；训练 BF16 autocast，无 GradScaler，验证与提交使用 FP32。
只按完整官方 valid 的连续概率 AUROC 严格改善更新 best；并列保留较早轮。
AP、梯形 PR-AUC、硬标签 AUROC、高特异度工作点及混淆指标分开报告。
验证先统一到 rank0 模型/BN 状态，使用无补齐分片，再恢复各卡训练用本地 BN 缓冲区。

每轮只原子替换当前新 run 自己的 `checkpoints/best.pt` 与 `last.pt`。
它们是 `format_version=3 / checkpoint_kind=inference_weight`，没有优化器或续训状态；入口拒绝 `--resume-from`。
完整流程已编排为训练、best 重载验证及逐图比较、test 推理、提交序列化读回检查。
只有四阶段均成功才记录总 completed；失败保留当前产物，不自动重训或补跑。
正式提交位于 `experiments/<experiment.name>/submission/submission.csv`。

### B0 基础设施与本实验训练配方

本版明确采用已在服务器验证的 B0 四卡训练基础设施：外部 torchrun、四个 local rank、
CUDA_VISIBLE_DEVICES=4,5,6,7、NCCL 兼容后端、DistributedDataParallel、spawn 数据加载器，
以及训练器的 no_sync 梯度累积和 successful-update warmup/cosine 调度。
`utils/timm_source.py` 是薄封装，直接调用 `utils/distributed.py` 中的现有初始化和 DDP 包装；
它不再寻找外部 gloo_staged 实现，也不声称复现其他实验的 staging 语义。

新模型和实验协议保持独立：timm EfficientNetV2-S、官方 train/valid/test、512 预处理、
pw5 BCE、BF16、每卡 batch 8/累积4、native local BN、全量官方 valid AUROC 选模，
以及只保留 best/last。B0 学习率/优化器工程参数在本配置里被显式选作本次配方：
AdamW、betas=(0.9,0.999)、eps=1e-8、全参数统一参数组、梯度裁剪5.0、
warmup起点1e-4、100轮成功更新调度、早停30轮/0.0001/15轮。
它们是本实现的明确选择，不再标注为缺失源引擎中核实出来的值。

此次改变是训练实现路线的决策更新：本实验使用 NCCL 原生 DDP，不继承 gloo_staged。
两者不是可以仅按名称互换的实现；本次以服务器已跑通的 B0 训练栈为可复用依据。
仍须通过目标设备的有限预检验证 timm 预训练权重载入、PPU算子、BF16和目标 batch；
本地 CPU 合成分布式测试不等价于这些目标设备验收。

### 环境、预检与验收边界

新增可选依赖 `timm-experiment = timm==1.0.22`，版本取自本机已有包并已检查实际模型结构。
未安装或更换任何服务器 torch/torchvision/PPU SDK；PPU 兼容性需要在目标环境单独确认。
初始化仅接受 `model.pretrained_file` 指定的已准备文件，或现有 HF 缓存中的同名权重；代码不自动下载。
重载时 pretrained=False，严格读取检查点自己的配置和权重，不依赖原预训练文件继续存在。

准备本地预训练权重、核对服务器路径并完成有限预检后，四进程入口为：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/run/run_timm_v2s_experiment.py \
  --project-root /实际代码项目绝对路径 \
  --config configs/timm_v2s_in21k_ft_in1k_512_pw5_official.yaml \
  --preflight-only
```

默认独立预检目录为 `experiments/_preflight/<experiment.name>_pf_v1/`；已有目录拒绝覆盖，
用 `--preflight-version pf_v2` 显式选择下个版本。所有相对路径锚定 project-root。
默认预算64，有限预检实际计划56个全局样本实例：16张训练增强缓存各做一次FP32参考/BF16更新，
以及8张正负均衡valid子集在更新后、best回载后、last回载后各评估一次。
梯度比较容差显式写入 preflight 配置；任一超差或非有限值停止，不重试或扩大预算。
默认只检查较小微批次，报告 target_batch_memory=not_verified，因此不会继续创建正式 run。
本配方的目标组需128张，含FP32参考和三次8张valid共需280个实例；更大真实预检须另行授权，
获准后才通过显式 `--max-sample-instances 280` 与新预检版本执行。
正式入口省略 `--preflight-only`，复用同配置、同设备环境的合格目标组记录后重新初始化正式模型与随机状态。

预检报告的 `peak_memory_local_rank.allocated_bytes/reserved_bytes` 记录该 worker 的峰值显存，
`max_reserved_bytes` 为各 rank 预留峰值的合计。它只代表这一次预检的实际物理微批次，
不能直接换算成更大的 batch；若计划加大 batch，先在独立预检版本上重测。
正式训练期间 rank0 每 `distributed.log_every_n_groups` 个累积组打印一次本地已消费样本、
损失、学习率与吞吐，便于区分取数等待和算力瓶颈。

相关离线合成测试为 `tests/test_timm_official.py` 与 `tests/test_timm_distributed.py`。
前者覆盖模型接口、清单、损失、文件、指标、阶段与提交规则；后者用四个 CPU/Gloo worker
检查DDP梯度数学、累积同步边界、无补齐验证、BN 恢复、错误广播和有界预检，不认证PPU/NCCL或BF16算子。
真实 timm 架构的合成32输入前后向另有独立测试，pretrained=False，不算预训练载入或512目标批量验收。
旧测试缺失的历史配置从 Git fac2347 原样固化到 tests/fixtures/legacy_configs，仅用于回归。
本轮未读取真实图像、未下载预训练权重、未进行真实预检/完整训练/官方test预测或平台上传；T4延迟未测。

### 单张 RTX 3090 训练（2026-10-23）

新增配置 `configs/timm_v2s_in21k_ft_in1k_512_pw5_official_single3090.yaml`，
以普通 Python 单进程运行，不初始化进程组、不包装 DDP、不执行 NCCL 通信。
原四卡配置与 torchrun 入口继续保留。单卡仍使用相同模型、官方 train/valid/test、
512 预处理、pw5、全参数微调、验证选模和训练后自动生成提交的流程。

初始配置为训练 microbatch 8、累积16次、验证 batch16、workers4、BF16，
完整更新的有效 batch 为 `1 × 8 × 16 = 128`。每轮处理全部51396张训练图，
最后不足一组的样本仍参与更新，按实际样本数归一化。
累积不会合并 BatchNorm 的微批次统计，因此有效 batch 相同不代表与四卡训练数值等价。
这些是待在24GB设备上验证的起始参数，代码不会自动切换精度或减小 batch。

`data.raw_root` 暂沿用此前3090服务器的
`/data2/liqinghao/Eye_Qing_Guang/data/raw/glaucoma_first`，运行前请核对新服务器实际位置；
该目录下应直接包含 train.csv、valid.csv、test.csv 及相应图像目录。
新配置使用独立 manifest 目录，避免复用PPU服务器上的绝对图像路径。
环境需有兼容3090驱动的 PyTorch/CUDA 构建、`timm==1.0.22`，以及
`model.pretrained_file` 指定的本地权重或已有HF缓存；代码不会自动下载权重。

进入实际代码项目目录后执行以下直接训练命令。物理卡编号2仅为示例，
务必替换为分配给自己的卡号；进程内 `cuda:0` 会映射到这张物理卡：

```bash
cd /实际代码项目目录
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u scripts/run/run_timm_v2s_experiment.py \
  --project-root "$PWD" \
  --config configs/timm_v2s_in21k_ft_in1k_512_pw5_official_single3090.yaml \
  --skip-preflight
```

此命令直接开始正式训练，不执行预检。若使用可选预检，仍需独立选择预检目录版本和预算；
默认小批量预检不证明目标 batch 的显存可用。单卡配置拒绝多rank启动。
输出固定为 `experiments/timm_v2s_in21k_ft_in1k_512_pw5_official_single3090_s42_v1/`，
已有目录拒绝覆盖；完整流程成功后提交文件为该目录的 `submission/submission.csv`。
本次没有增加断点续训能力，best/last仍是仅供推理的format3权重。
本地验证覆盖单卡合成流程与原四进程CPU/Gloo回归，不代表真实3090、BF16、预训练权重或完整数据验收。

### V2-S 单卡离线预处理缓存（2026-10-23）

新配置 `configs/timm_v2s_in21k_ft_in1k_512_pw5_official_single3090_cache_v1.yaml`
显式启用 `data.cache`，原四卡和原单卡配置继续从原图执行旧版在线处理。
原始目录为 `/data2/liqinghao/Eye_Qing_Guang/data/raw/glaucoma_first`，缓存目录为
`/data2/liqinghao/Eye_Qing_Guang/data/processed/glaucoma_first_fov512_cache_v1`。
缓存可以被不同训练进程数共用；不要让多个训练rank重复构建。

训练缓存为FOV裁剪、最长边缩放至512的RGB无损PNG，保持长宽比、不提前补边；
运行时继续随机翻转、旋转、亮度/对比度，随后letterbox至512及Normalize。
valid/test提前完成FOV与letterbox，运行时只转Tensor和Normalize。
预处理版本为 `fov_presized_letterbox_v2`：训练先缩小再增强，与旧版增强顺序有差异，
因此使用独立实验名从头训练。验证/测试保持旧版固定处理的像素结果。

在服务器实际项目根目录、已准备好的Python环境中执行一次CPU缓存生成：

```bash
python -u scripts/preprocess/build_offline_cache.py \
  --project-root "$PWD" \
  --config configs/timm_v2s_in21k_ft_in1k_512_pw5_official_single3090_cache_v1.yaml \
  --workers 4
```

`--workers`是缓存生成进程数，与训练的 `data.num_workers` 无关；0为串行。
程序先核对官方数量51396/5849/5855及源文件，再检查每个split前两张图片，
通过后处理全部63100张。每500张打印进度，末尾输出 `Cache completed`。
无需GPU、timm模型或预训练权重，不启动实验训练、不进行预测。
只生成图像，不使用test标签；原始数据不变。

`index.csv`保存原始ID/划分/标签与缓存PNG的对应关系，PNG按split和序号命名，
提交仍保留原始ID及扩展名；`metadata.json`记录处理版本、Pillow版本、数量与完成状态。
已有缓存目录拒绝覆盖；生成失败保留部分文件并拒绝训练使用，重新构建应选择新的
`data.cache.root`版本目录，不会自动删除或续写旧目录。
缓存与原始图片/CSV需同时保留，源路径、大小或修改时间变化时拒绝使用，避免读取过期缓存。
缓存启用但文件缺失时直接报错，不静默退回在线处理。

确认缓存生成成功、预训练权重已在本地文件或HF缓存中准备好后，可直接训练。
以下物理卡2仅为示例，替换成分配给自己的卡号：

```bash
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u scripts/run/run_timm_v2s_experiment.py \
  --project-root "$PWD" \
  --config configs/timm_v2s_in21k_ft_in1k_512_pw5_official_single3090_cache_v1.yaml \
  --skip-preflight
```

默认训练batch8、累积16、验证batch16、workers4、BF16不变。
训练、每轮验证、best重载验证及test提交全部通过同一缓存读取入口。
输出为 `experiments/timm_v2s_in21k_ft_in1k_512_pw5_official_single3090_cache_s42_v1/`，
完整流程成功后包含 `submission/submission.csv`，已有实验目录拒绝覆盖。
检查点保存新版处理配置；在另一环境需要读取原图时，可显式将同版本配置的
`data.cache.enabled`设为false，通过相同v2固定处理恢复输入，程序不自动切换。
此功能尚未在服务器全量数据上执行，不承诺实际提速比例；PNG仍需解码，随机增强仍在线执行。

### ConvNeXt-Tiny 单卡与共享缓存（2026-10-23）

新增原图单卡配置 `configs/timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official_single3090.yaml`，
以及缓存单卡配置 `configs/timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official_single3090_cache_v1.yaml`。
原四卡配置保持不变。单卡使用普通Python进程，不初始化进程组或DDP/NCCL，
通过 `CUDA_VISIBLE_DEVICES` 选择分配给自己的物理卡，进程内使用 `cuda:0`。

两份新配置均为batch8、累积16、验证batch16、workers4、BF16，有效batch128；
原来的AdamW、学习率、pw5、官方划分、早停和AUROC选模规则保留。
这些batch参数尚未在真实3090上验收，不会自动降级精度或batch。
ConvNeXt始终使用mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]，
模型为 `convnext_tiny.fb_in22k_ft_in1k`，不会使用V2-S的模型权重或Normalize。

缓存配置直接复用 `/data2/liqinghao/Eye_Qing_Guang/data/processed/glaucoma_first_fov512_cache_v1`，
因为缓存为Normalize之前的RGB图，缓存标识不依赖模型或mean/std。
V2-S已生成并完成的缓存无需重新生成；ConvNeXt使用独立manifest与实验目录。
原始图片与CSV仍需保留，源数据路径为 `/data2/liqinghao/Eye_Qing_Guang/data/raw/glaucoma_first`。
训练继续在线随机增强，固定预处理采用已记录的v2顺序，验证及提交读取相同缓存版本。

如果缓存尚未生成，可以使用前述V2-S命令，也可使用同一个构建程序传ConvNeXt配置：

```bash
python -u scripts/preprocess/build_offline_cache.py \
  --project-root "$PWD" \
  --config configs/timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official_single3090_cache_v1.yaml \
  --workers 4
```

已有缓存目录会拒绝覆盖，不能再次执行生成命令覆盖已完成的V2-S缓存。
准备好 `timm==1.0.22` 和ConvNeXt自己的预训练权重后，在项目根目录运行：

```bash
# 物理卡2仅为示例，替换为分配给自己的卡号。
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u scripts/run/run_timm_convnext_experiment.py \
  --project-root "$PWD" \
  --config configs/timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official_single3090_cache_v1.yaml \
  --skip-preflight
```

该命令跳过预检直接训练。`model.pretrained_file: null`仅查已有HF缓存；
独立权重文件需填入此字段，程序不会自动下载。
完整流程结束后输出
`experiments/timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official_single3090_cache_s42_v1/submission/submission.csv`。
如果选择原图单卡配置，输出名中不含cache，走旧版在线预处理。
现有best/last仍为format3推理权重，没有增加续训能力；已有输出目录拒绝覆盖。
本地以合成模型验证控制流程、梯度与缓存复用，不代表真实ConvNeXt预训练载入或3090/BF16验收。
