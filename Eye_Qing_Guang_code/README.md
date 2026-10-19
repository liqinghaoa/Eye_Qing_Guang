# 青光眼二分类实验工程

本项目使用 PyTorch 完成眼底图像二分类，支持图像基线和可选的年龄特征融合。工程参考 `skin_optics` 的组织方式：配置描述实验、脚本解析命令、Python 包实现可复用功能，每次实验独立保存结果。

已接通数据审计、样本清单生成、训练、检查点评估、无标签测试集预测、提交文件转换和可视化。项目搭建只使用合成样本验证流程，尚未运行正式比赛数据训练，未产生模型性能结论。

新增的 **B0-dev ResNet50-512** 使用独立内部开发协议，服务器步骤见本文末尾“B0-dev 服务器运行”。下方原始 ResNet18 快速流程继续保留，两者的数据划分和优化预算不同。

## 目录

```text
Eye_Qing_Guang_code/
├── configs/
│   ├── resnet18.yaml                 # 初始基线，兼容旧实验
│   └── b0_resnet50_512_dev_v1.yaml    # B0 内部开发实验
├── data/
│   ├── raw/glaucoma_first/           # 原始图像、官方 CSV 及原始划分
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

要求 Python 3.10+，推荐独立 Python 3.11/3.12 环境。GPU 训练应先按 PyTorch 官方说明安装与驱动匹配的 `torch`、`torchvision`，再安装本项目。以下命令在项目根目录执行：

```powershell
cd E:\projects\Eye_Qing_Guang\Eye_Qing_Guang_code
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv pip install -e .
```

Linux 也可使用 `python -m venv .venv`、`source .venv/bin/activate`、`python -m pip install -e .`。

已有完整依赖环境可使用 `python -m pip install --no-deps -e .`。开发时也可以将 `src` 加入 `PYTHONPATH`，无需修改其他项目的环境。当前机器默认 Anaconda Python 导入 PyTorch 时存在 OpenMP 冲突；请使用可正常导入 PyTorch 的独立环境，不在项目代码中设置绕过冲突的环境变量。

安装后，脚本与统一命令等价：

```powershell
python scripts/train/run_binary_classification.py --help
eye-qing-guang train --help
python -m eye_qing_guang.cli train --help
```

## 数据与评估约定

当前 `data/raw/glaucoma_first/` 内有官方 `train`、`valid`、`test` 三套划分，对应 51,396、5,849、5,855 张图像。训练和验证表包含 `Image_name, Age, True_Label`，测试表只有 `Image_name, Age`。

- 清单保留官方 `Image_name` 作为 `image_id`；实际文件路径写入 `image_path`，解决 CSV 的 `.jpg` 和磁盘 `.JPG` 大小写差异。
- 标签只接受 `0/1`。模型输出一个 logit，`sigmoid(logit)` 表示 `True_Label=1` 的概率，临床语义以比赛说明为准。
- 基线仅使用图像。设置 `model.use_age: true` 可启用年龄融合；缺失年龄以训练集年龄中位数填充，再用训练集均值、标准差归一化，统计量随检查点保存。
- 初始 ResNet18 流程用官方 `valid` 早停；B0-dev 只在官方 train 内划分和选模，官方 valid/test 不进入开发训练。两套流程不自动互换。
- 当前模型选择指标是 ROC AUC，并报告 AP、Accuracy、Balanced Accuracy、F1、Precision、Sensitivity、Specificity 和混淆计数。这是可修改的基线设置，尚不代表已核实的比赛排名规则。
- 分类阈值来自配置，默认 `0.5`。只含单一标签的评估集，其 ROC AUC/AP 返回 JSON `null`；使用 ROC AUC 选模时要求验证集包含两类。

## 使用流程

配置中的相对路径、命令参数中的数据/输出路径都相对于项目根目录解析；可用 `--project-root` 指定另一份项目或数据布局。每条命令均支持 `--help`。以下命令供正式实验使用，搭建阶段不会自动执行它们。

### 1. 审计数据并生成清单

```powershell
python scripts/preprocess/audit_dataset.py --output data/manifests/audit_v1.json
python scripts/preprocess/build_manifest.py
```

审计检查必要列、标签、年龄、标识唯一性和图像路径；增加 `--decode-images` 可逐张验证图像解码。发现错误时立即停止并报告首个错误。清单生成后，`data/manifests/glaucoma_first_v1/` 包含 `train.csv`、`valid.csv`、`test.csv` 和 `summary.json`。已有清单版本拒绝覆盖；改版需同时修改 `data.manifest_dir`，或通过 `--output-dir` 指定新目录并同步配置。

### 2. 训练基线

```powershell
python scripts/train/run_binary_classification.py --config configs/resnet18.yaml --name resnet18_image_v1
```

支持 `resnet18`、`resnet50`、`efficientnet_b0`。`tiny_cnn` 供离线流程验证，必须设置 `pretrained: false`。默认 ResNet18 使用 ImageNet 预训练，首次正式训练可能下载权重；离线运行可在配置中关闭。当前未做模型优劣比较。

初始 ResNet18 流程使用 AdamW、BCE 或 Focal BCE，可选按训练集比例计算 `pos_weight`，采用单设备 FP32 和普通 patience。B0 使用后文的专用 AMP/累积/调度/早停逻辑。当前均不支持断点续训、DDP、自动调参或集成。

每次运行在启动前确定唯一实验目录，同名运行拒绝覆盖。改进轮保存 `epoch_XXXX.pt`，完成后额外保存 `best.pt` 和 `last.pt`。检查点包含模型/优化器状态、配置、年龄统计量、损失权重和轮次。逐轮进度写入 `reports/epoch_XXXX.json`，完成后生成 `history.csv` 和最佳指标 `metrics.json`。失败时保存 `run_manifest.json`，记录已完成的验证轮数和错误；进程被强制结束时可参考 `run_started.json` 和已有逐轮文件。

### 3. 评估与汇总

```powershell
python scripts/evaluate/evaluate_checkpoint.py --checkpoint experiments/resnet18_image_v1/checkpoints/best.pt --output-dir experiments/resnet18_image_v1/reports/valid_best
python scripts/evaluate/summarize_experiment.py --experiments experiments/resnet18_image_v1 --output experiments/summary_v1.csv
```

评估读取检查点内的模型和预处理配置，避免外部 YAML 改动导致不一致；支持 `--split train` 或默认 `valid`，用 `--manifest` 指定另一份同格式清单。输出指标、逐样本预测及运行记录。`best.pt` 的官方验证集成绩是选模成绩，不应解释为独立测试性能。

### 4. 测试集预测与提交转换

```powershell
python scripts/infer/predict_test.py --checkpoint experiments/resnet18_image_v1/checkpoints/best.pt --output-dir experiments/resnet18_image_v1/predictions/test_best
python scripts/submit/build_submission.py --predictions experiments/resnet18_image_v1/predictions/test_best/test_predictions.csv --output experiments/resnet18_image_v1/predictions/submission_v1.csv
```

内部预测字段为 `image_id,probability`。提交转换严格匹配官方测试 CSV 的标识和行序，检查缺失、多余、重复标识和概率范围。默认输出 `Image_name,probability`，**该格式是工程默认值，须确认官方提交模板后调整**。可在 `submission` 配置或 `--id-column`、`--value-column`、`--mode probability|label` 中指定字段和分数类型；此命令只生成本地 CSV。

### 5. 可视化

```powershell
python scripts/visualize/plot_training_curves.py --history experiments/resnet18_image_v1/reports/history.csv --output experiments/resnet18_image_v1/figures/training_v1.png
python scripts/visualize/plot_confusion_matrix.py --metrics experiments/resnet18_image_v1/reports/valid_best/metrics.json --output experiments/resnet18_image_v1/figures/confusion_v1.png
python scripts/visualize/generate_gradcam.py --checkpoint experiments/resnet18_image_v1/checkpoints/best.pt --image data/raw/glaucoma_first/valid/valid_00000.JPG --age 61 --output experiments/resnet18_image_v1/figures/gradcam_v1.png
```

Grad-CAM 默认解释标签 1；`--class-index 0` 解释标签 0。年龄融合模型的 `--age` 应使用样本实际年龄，省略时按训练集缺失值规则处理。输出为缩放后图像上的 PNG 热图，用于模型检查。

## 验证和版本控制

```powershell
python -m unittest discover -s tests -v
```

测试在临时目录创建合成图像，覆盖文件名大小写、无标签测试数据、年龄统计、指标边界、图像/年龄模型，以及训练→检查点加载→评估→推理→提交→可视化。测试使用 CPU、极小模型和一轮训练，不下载预训练权重。

`.gitignore` 排除原始数据、派生数据、实验结果、权重、环境、缓存和本机凭据；保留源码、配置、测试、说明和 `data/contracts/`。生成的清单默认不提交。已被 Git 跟踪的数据不受新增忽略规则影响。正式实验应使用独立版本目录保存结果。

## B0-dev 服务器运行

实现依据为研究目录中的 `03_Method_and_Experiment/B0_ResNet50_512_基线实验实现方案_v1.md`。本次交付的是 seed=42 的实现文件；本地没有生成比赛内部划分、启动比赛训练、访问官方 valid 选模或预测比赛 test。真实图像检查与目标服务器显存/吞吐尚待以下预检，不能把合成测试解释为资源验收或性能结果。

配置 `configs/b0_resnet50_512_dev_v1.yaml` 固定：随机初始化 torchvision ResNet50，全网络训练，2048→1 单头，dropout=0，图像单输入；普通 BCE、pos_weight=1、自然比例随机遍历、不丢尾批。训练 physical batch=16、累积=2，评估 batch=16；AdamW 的峰值 lr=1e-3、weight_decay=1e-4、betas=(0.9,0.999)、eps=1e-8。最多80轮，最短30轮允许早停，patience=15，min_delta=1e-4。配置中的关键字段会被校验，拼错或未实现的 B0 字段会报错。

保持当前项目目录复制到服务器，原始图像仍在 `data/raw/glaucoma_first`；相对路径无需改写为 Windows 路径。安装与服务器驱动匹配的 PyTorch 后，在工程根目录执行：

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/preprocess/audit_dataset.py --config configs/b0_resnet50_512_dev_v1.yaml --decode-images --output data/manifests/b0_train_audit_v1.json
python scripts/preprocess/build_development_manifest.py --config configs/b0_resnet50_512_dev_v1.yaml
python scripts/train/preflight_b0.py --config configs/b0_resnet50_512_dev_v1.yaml
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

检查 `experiments/b0_resnet50_512_preflight_s42_v1/figures/` 后，再冻结资源配置。对照图只是候选，需确认覆盖正常曝光、极暗、强反光、宽黑边、偏心和非方形案例；不足时先补充有界人工检查，不据少量例子宣称 FOV 已全面通过。原方案最多24张人工候选。若显存不足，预检失败并保留进度，不切CPU、不自动减batch、不更换损失；在正式开始前显式修订并记录物理batch、累积、eval batch与workers。若FOV规则不可靠，应另立“原图等比填充”变体，不能静默修改 `fov_letterbox_v1`。

预检数值和图像检查通过后，由你在服务器启动正式训练：

```bash
python scripts/train/run_binary_classification.py --config configs/b0_resnet50_512_dev_v1.yaml
```

预检与正式目录分别是 `b0_resnet50_512_preflight_s42_v1` 和 `b0_resnet50_512_dev_s42_v1`。已有目录拒绝覆盖，失败运行保留原产物，新版本需要新名称；当前无 `--resume`，不承诺中断后无缝恢复。请保留训练开始时记录的源码版本与环境，特别是 Pillow、torchvision、scikit-learn 版本；不同 scikit-learn 版本导致来源记录变化时不会静默接受旧划分，应保留原环境或另定新清单版本。

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
