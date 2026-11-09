# LTN-MNIST-Addition

毕业设计「基于 Logic Tensor Networks 的 MNIST 加法软约束实验」。训练时用同一个 CNN 处理两张 MNIST 图像，只提供两图之和。训练入口支持 LTN 满足度损失和用于流程验证的精确概率和负对数似然。

## 项目目录

```text
LTN-MNIST-Addition/
├── 课题简介.docx
├── research/          调研材料、实验协议草案和待确认问题
├── data/              MNIST 下载和解压目录
├── src/
│   ├── datasets/      官方 MNIST 加载、图像划分、配对数据集
│   ├── models/        共享 CNN、19 类和概率计算
│   ├── logic/         LTN 数字谓词、加法 guard 和满足度
│   ├── training/      仅使用和标签的训练入口
│   ├── evaluation/    指标、独立测试和学习曲线
│   └── utils/         配置、随机种子、检查点和环境验证
├── configs/          概率和验证配置、旧版 LTN 草案与六组对比配置
├── experiments/      按运行 ID 保存配置、配对、日志和测试指标
├── checkpoints/      按运行 ID 保存初始、续训和末轮检查点
├── figures/          学习曲线 PNG
├── tests/            数据、指标、LTN 公式和调度单元测试
├── requirements-win-cu126.txt
└── README.md
```

## 环境和安装

目标环境为 Windows 11 专业版 64 位、Python 3.12.8 和 NVIDIA GeForce RTX 3050 Laptop GPU（4 GiB）。本机驱动为 566.07，`nvidia-smi` 显示 CUDA 12.7，`nvcc` 对应 CUDA Toolkit 12.6.85。系统 Python 使用 PyTorch 2.7.1+cpu 与 torchvision 0.22.1+cpu。项目 `.venv` 使用官方 PyTorch 2.7.1+cu126 和 torchvision 0.22.1+cu126，安装包提供运行所需的 CUDA 库。LTNtorch 的发布包版本为 1.0.2，导入名为 `ltn`。完整依赖版本见 [requirements-win-cu126.txt](requirements-win-cu126.txt)。[PyTorch 官方安装版本](https://pytorch.org/get-started/previous-versions/)、[LTNtorch 发布页](https://pypi.org/project/LTNtorch/1.0.2/)

在 PowerShell 中从项目根目录执行。把项目放在 D 盘时，以下设置会把虚拟环境、下载缓存和安装临时文件也放到 D 盘。当前机器已创建 `.venv`，可直接运行验证命令。

```powershell
Set-Location D:\LTN-MNIST-Addition
$downloadRoot = Join-Path (Get-Location) '.cache'
$env:PIP_CACHE_DIR = Join-Path $downloadRoot 'pip'
$env:TEMP = Join-Path $downloadRoot 'tmp'
$env:TMP = $env:TEMP
$env:MPLCONFIGDIR = Join-Path $downloadRoot 'matplotlib'
$wheelCache = Join-Path $downloadRoot 'wheels'
New-Item -ItemType Directory -Force $env:PIP_CACHE_DIR, $env:TEMP, $env:MPLCONFIGDIR, $wheelCache | Out-Null

python -m venv .venv
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install --find-links $wheelCache -r requirements-win-cu126.txt
& $python -m pip check
& $python -m src.utils.check_environment --require-cuda
& $python -m unittest discover -s tests -v
```

环境验证会在 CPU 和可用的 GPU 上测试 torchvision 原生算子，并执行 LTNtorch 的谓词、合取、量化和反向传播。只使用 CPU 时可省略 `--require-cuda`，并在实验配置中指定 `device = "cpu"`。CUDA 版 PyTorch 的 Windows 安装包约 2.7 GB。

## 运行一次小规模验证

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m src.training.train --config configs/smoke.toml

# 把下面的值替换为训练命令打印的运行 ID。
$run = '<run-id>'
& $python -m src.evaluation.evaluate --run-id $run
& $python -m src.evaluation.plot --metrics "experiments/$run/metrics.csv" --output "figures/$run/learning_curves.png"
```

首次运行会将 MNIST 缓存到 `data/MNIST/`。已备好数据时可在 TOML 中设置 `download = false`。默认配置用 512 对训练样本、128 对验证样本跑两轮。删除 `train_pairs` 和 `validation_pairs` 两行后，默认划分对应 27,000 对训练样本和 3,000 对验证样本。独立测试始终使用官方测试集中的 5,000 对图像和 10,000 张单图。

## 六组 LTN 对比实验

| 配置 | 合取 | 存在量词 p | 旧版草案 |
| --- | --- | --- | --- |
| `A1` | Product | 1 | `P1` |
| `A2` | Product | 2 | `P2` |
| `A3` | Product | 递增调度 | `PS` |
| `B1` | Łukasiewicz | 1 | `L1` |
| `B2` | Łukasiewicz | 2 | `L2` |
| `B3` | Łukasiewicz | 递增调度 | `LS` |

六组配置位于 `configs/comparison/`。在 `shared.toml` 中统一设置学习率、batch size、轮数、默认训练种子、数据划分种子、全称量词参数和稳定模式。A3、B3 共同继承 `exists_schedule`。每条 `[起始轮次, p]` 从指定轮起生效，直到下一条，首条从第 1 轮开始。当前保留原调研草案的数值供确认，正式训练前按确认结果编辑此处。`--epochs` 只修改轮数，切换点按配置的绝对轮次执行。

模型先输出两路数字概率。LTNtorch 对图像对和和标签进行对角量化，数字谓词在 0–9 域内取值。存在量词仅聚合满足 `a+b=z` 的组合，外层全称量词聚合同一批中的配对，损失为 `1 − Sat`。每批 CNN 只处理一次合并后的图像，LTN 广播发生在数字概率上。[LTNtorch 官方示例](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/examples/4-semi-supervised_pattern_recognition.ipynb)

### 独立运行与续训

确认实验协议后，从项目根目录执行。将 `A1` 换成任一组号即可独立运行。多种子比较时，每组都使用同一份预先确定的种子列表。`--seed` 只覆盖初始化与训练批次顺序的种子，所有运行沿用同一个 `data_seed`。

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m src.training.train --config configs/comparison/A1.toml --seed 0
```

可在首次运行时加 `--stop-after 1`，在第 1 轮落盘后暂停，配置中的总轮数继续作为训练预算。程序中断后也可按打印的运行 ID 恢复。

```powershell
$run = '<run-id>'
& $python -m src.training.train --resume $run
```

续训使用保存的生效配置、模型、优化器和随机状态，从最近已保存的轮次之后继续。程序保留原运行 ID、配对表和已经提交的逐轮日志。中断发生在一轮之内时，该轮从头重跑。旧运行若只有 `initial.pt` 或 `final.pt`，仍可按原流程查看和评估。续训需要由新版入口生成的 `latest.pt`。

### 数据与监督

- 先用 `data_seed` 划分官方训练图像，再在每个子集里将前、后半段逐项配对。训练和验证图像索引互斥，官方测试集独立配对。
- 训练配对数据集只提供 `(left, right, sum)`。每轮验证额外读取验证配对中的单数字标签计算识别准确率。每张已配对的验证图像计一次，独立测试使用官方测试集。
- 输入为 `[B, 1, 28, 28]` 的浮点图像，像素除以 255。CNN 沿用调研中记录的 6/16 通道架构，输出 10 类 logits。两张图合并成一批通过同一个 CNN，共用权重和 BatchNorm 统计量。[CNN 结构来源](research/LTNtorch源码分析.md)
- `seed` 控制 Python、NumPy、PyTorch 和训练批次顺序。训练和验证使用独立的随机生成器。程序启用 PyTorch 确定性算法，复现实验请固定软硬件环境。

### 指标与产物

| 指标 | 计算口径 |
| --- | --- |
| HardSum | 两张图各取预测数字，再相加，与和标签比较 |
| ProbSum | 枚举每种合法数字组合，将联合概率按和聚合，再取最可能的和 |
| DigitAcc | 对单图取预测数字，与真实数字比较。逐轮曲线使用验证集，独立测试命令使用官方测试集 |
| 误差抵消率 | 两张图的数字预测有误、但相加结果正确的对数，除以全部测试对数 |
| 合取激活率 | 合取真值大于零的合法数字组合数，除以全部合法组合数 |

每轮 `metrics.csv` 记录训练和验证损失、验证 HardSum/ProbSum/DigitAcc、累计步数、耗时、更新前的梯度范数均值和零梯度批次比例。LTN 运行还记录当轮 p 值，以及分别按样本数加权的训练和验证批次满足度均值。损失和满足度按本组当轮语义计算。跨组准确率使用相同解码口径。CUDA 运行记录 PyTorch 分配和保留的显存峰值。`train_seconds` 是当轮训练耗时，`total_train_seconds` 为累计训练耗时，`elapsed_seconds` 为累计训练和验证耗时。以上时间均排除暂停和检查点写入。

合取激活率按整轮候选数累计。在 Łukasiewicz 下，`u[a] + v[b] > 1` 的合法组合计为激活，存在量词的稳定处理发生在计数之后。日志同时记录激活数、合法数及比例，学习曲线将激活率和零梯度批次比例放在同一面板。

每次运行保存生效配置、依赖版本、设备、初始状态的 SHA-256，以及训练和验证配对表与各自的 SHA-256。检查点包括初始权重、最近已保存轮次的完整状态和固定末轮权重。独立测试读取末轮权重，对每张官方测试图像推理一次，用固定配对计算和准确率和误差抵消率。测试样本数、三项准确率、抵消对数及比例写入 `test_metrics.json`。

### 生成对比图和汇总表

将六组所有入选种子的已完成运行 ID 放进 `$runIds`，指定一个新输出目录。

```powershell
$runIds = @('<A1-run-id>', '<A2-run-id>', '<A3-run-id>', '<B1-run-id>', '<B2-run-id>', '<B3-run-id>')
& $python -m src.evaluation.compare --run-id $runIds --output figures/comparison
```

汇总会核对六组种子集合、控制变量、数据配对、运行环境、同种子的初始模型和 A3/B3 的调度。输出四张曲线图（验证 HardSum、训练损失、验证满足度、验证 DigitAcc）、逐轮汇总 `epoch_summary.csv`、各组固定末轮汇总 `summary.csv`、各运行末轮明细 `final_runs.csv` 和来源清单 `manifest.json`。均值和样本标准差按种子计算，图中阴影为均值 ±1 样本标准差。单种子时标准差留空。源运行日志保留在各自目录中。

### 收敛记录

在试跑配置中加入以下表格可测试收敛记录。示例阈值仅用于验证记录流程。

```toml
[convergence]
hard_sum_threshold = 0.90
consecutive_epochs = 2
```

验证 HardSum 连续达到阈值时，`convergence.json` 记录首次达标区间的起始轮次、确认轮次、累计更新步数和墙钟时间。训练继续到配置指定的末轮。固定预算内未确认写为 `not_reached`，未设置该表写为 `not_configured`。

## 当前边界和后续安排

正式多种子比较前，按导师确认的协议固定 p 调度、验证划分、收敛阈值、退化判据与对照方法。合取激活率、梯度范数和零梯度比例可用于诊断优化停滞。系统性数字偏移诊断和自动退化判定仍待实现。[待导师确认的问题](research/待导师确认的问题.md)
