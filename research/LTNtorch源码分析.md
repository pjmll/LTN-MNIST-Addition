# LTNtorch 源码分析

调研日期：2026-11-05。以下为官方源码静态核查，不代表代码已运行。理论来源、源码事实与改进建议分开说明。

## 1 版本和可定位的阅读入口

核查仓库为 `logictensornetworks/LTNtorch`，固定提交为 `d1bd98169cc2121f8cdd25ff99901e4589923c95`。下载并检查了 README、setup.py、ltn/core.py、ltn/fuzzy_ops.py 和 MNIST notebook；未执行其中训练单元。下面所有源码链接均绑定该提交。

| 文件或对象 | 定位 | 核查内容 |
| --- | --- | --- |
| examples/4-semi-supervised_pattern_recognition.ipynb | [官方 notebook](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/examples/4-semi-supervised_pattern_recognition.ipynb) | 完整单数字与多数字加法流程 |
| ltn/core.py | [Predicate.forward](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/core.py#L583) | 广播、模型调用、真值检查 |
| ltn/core.py | [diag](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/core.py#L903)；[Quantifier.__call__](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/core.py#L1428) | 配对量化、guard、参数透传 |
| ltn/fuzzy_ops.py | [AndProd](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/fuzzy_ops.py#L335)；[AndLuk](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/fuzzy_ops.py#L423) | 两种合取 |
| ltn/fuzzy_ops.py | [AggregPMean](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/fuzzy_ops.py#L1292)；[AggregPMeanError](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/fuzzy_ops.py#L1405) | 存在与全称聚合 |
| ltn/fuzzy_ops.py | [SatAgg](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/fuzzy_ops.py#L1517) | 多条闭公式的汇总 |

notebook 单元编号采用 JSON `cells` 数组的 **0 起始索引**，不是易变化的执行编号 `In[n]`。cell 1 数据、cell 5 网络与算子、cell 7 DataLoader、cell 9 单数字训练、cell 11 多数字训练。

【源码】setup.py 声明版本 1.0.2，安装包名 LTNtorch、导入名 `ltn`，库依赖列出 numpy 和 torch；示例另导入 torchvision、pandas、matplotlib。[setup.py](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/setup.py)。在线文档页眉仍显示 0.9；本调研遇到差异时优先采用固定提交实现，不声称当前软件组合已验证兼容。

## 2 数据进入训练的完整过程

【源码，cell 1/7/9】

1. 读取 MNIST 的 `.data`、`.targets`，图像除以 255 并增加通道维度。这里直接读 `.data`，实际缩放来自手动操作，不依赖传入的 ToTensor 被执行。
2. 单数字训练集前 30,000 张作左操作数，后 30,000 张作右操作数，构成 30,000 对；测试集同理构成 5,000 对。目标只保留两标签相加。
3. 堆叠后输入形状为 `[N,2,1,28,28]`，目标为 `[N]`。
4. 自定义 DataLoader 使用 NumPy shuffle，batch_size=32；每轮 batch 中重新创建图像变量和和标签变量。
5. 对十个数字值各建一个变量域，计算两个 digit 谓词，合取后作有 guard 的存在量化，再对样本作全称量化。
6. loss 是 `1. - sat_agg`；反向传播后更新共享 CNN。[原始完整 notebook](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/examples/4-semi-supervised_pattern_recognition.ipynb)。

## 3 CNN 的真实结构与张量形状

【源码，cell 5】`MNISTConv` 并不是直接输出 10 类，它先输出 100 维特征。真正的 logits 模型为 `SingleDigitClassifier`。

| 顺序 | 操作 | 单图批输入对应输出 |
| --- | --- | --- |
| 输入 | 灰度图 | `[B,1,28,28]` |
| 1 | Conv2d(1,6,5) → ReLU → MaxPool2d(2,2) | `[B,6,12,12]` |
| 2 | Conv2d(6,16,5) → ReLU → MaxPool2d(2,2) | `[B,16,4,4]` |
| 3 | flatten → Linear(256,100) → BatchNorm1d → Tanh | `[B,100]` |
| 4 | Linear(100,84) → BatchNorm1d → Tanh | `[B,84]` |
| 5 | Linear(84,10) | `[B,10]` logits |

卷积核权重采用 kaiming_uniform_，线性权重 xavier_uniform_，相应偏置 normal_。不能把它描述成“完全照搬 DeepProbLog 论文的 LeNet”；两者结构和激活细节不同。[cell 5](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/examples/4-semi-supervised_pattern_recognition.ipynb)。

## 4 digit 谓词确切做了什么

【源码，cell 5】`LogitsToPredicate` 持有 logits 模型，forward 的核心是：

```python
logits = self.logits_model(x)
probs = self.softmax(logits)   # Softmax(dim=1)
out = torch.gather(probs, 1, d)
return out
```

这段是官方 MIT 许可源码的短摘录。构造关系为：

```python
cnn_s_d = SingleDigitClassifier()
Digit_s_d = ltn.Predicate(LogitsToPredicate(cnn_s_d)).to(ltn.device)
```

它们共享同一个 CNN 对象，而不是两套独立权重。`Variable` 会把一维数字域转成 `[10,1]`，因此 `gather` 使用列索引；经过 LTN 广播，对 B 张图和 10 个数字的谓词结果形状是 `[B,10]`。[Variable](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/core.py#L223)、[Predicate.forward](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/core.py#L613)。

## 5 连接词、量词与调度的真实 API

以下是依照已核查 API 改写的配置示意，未运行；用于说明将来实现方式，不是完整模型。

```python
import ltn
And_product = ltn.Connective(ltn.fuzzy_ops.AndProd(stable=True))
And_lukasiewicz = ltn.Connective(ltn.fuzzy_ops.AndLuk())
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=2, stable=True), quantifier="e")
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2, stable=True), quantifier="f")
```

`AndLuk` 没有 stable 参数。`Quantifier.__call__` 接收 `**kwargs` 并传递给聚合器，因此可以用 `Exists(..., p=current_p)` 覆盖构造时默认的 2；不能只看构造处就断言训练始终 p=2。[Quantifier](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/core.py#L1503)、[AggregPMean.__call__](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/fuzzy_ops.py#L1389)。

官方单数字 cell 9 的调度如下，epoch 从 0 计数：

| epoch | 读者习惯的训练轮次 | 存在量词 p |
| --- | --- | --- |
| 0–3 | 1–4 | 1 |
| 4–7 | 5–8 | 2 |
| 8–11 | 9–12 | 4 |
| 12–19 | 13–20 | 6 |

这是分段常数调度，不是线性逐轮增长。课题简介尚未确认要沿用这些切点，也未说明延长训练预算后如何处理。

## 6 加法公式的实际写法与 backward

下面摘录官方 cell 9 的核心表达式。前提是 `images_x`、`images_y`、`labels_z` 已由当前批建立，Digit_s_d 等对象已构造，p 为当前调度值。

```python
sat_agg = Forall(
    ltn.diag(images_x, images_y, labels_z),
    Exists(
        [d_1, d_2],
        And(Digit_s_d(images_x, d_1), Digit_s_d(images_y, d_2)),
        cond_vars=[d_1, d_2, labels_z],
        cond_fn=lambda d1, d2, z: torch.eq(d1.value + d2.value, z.value),
        p=p
    )).value
loss = 1. - sat_agg
loss.backward()
optimizer.step()
```

循环前使用 `torch.optim.Adam(Digit_s_d.parameters(), lr=0.001)`，每步先 `optimizer.zero_grad()`。此处 `.value` 仍是连接计算图的 Tensor；它不同于 `.item()`。唯一公理已经得到标量，故虽然 cell 5 定义了 `SatAgg = ltn.fuzzy_ops.SatAgg()`，加法训练中没有实际调用它。[cell 5/9](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/examples/4-semi-supervised_pattern_recognition.ipynb)。

【源码与分析】Python 会先计算外层调用的第一个参数 `ltn.diag(...)`，然后才计算第二参数中的谓词；若把公式提前放进变量、但之后才调用 diag，可能先构造了错误的笛卡尔积。建议显式先做 `ltn.diag(images_x,images_y,labels_z)`，再构造内部公式；每批重建变量，并理解量词结束会调用 `undiag`。[diag](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/core.py#L994)、[Quantifier 尾部](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/core.py#L1524)。

## 7 mask 和 stable 实现不能省略的细节

【源码】`AggregPMean` 对合法元素的幂求和，除以 mask 的 True 数量，再开 p 次方根。`Quantifier` 对空 guard 导致的 NaN 作替换：存在为 0，全称为 1。但不应把这个处理当作保证反向传播总安全的机制；本任务合法和在 0–18，正常情况下每个 guard 都应非空。[聚合器](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/fuzzy_ops.py#L1394)、[空域处理](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/core.py#L1505)。

默认 stable 的 epsilon 为 1e-4，Product 合取先投影输入，存在与全称聚合也各有投影。正式结果必须报告 stable 配置，不宜把数值直接拿去与理想算式对齐后宣称有 bug。[实现](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/fuzzy_ops.py#L21)。

## 8 评估循环的局限与正式实验建议

【源码事实】cell 9 用两个单数字 logits 的 argmax 相加计算和准确率；没有单数字真值 accuracy。cell 9/11 没有显式 `train()`、`eval()`、`torch.no_grad()`，指标按每批准确率简单平均，并用训练过程中的模型即时累计训练指标。

【分析】网络含 BatchNorm，测试时若保持训练模式，会使用测试批统计并更新 running statistics。`no_grad` 不能替代 `eval`。这是静态阅读发现的风险，不是已经测得的性能偏差。

【建议】后续复现代码应明确训练与评估模式、评估关闭梯度、按正确数/总数汇总准确率，单独记录官方流程与规范评估的差别；使用验证集选模型，测试集不用于选择 p 或训练轮数。指标累计用 `.detach()`，避免日志持有训练图。最后一个训练 batch 若仅 1 个样本，需事先处理 BatchNorm 约束并报告实际样本数。

## 9 多数字分支与本课题范围

【源码】cell 11 用四张图、四个数字变量，guard 为 `10*d1+d2+10*d3+d4==z`，合取按二元算子嵌套。候选空间由 100 扩大到 10,000。该分支有助于理解计算成本，但简介描述“两图之和”，六组主实验是否扩到多数字需导师决定，不应自动增加工作量。
