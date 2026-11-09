# LTN 理论知识梳理

调研日期：2026-11-05。依据：项目根目录《课题简介.docx》。本阶段仅进行文献与源码静态核查，未安装训练环境、运行 MNIST 训练或完成复现。

证据标记：**【课题要求】**来自简介；**【论文】**为原文的转述；**【源码】**为固定版本实现事实；**【推导】**为本文数学分析；**【建议／待确认】**为尚未批准的设计。参考编号见[参考文献清单](D:/LTN-MNIST-Addition/research/参考文献清单.md)。

## 1 从加法监督到可学习的数字谓词

【课题要求】输入是两张 MNIST 图像，训练标签只有它们的和；研究不同模糊合取与存在量词参数对学习的影响。单数字标签可以用于独立评估，不能进入训练损失。

【论文】LTN 使用 Real Logic，把符号解释为实数张量及其运算，再优化逻辑公式的满足度。2016 年初始框架与 2022 年扩展框架的形式细节不同；本课题以现代 LTNtorch 实现为操作依据。[R1 §2–3](https://arxiv.org/pdf/1606.04422#page=3)、[R2 §2–3](https://arxiv.org/pdf/2012.13635#page=3)。

| 概念 | 含义 | 本任务中的对象 |
| --- | --- | --- |
| Grounding，记为 \(\mathcal G\) | 把常量、变量、函数和公式解释为张量或实函数 | 图像成为浮点张量，digit 成为 CNN 的类概率查询 |
| Predicate | 接收对象，返回 \([0,1]\) 真值的函数 | \(D_\theta(x,d)=\mathrm{softmax}(f_\theta(x))_d\) |
| Connective | 组合真值的模糊运算 | \(D(x,a)\land D(y,b)\) |
| Quantifier | 沿变量对应维度聚合真值 | 对数字候选作存在聚合，对样本作全称聚合 |
| Function | 把对象映射为对象；输出不必为真值 | 整数加法用于限定合法数字候选 |

【源码】`ltn.Variable` 保存 `.value` 和 `.free_vars`，框架按变量名进行广播。不同变量通常产生笛卡尔积；`ltn.diag(x,y,z)` 把同一行图像与和标签绑定。`d1,d2` 不应被 diag，否则只能考虑相同数字。[R5 core.py：Variable、process_ltn_objects、diag](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/core.py#L223)。

## 2 模糊真值与概率需要区分

Fuzzy Logic 允许命题部分成立，真值不只是 0 或 1。概率通常衡量事件不确定性，模糊真值衡量成立程度；两者都落在 \([0,1]\)，不代表可任意互换。这里 digit 恰好使用 softmax 概率作为原子真值，但整条公式的聚合输出不自动成为“和的概率分布”。这是解释 LTN 与概率逻辑差异的关键。[R3 §2](https://arxiv.org/pdf/2409.16045#page=2)、[R6 §2–4](https://proceedings.neurips.cc/paper_files/paper/2018/file/dc5d637ed5e62c36ecb73b654b05ba2a-Paper.pdf#page=2)。

## 3 两类 t-norm 与梯度

理想算子如下。t-norm 满足交换、结合、单调性及 \(T(a,1)=a\)。本任务公式只用到合取，无需为六组实验额外引入蕴含。

| 运算 | Product 配置 | Łukasiewicz 配置 |
| --- | --- | --- |
| 合取 | \(T_P(a,b)=ab\) | \(T_L(a,b)=\max(0,a+b-1)\) |
| 对偶析取 | \(S_P(a,b)=a+b-ab\) | \(S_L(a,b)=\min(1,a+b)\) |
| 标准否定 | \(1-a\) | \(1-a\) |

这些算子定义可核对 [R5 fuzzy_ops.py：AndProd 与 AndLuk](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/fuzzy_ops.py#L335)；一般模糊算子背景见 [R4](https://arxiv.org/pdf/2002.06100)。Product 配置常见的 Reichenbach 蕴含 \(1-a+ab\) 不等于 Product 剩余蕴含；正文出现“Product”时应说明具体算子，不能推定所有连接词已经替换。

【推导】对内部点求导：

\[
\frac{\partial T_P}{\partial a}=b,\qquad
\frac{\partial T_L}{\partial a}=\begin{cases}0,&a+b<1,\\1,&a+b>1.\end{cases}
\]

边界 \(a+b=1\) 不可微，由实现的自动微分约定处理。Product 在另一分量很小时梯度较小；Łukasiewicz 在一整片区域梯度为零，而在激活区域导数为 1。因此“更小真值＝更弱梯度”不是一般规律。

【推导】当 10 类概率均为 0.1 时，Product 合取为 0.01，Łukasiewicz 合取为 0。如果所有合法候选均处于 Łukasiewicz 的零区，则增大存在量词 p 也无法穿过零合取导数来更新 CNN。均匀分布只是分析点，不能据此声称官方随机初始化必然训练失败。

## 4 p-mean 与 p-mean-error

对于非空集合的 \(m\) 个真值，理想定义为：

\[
M_p(t)=\left(\frac1m\sum_{j=1}^{m}t_j^p\right)^{1/p},\quad p\ge1;
\qquad
E_q(t)=1-\left(\frac1m\sum_{j=1}^{m}(1-t_j)^q\right)^{1/q}.
\]

存在量词使用 `AggregPMean`，全称量词使用 `AggregPMeanError`。用 \(p_\exists\) 与 \(q_\forall\) 区分，避免调度存在量词时同时改变全称量词。[R5 AggregPMean](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/fuzzy_ops.py#L1292)、[AggregPMeanError](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/fuzzy_ops.py#L1405)。

【推导】\(M_1\) 是均值，\(M_2\) 是均方根，\(p\to\infty\) 时趋向最大值；对固定输入，随 p 增大，存在聚合值不减。全称的 \(q\to\infty\) 则趋向最小真值。两种 p 的方向不能混写。

当 \(M_p>0\) 时：

\[
\frac{\partial M_p}{\partial t_j}=\frac1m t_j^{p-1}M_p^{1-p},\qquad
\frac{\partial M_p/\partial t_j}{\partial M_p/\partial t_k}
=\left(\frac{t_j}{t_k}\right)^{p-1}\quad(t_k>0).
\]

p=1 时候选梯度相同，p 较大时相对偏向当前强候选；这既可能帮助寻找一个解释，也可能强化早期错误解释。这里描述的是对候选真值的局部导数，CNN 梯度还受 softmax、合取与共享参数影响。

【推导示例】两个候选真值为 0.2、0.8 时，\(M_1=0.5\)，\(M_2\approx0.5831\)，\(M_6\approx0.7127\)。改变 p 本身就能提高 Sat，不能把调度切换处的 Sat 跳升当作模型学习进步。

## 5 守卫量词的分母决定目标尺度

令 \(\mathcal D=\{0,\ldots,9\}\)，\(S_z=\{(a,b)\in\mathcal D^2:a+b=z\}\)，\(m_z=|S_z|\)。

\[
m_z=\begin{cases}z+1,&0\le z\le9,\\19-z,&10\le z\le18.\end{cases}
\]

【源码】官方通过 `cond_vars`、`cond_fn` 建立布尔 mask，仅对合法位置取平均；分母是合法位置数，绝不是固定的 100。[R5 Quantifier](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/core.py#L1497)、[AggregPMean mask 分支](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/fuzzy_ops.py#L1389)。guard 对离散候选做过滤，不需要也不会对整数等式求导；过滤后真值计算仍保留 CNN 梯度。

## 6 从满足度到训练损失

【推导，对应官方单公理结构】忽略数值稳定投影时，批内第 i 个样本的满足度为

\[
s_i=M_{p_\exists}\big(\{T(D_\theta(x_i,a),D_\theta(y_i,b)):(a,b)\in S_{z_i}\}\big).
\]

\[
\mathrm{Sat}=1-\left[\frac1B\sum_i(1-s_i)^{q_\forall}\right]^{1/q_\forall},\qquad
\mathcal L=1-\mathrm{Sat}.
\]

更一般的多公理知识库可使用 SatAgg；官方加法循环直接取唯一全称公式的 `.value`，并没有在它外面再次调用 `SatAgg`。[R5 notebook cell 9](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/examples/4-semi-supervised_pattern_recognition.ipynb)。增加额外规则、正则或监督损失都属于另一个设计决定。

## 7 梯度如何到达 CNN

计算链是图像 → CNN logits → softmax → digit 真值 → 合取 → 守卫存在聚合 → 全称聚合 → loss → `backward()`。设 \(u_{ia}=D_\theta(x_i,a)\)，则

\[
\nabla_\theta\mathcal L=
\sum_i\frac{\partial\mathcal L}{\partial s_i}
\sum_{(a,b)\in S_{z_i}}
\frac{\partial s_i}{\partial t_{iab}}
\left(\frac{\partial T}{\partial u_{ia}}\nabla_\theta u_{ia}
+\frac{\partial T}{\partial v_{ib}}\nabla_\theta v_{ib}\right).
\]

softmax 使各类 logits 耦合：虽然 guard 只选合法候选真值，不能据此断言“不合法类别的 logits 梯度必定为零”。训练图内不能对 CNN 输出作 argmax、`.detach()`、`.item()` 或 NumPy 转换。argmax 只用于评估。[R5 Predicate.forward](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/core.py#L583)。

## 8 理想数学与 stable 实现的差别

【源码】\(\epsilon=10^{-4}\)，\(\pi_0(t)=(1-\epsilon)t+\epsilon\)，\(\pi_1(t)=(1-\epsilon)t\)。默认 `AndProd(stable=True)` 先投影两输入；`AggregPMean(stable=True)` 在幂运算前作 \(\pi_0\)；`AggregPMeanError(stable=True)` 先作 \(\pi_1\)。[R5 fuzzy_ops.py 21–61](https://github.com/logictensornetworks/LTNtorch/blob/d1bd98169cc2121f8cdd25ff99901e4589923c95/ltn/fuzzy_ops.py#L21)、[官方 stable 文档](https://logictensornetworks.github.io/LTNtorch/stableconf.html)。

因此官方默认数值不严格等于前述理想式，某些端点不再恰为 0 或 1。`AndLuk()` 没有 stable 参数；不能编造 `AndLuk(stable=True)`。存在量词把零真值投影成 epsilon，也无法消除上游合取的零导数。实验应显式记录所有 stable 配置，避免把数值保护与模糊语义变化混为一谈。
