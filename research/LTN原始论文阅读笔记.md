# LTN 原始论文阅读笔记

调研日期：2026-11-05。依据：项目根目录《课题简介.docx》。本阶段仅进行文献与源码静态核查，未安装训练环境、运行 MNIST 训练或完成复现。

证据标记：**【课题要求】**来自简介；**【论文】**为原文的转述；**【源码】**为固定版本实现事实；**【推导】**为本文数学分析；**【建议／待确认】**为尚未批准的设计。参考编号见[参考文献清单](D:/LTN-MNIST-Addition/research/参考文献清单.md)。

## 1 阅读范围与文献关系

本笔记按本课题问题定点阅读，并非声称逐页精读所有论文。核心核查内容为 Real Logic、可满足性优化、MNIST 公式与监督方式、聚合参数和实现来源。2016 年奠基工作、2022 年完整形式化和 2024 年 PyTorch 实现论文不能互相替代引用。

| 文献 | 作者与年份 | 已核查位置 | 对本课题的用途 |
| --- | --- | --- | --- |
| R1 Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge | Luciano Serafini；Artur d’Avila Garcez，2016 | §2 Real Logic、§3 Learning as approximate satisfiability | 理论起源 |
| R2 Logic Tensor Networks | Samy Badreddine；Artur d’Avila Garcez；Luciano Serafini；Michael Spranger，2022 | §2–3、§4.4、附录 B；预印本 v4 | MNIST Addition 直接理论来源 |
| R3 LTNtorch: PyTorch Implementation of Logic Tensor Networks | Tommaso Carraro；Luciano Serafini；Fabio Aiolli，2024 | §2、§3 | PyTorch 实现依据；按 arXiv 预印本引用 |
| R4 Analyzing Differentiable Fuzzy Logic Operators | Emile van Krieken；Erman Acar；Frank van Harmelen | arXiv 2020，修订 2021；摘要与算子分析部分 | 梯度分析背景；不等于本课题实测 |
| R6 DeepProbLog: Neural Probabilistic Logic Programming | Robin Manhaeve；Sebastijan Dumančić；Angelika Kimmig；Thomas Demeester；Luc De Raedt，2018 | §2–5、§6 T1/T2、Fig.3、实现细节 | 概率逻辑路线与加法任务来源 |

## 2 R1：最初的 LTN

【论文】R1 用实数特征解释对象，以实函数解释符号，把学习视为寻找尽量满足逻辑理论的 grounding。§3 使用置信区间与满足误差定义近似可满足性。该文主要演示知识补全，不能作为“LTNtorch 已完成 MNIST 加法”的证据。[全文](https://arxiv.org/pdf/1606.04422)、[元数据与版本](https://arxiv.org/abs/1606.04422)。

【分析】毕业论文可用它说明历史起点；具体 p-mean、diag、守卫量词和现代 API 应转引 R2/R5，避免把后来的接口倒写进 2016 年论文。

## 3 R2：与毕业设计最直接相关

【论文】R2 的 Real Logic 与学习定义见 §2–3。MNIST 任务位于 §4.4，单数字公式 (39)、多数字公式 (40)，计算图 Fig.13。单数字训练只监督和，内部 digit 是潜变量；外层 Diag 保证图像对与标签对应，存在量词 guard 限定合法加法解释。[§4.4，PDF 印刷页 27–31](https://arxiv.org/pdf/2012.13635#page=27)。

【论文】Fig.14 是单数字加法，Fig.15 是多数字加法。正文在多数字实验比较固定 p=1、固定 p=2 和递增调度；单数字实验只报告调度情景。全称聚合使用 p-mean-error，参数为 2；存在 p 逐渐由 1 变为 6。[原文 Learning 段，PDF 第30页](https://arxiv.org/pdf/2012.13635#page=30)。

【分析】因此本课题把固定 p 方案与 Łukasiewicz 加入单数字加法，是在官方任务上的受控扩展，不应声称论文已经报告同样的六组结果。正文“逐渐递增”不能替代具体 epoch 切点；实际分段见源码分析。

原文 Table 2 的单数字测试和准确率：30,000 个训练图像对时，LTN 为 \(98.04\pm0.13\%\)，DeepProbLog 为 \(97.20\pm0.45\%\)。这是原论文汇报的跨方法数值，非当前项目运行结果；表注说明 DeepProbLog 数值来自其引用文献，不能视作同一环境重新训练后的因果比较。[Table 2，PDF 第32页](https://arxiv.org/pdf/2012.13635#page=32)。本笔记不推定 ± 的统计定义，也不把它写成单数字识别准确率。

## 4 R3：LTNtorch 的来源和引用边界

【论文】R3 §2 将符号张量、可微逻辑与优化目标连接起来，§3 用二分类解释实现。其目标是介绍库，并非完整报告本课题六组实验。[全文](https://arxiv.org/pdf/2409.16045)。

【核查】arXiv 标题页记载 2024-09-24 提交；PDF 存在期刊模板的占位信息，不能只凭页眉写成正式 JMLR 发表论文。本调研按 arXiv:2409.16045v1 引用。[元数据](https://arxiv.org/abs/2409.16045)。

## 5 R4：为什么应关注梯度而不只是逻辑值

【论文】该工作分析模糊算子在梯度训练中的适用性，指出部分常见算子存在梯度传递问题。[摘要与全文](https://arxiv.org/abs/2002.06100)。

【本课题分析】建议把“引导强度”拆成合取激活比例、参数梯度范数和候选梯度集中程度。理论文献不能替代本课题对死区出现频率的实测；本文的均匀初始化示例属于数学推导。

## 6 R6：DeepProbLog 原始研究

【论文】R6 引入神经谓词及概率逻辑推断，§6 T1 使用单数字图像对与和标签，T2 展示更长数字的组合泛化。推断先 grounding，再构造逻辑表达式及 SDD，并用算术运算得到查询概率；学习使查询误差的梯度回传网络。[官方 NeurIPS 论文](https://proceedings.neurips.cc/paper_files/paper/2018/file/dc5d637ed5e62c36ecb73b654b05ba2a-Paper.pdf)。

【论文】实现细节描述 6/16 通道卷积及 120/84/10 全连接结构，优化查询概率的交叉熵。Fig.3 图注明确写 F1 score，不能把读图值随意写成 accuracy。该结构也不等于当前 LTNtorch notebook 的 256→100→84→10 结构。[§6 与 Fig.3](https://proceedings.neurips.cc/paper_files/paper/2018/file/dc5d637ed5e62c36ecb73b654b05ba2a-Paper.pdf#page=6)。

## 7 DeepLog 名称核查后的结论

截至本次调研，至少有三种需要区分的名称：

1. DeepProbLog（2018）：神经概率逻辑编程，R2 的 MNIST 任务直接引用它。
2. DeepLog（KU Leuven，2026）：模块化神经符号框架，能够编译不同语义，也提供 DeepProbLog 风格加法示例。
3. DeepLog（Du 等，2017）：系统日志异常检测，主题与此处不同。

【论文】2026 年工作确为真实论文：*DeepLog: A Software Framework for Modular Neurosymbolic AI*，Robin Manhaeve、Stefano Colamonaco、Vincent Derkinderen、Rik Adriaensen、Lucas Van Praet、Luc De Raedt、Giuseppe Marra；IJCAI-26 Demo Track，8462–8465。[会议官方页面](https://ijcai.org/proceedings/2026/980)。已核查其摘要及 §1–4 的框架定位；未核查其运行性能表的实验复现。

【源码】当前 DeepLog `examples/deepproblog.md` 同时展示精确计算与采样后端。[固定提交示例](https://github.com/ML-KULeuven/deeplog/blob/af035231a881ff1d803c3d39ea194043cd096cf3/examples/deepproblog.md#L181)。

【待确认】简介可能在说现代 DeepLog，也可能把 DeepProbLog 简写成 DeepLog。应请导师提供链接，并明确“精确概率聚合”指哪一种配置。不能擅自改名，也不能把 DeepLog 整个框架定性为只支持精确概率语义。

## 8 可以写与暂时不能写的结论

可以写：LTN 将加法知识参与可微目标构建；官方源码存在该任务；六组对比能够检验算子和量词设置的作用。暂时不能写：本项目复现成功；Łukasiewicz 一定更差；调度一定优于固定 p；精确概率推断必然消除退化解。这些都需要本项目实验或更具体的证明。
