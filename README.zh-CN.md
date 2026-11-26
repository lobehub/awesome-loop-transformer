# awesome-Loop-Transformer 中文导读

[English](README.md) · [论文索引](docs/research-and-code.md) · [模型与权重](docs/models-and-weights.md) · [技术报告与工具](docs/reports-and-tools.md) · [语言模型里的其他循环](docs/language-model-loops.md)

## 什么是 Loop Transformer？

**Loop Transformer 指的是在深度方向复用同一个 Transformer 层或一组层的模型。** 假设循环部分只有 A、B 两层，运行三轮，实际路径就是 **A → B → A → B → A → B**。模型执行了六次层计算，循环部分只存两套层权重。第二次走到 A 时，它读到的是 B 刚刚更新过的隐藏状态。用在语言模型里，输入前缀可以保持不动，模型先在内部走完这几轮，再预测下一个 token。[Universal Transformers](https://arxiv.org/abs/1807.03819) · [Huginn 模型说明](https://huggingface.co/tomg-group-umd/huginn-0125)

读论文时，先问**轮数由谁决定**。ALBERT 的层共享用在预先设定的深度上，研究的是能省多少参数。Huginn 允许使用者拿同一份权重改变轮数，可以直接测多算几步对答案有什么影响。Mixture-of-Recursions 则让路由器决定哪些 token 值得再算一轮。同样叫循环，三者该做的对照实验各不相同。[ALBERT](https://arxiv.org/abs/1909.11942) · [Huginn](https://arxiv.org/abs/2502.05171) · [Mixture-of-Recursions](https://arxiv.org/abs/2507.10524)

## 这条研究线是怎么来的？

2016 年，Graves 在 RNN 上提出 Adaptive Computation Time，让模型学习从接收输入到给出输出之间需要更新几次。2017 年的 Transformer 提供了另一块基础：同一层里，各个位置可以通过自注意力交换信息。2018 年 7 月的 *Universal Transformers* 把共享的注意力与状态更新沿深度反复执行，并用 ACT 让不同位置可以在不同轮次停下来。[ACT](https://arxiv.org/abs/1603.08983) · [原始 Transformer](https://arxiv.org/abs/1706.03762) · [Universal Transformer 原文及图 2](https://arxiv.org/html/1807.03819v3)

同一个月，Dabre 和 Fujita 从机器翻译提出了一个更容易动手验证的问题：把六个独立层换成同一层跑六次，翻译质量会怎样？在他们测试的设置里，两者质量相近。到 2019 年，ALBERT 把跨层共享用进了预训练编码器；到 2025 年，Huginn 又把轮数交给推理时的使用者。沿这条线读，可以看到研究问题从“固定深度下能省多少权重”延伸到了“同一份权重多算一轮值多少”。[循环堆叠机器翻译论文](https://arxiv.org/abs/1807.05353) · [ALBERT](https://arxiv.org/abs/1909.11942) · [Huginn 技术报告](https://arxiv.org/abs/2502.05171)

## 五篇值得按顺序读的论文

| 顺序 | 带着什么问题读 | 从哪里看起 |
| --- | --- | --- |
| 1 · Universal Transformers | 一轮到底更新了什么？ | 看图 2，选一个 token，沿深度跟两轮。横向看它从哪些位置读信息，纵向看更新后的状态去了哪里。[论文与架构图](https://arxiv.org/html/1807.03819v3) |
| 2 · Looped Transformers are Better at Learning Learning Algorithms | 共享更新规则对哪类任务有帮助？ | 从上下文线性回归实验入手。模型看到若干输入输出样本后预测新样本，可以直接比较循环模型和独立层模型的预测误差、参数量。[论文](https://arxiv.org/abs/2311.12424) · [实验代码](https://github.com/Leiay/looped_transformer) |
| 3 · Huginn | 同一个模型多跑几轮，答案会变好吗？ | 先看同一检查点的深度实验。模型卡公开了 `num_steps`，可以在同一组算术题上分别试 8 轮和 32 轮，固定贪心解码，同时记正确率和耗时。[论文](https://arxiv.org/abs/2502.05171) · [权重与用法](https://huggingface.co/tomg-group-umd/huginn-0125) |
| 4 · Latent Chain-of-Thought? | 答案变好了，能从中间状态读出算术步骤吗？ | 作者用两种探针读取 Huginn。重点看同一深度换一种读出方法后，能看到的中间结果如何变化。这给“模型在心算”提供了一个可检验的具体问题。[论文](https://arxiv.org/abs/2507.02199) · [探针代码](https://github.com/wenquanlu/huginn-latent-cot) |
| 5 · How Much Is One Recurrence Worth? | 节省参数的同时，付出了多少计算？ | 论文中的一个四轮、4.1 亿参数模型，在验证损失上追平了 5.8 亿参数的独立层模型，训练计算量却接近 10 亿参数的独立层模型。读图时要分别看参数量、执行深度和训练计算量。[论文](https://arxiv.org/abs/2604.21106) · [实验代码](https://github.com/kschwethelm/looped-lm-scaling) |

## 按兴趣往下读

- 想跟进架构和算法，可以从[算法与长度泛化](docs/research-and-code.md#algorithms-and-length-generalization)、[动态深度与规模实验](docs/research-and-code.md#depth-allocation-and-scaling)或[推理与显存](docs/research-and-code.md#inference-attention-and-memory)进入。
- 想上手跑模型，先在[权重目录](docs/models-and-weights.md)选具体版本，再到[报告与工具](docs/reports-and-tools.md)找训练配方、评测代码或探针。
- 想了解更广义的 loop，可以对照四种“下一轮的输入”：Transformer-XL 传递上一段的记忆，Coconut 传递思考向量，LLaDA 传递尚未填完的文本，Self-Refine 传递草稿和反馈。[语言模型里的其他循环](docs/language-model-loops.md)
