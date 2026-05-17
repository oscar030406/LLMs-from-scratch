# LLMs-from-scratch 学习复现总结

> **本分支的核心成果位于 [`practice/structure.py`](practice/structure.py)，说明文档位于 [`practice/README.md`](practice/README.md)。**

这个分支是在学习并实践《Build a Large Language Model (From Scratch)》及其配套项目 `LLMs-from-scratch` 后整理出的个人复现成果。原仓库的章节代码仍保留在各章节目录中，方便对照；本分支新增的重点是 `practice/` 目录。

## 本分支做了什么

我将书中的主线内容整合进一个可运行的总结文件：

```text
practice/structure.py
```

它不是简单复制章节 notebook，而是把从零构建大模型的主要链路串联起来，包括：

- 文本 tokenization 与滑动窗口数据集
- DataLoader 构造
- causal multi-head self-attention
- GPT 模型结构
- 文本生成
- loss 计算与训练循环
- GPT-2 预训练权重加载
- 第 6 章文本分类微调
- 第 7 章指令微调
- Appendix D 的 warmup、cosine decay、gradient clipping
- Appendix E 的 LoRA 参数高效微调
- 使用本地 Ollama 模型对指令微调结果进行评分

## 目录说明

```text
practice/
├── structure.py   # 全链路复现与总结代码
└── README.md      # practice 目录说明
```

训练过程中产生的模型权重、GPT-2 下载文件、短信数据集、生成结果等不提交到 GitHub，已经通过 `.gitignore` 忽略。

## 运行方式

在仓库根目录执行：

```powershell
python practice\structure.py --help
```

常用命令：

```powershell
python practice\structure.py --skip-ollama-eval
python practice\structure.py --eval-model llama3.1:8b
python practice\structure.py --no-use-lora
```

默认情况下：

- GPT-2 权重目录：`practice/gpt2/`
- SMS spam 数据目录：`practice/spam_data/`
- 输出目录：`practice/outputs/`
- 第 7 章指令微调默认启用 LoRA
- Ollama 评估默认使用 `llama3.1:8b`

## 学习结果

通过这个实践文件，我完成了从基础组件到下游微调的完整复现：

1. 从文本数据开始，构建 token 级训练样本。
2. 手写注意力机制、Transformer block 和 GPT 模型。
3. 加载 GPT-2 预训练权重验证模型结构兼容性。
4. 使用 GPT 做分类微调，复现垃圾短信识别任务。
5. 使用 GPT-2 medium 做 instruction finetuning。
6. 引入 LoRA，降低指令微调阶段的可训练参数规模。
7. 使用本地 LLM 对生成结果进行自动评分。

这个分支的目的不是替代原项目，而是保留一个清晰的个人学习成果入口：`practice/structure.py`。

## 原项目说明

本仓库 fork/基于 Sebastian Raschka 的 `LLMs-from-scratch` 项目。原项目包含完整章节代码、notebook、附录和练习内容。若需要查看原始教材配套代码，请参考各章节目录或原仓库：

```text
https://github.com/rasbt/LLMs-from-scratch
```
