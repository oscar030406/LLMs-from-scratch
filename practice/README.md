# Practice：从零构建大模型的全链路复现

> **核心实现文件：[`structure.py`](structure.py)。**

这个目录是我学习《Build a Large Language Model (From Scratch)》和配套 `LLMs-from-scratch` 项目后的总结性实践。目标是把书中从数据处理、模型构建、预训练、微调到评估的主要路径整合到一个可读、可运行的 Python 文件中。

## 覆盖内容

`structure.py` 覆盖以下模块：

- 第 2 章：tokenization、滑动窗口数据集、DataLoader
- 第 3 章：causal multi-head self-attention
- 第 4 章：GPT 模型结构与文本生成
- 第 5 章：loss 计算、训练循环、采样生成、checkpoint、GPT-2 权重加载
- 第 6 章：基于 GPT 的 SMS spam 文本分类微调
- 第 7 章：instruction finetuning 与响应生成
- Appendix D：learning-rate warmup、cosine decay、gradient clipping
- Appendix E：LoRA 参数高效微调

## 运行方式

在仓库根目录运行：

```powershell
python practice\structure.py --help
```

常用命令：

```powershell
python practice\structure.py --skip-ollama-eval
python practice\structure.py --eval-model llama3.1:8b
python practice\structure.py --no-use-lora
```

## 默认目录

```text
practice/gpt2/       # GPT-2 权重目录，不提交 Git
practice/spam_data/  # SMS spam 数据目录，不提交 Git
practice/outputs/    # 生成结果和微调 checkpoint，不提交 Git
```

## 说明

这个文件不是原章节 notebook 的替代品，而是一个学习后的整合版本。它的价值在于把核心流程连接起来，展示从零构建 GPT 架构到完成分类微调、指令微调和本地模型评估的完整路径。

大模型权重、训练产物、数据集和生成结果已经通过 `.gitignore` 排除，GitHub 上只保留源码和说明文档。
