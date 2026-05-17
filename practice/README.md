# Practice Summary: Building an LLM From Scratch

This directory contains my end-to-end practice implementation after studying
*Build a Large Language Model (From Scratch)* and the accompanying
`LLMs-from-scratch` project.

The goal of `structure.py` is to consolidate the main technical path of the
book into one readable Python file: from tokenization and data loading, through
GPT-style model construction, pretraining utilities, pretrained GPT-2 weight
loading, classification finetuning, instruction finetuning, LoRA-based
parameter-efficient finetuning, and local response evaluation.

## What Is Covered

- Chapter 2: tokenization, sliding-window datasets, and dataloaders
- Chapter 3: causal multi-head self-attention
- Chapter 4: GPT model components and text generation
- Chapter 5: loss calculation, training loops, sampling, checkpointing, and
  OpenAI GPT-2 weight loading
- Chapter 6: SMS spam classification finetuning with a GPT model
- Chapter 7: instruction finetuning and response generation
- Appendix D: learning-rate warmup, cosine decay, and gradient clipping
- Appendix E: LoRA adapters for parameter-efficient finetuning

## Main Script

```powershell
python practice\structure.py
```

Useful options:

```powershell
python practice\structure.py --help
python practice\structure.py --skip-ollama-eval
python practice\structure.py --no-use-lora
python practice\structure.py --eval-model llama3.1:8b
```

By default, the script expects GPT-2 weights under `practice/gpt2/`, writes
generated outputs under `practice/outputs/`, and uses LoRA for the chapter 7
instruction-finetuning path.

## Notes

Large model files, datasets, generated responses, and finetuned checkpoints are
ignored by Git. The repository keeps the implementation and documentation, not
local training artifacts.

This practice file is not intended to replace the original chapter notebooks.
It is a consolidated study artifact showing that the full workflow can be
reproduced and connected in a single implementation.
