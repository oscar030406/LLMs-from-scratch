import os
import json
import math
import zipfile
from pathlib import Path
import tiktoken
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # Causal mask blocks attention to future tokens.
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  # type: ignore
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (self.head_dim**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device, dtype=x.dtype)) * (x + 0.044715 * torch.pow(x, 3))))
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算一个 batch 的 next-token prediction 交叉熵损失。"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """在 DataLoader 上平均若干个 batch 的 loss。用于训练/验证评估。"""
    if len(data_loader) == 0:
        return float("nan")
    total_loss = 0.0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """临时切到 eval 模式评估，再恢复 train 模式。"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """每个 epoch 后生成一段样本文本，直观看训练进展。"""
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    """ch05 的最小预训练循环：forward -> loss -> backward -> AdamW update。"""
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def get_lr_with_warmup_cosine(global_step, total_steps, warmup_steps, initial_lr, peak_lr, min_lr):
    """Appendix D: linear warmup followed by cosine decay."""
    if warmup_steps > 0 and global_step < warmup_steps:
        return initial_lr + global_step * (peak_lr - initial_lr) / warmup_steps

    if total_steps <= warmup_steps:
        return peak_lr

    progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine_decay


def train_model_with_appendix_d(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    warmup_steps,
    initial_lr=3e-5,
    peak_lr=5e-5,
    min_lr=1e-6,
    max_grad_norm=1.0,
):
    """Appendix D training loop: warmup, cosine decay, and gradient clipping."""
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen = 0
    global_step = -1
    total_steps = max(1, len(train_loader) * num_epochs)

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step += 1

            lr = get_lr_with_warmup_cosine(
                global_step=global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                initial_lr=initial_lr,
                peak_lr=peak_lr,
                min_lr=min_lr,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            if global_step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

            optimizer.step()

            tokens_seen += input_batch.numel()

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                track_lrs.append(lr)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, LR {lr:.2e}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen, track_lrs


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """支持 top-k 和 temperature 的生成函数；temperature=0 时退化为贪心解码。"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def assign(left, right):
    """把外部权重转成 Parameter，并检查形状是否能对齐。"""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return nn.Parameter(torch.tensor(right, dtype=left.dtype))


def load_weights_into_gpt(gpt, params):
    """把 OpenAI GPT-2 checkpoint 中的权重映射到本文件实现的 GPTModel。"""
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for block_idx in range(len(params["blocks"])):
        block_params = params["blocks"][block_idx]
        block = gpt.trf_blocks[block_idx]

        q_w, k_w, v_w = np.split(block_params["attn"]["c_attn"]["w"], 3, axis=-1)
        block.att.W_query.weight = assign(block.att.W_query.weight, q_w.T)
        block.att.W_key.weight = assign(block.att.W_key.weight, k_w.T)
        block.att.W_value.weight = assign(block.att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(block_params["attn"]["c_attn"]["b"], 3, axis=-1)
        block.att.W_query.bias = assign(block.att.W_query.bias, q_b)
        block.att.W_key.bias = assign(block.att.W_key.bias, k_b)
        block.att.W_value.bias = assign(block.att.W_value.bias, v_b)

        block.att.out_proj.weight = assign(block.att.out_proj.weight, block_params["attn"]["c_proj"]["w"].T)
        block.att.out_proj.bias = assign(block.att.out_proj.bias, block_params["attn"]["c_proj"]["b"])

        block.ff.layers[0].weight = assign(block.ff.layers[0].weight, block_params["mlp"]["c_fc"]["w"].T)
        block.ff.layers[0].bias = assign(block.ff.layers[0].bias, block_params["mlp"]["c_fc"]["b"])
        block.ff.layers[2].weight = assign(block.ff.layers[2].weight, block_params["mlp"]["c_proj"]["w"].T)
        block.ff.layers[2].bias = assign(block.ff.layers[2].bias, block_params["mlp"]["c_proj"]["b"])

        block.norm1.scale = assign(block.norm1.scale, block_params["ln_1"]["g"])
        block.norm1.shift = assign(block.norm1.shift, block_params["ln_1"]["b"])
        block.norm2.scale = assign(block.norm2.scale, block_params["ln_2"]["g"])
        block.norm2.shift = assign(block.norm2.shift, block_params["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


class LoRALayer(nn.Module):
    """Appendix E: low-rank trainable update for a frozen linear projection."""

    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank

    def forward(self, x):
        return (self.alpha / self.rank) * (x @ self.A @ self.B)


class LinearWithLoRA(nn.Module):
    """Wrap an existing Linear layer as frozen base + trainable LoRA adapter."""

    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False


def replace_linear_with_lora(model, rank=16, alpha=16, module_names=None):
    """Recursively replace Linear layers with LoRA wrappers.

    module_names can restrict replacement to names such as {"W_query", "W_value"}.
    The default mirrors appendix E and wraps every Linear layer in the GPT model.
    """
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            if module_names is None or name in module_names:
                setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank=rank, alpha=alpha, module_names=module_names)


def count_parameters(model, only_trainable=False):
    params = model.parameters()
    if only_trainable:
        params = (param for param in params if param.requires_grad)
    return sum(param.numel() for param in params)


# ==================== ch06: 分类微调 ====================


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    """下载并解压 SMS Spam Collection，得到 ch06 使用的 tsv 数据文件。"""
    zip_path = Path(zip_path)
    extracted_path = Path(extracted_path)
    data_file_path = Path(data_file_path)

    if data_file_path.exists():
        return

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    extracted_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)

    original_file_path = extracted_path / "SMSSpamCollection"
    os.replace(original_file_path, data_file_path)


def create_balanced_dataset(df):
    """让 ham 和 spam 数量相同，避免分类器只学多数类。"""
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    return pd.concat([ham_subset, df[df["Label"] == "spam"]]).reset_index(drop=True)


def random_split(df, train_frac, validation_frac):
    """按 train/validation/test 切分 DataFrame。"""
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    return df[:train_end], df[train_end:validation_end], df[validation_end:]


class SpamDataset(Dataset):
    """把短信文本预编码成等长 token 序列，并返回二分类标签。"""

    def __init__(self, data, tokenizer, max_length=None, pad_token_id=50256):
        if isinstance(data, (str, Path)):
            self.data = pd.read_csv(data)
        else:
            self.data = data.copy()

        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [encoded_text[: self.max_length] for encoded_text in self.encoded_texts]

        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        return max(len(encoded_text) for encoded_text in self.encoded_texts)


def prepare_model_for_classification(model, emb_dim, num_classes=2, train_last_block=True):
    """冻结 GPT 主体，替换输出头，并只开放分类头/末层等少量参数。"""
    for param in model.parameters():
        param.requires_grad = False

    model.out_head = nn.Linear(in_features=emb_dim, out_features=num_classes)

    if train_last_block:
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True

    return model


def calc_classification_accuracy_loader(data_loader, model, device, num_batches=None):
    """用最后一个 token 位置的 logits 计算分类准确率。"""
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break

        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        with torch.no_grad():
            logits = model(input_batch)[:, -1, :]

        predicted_labels = torch.argmax(logits, dim=-1)
        num_examples += predicted_labels.shape[0]
        correct_predictions += (predicted_labels == target_batch).sum().item()

    return correct_predictions / num_examples


def calc_classification_loss_batch(input_batch, target_batch, model, device):
    """分类微调只优化最后一个 token 的类别 logits。"""
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    return F.cross_entropy(logits, target_batch)


def calc_classification_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for batch_idx, (input_batch, target_batch) in enumerate(data_loader):
        if batch_idx >= num_batches:
            break
        loss = calc_classification_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches


def evaluate_classifier(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_classification_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_classification_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_classifier_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_classification_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_classifier(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        train_acc = calc_classification_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_acc = calc_classification_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def classify_text(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """用微调后的分类 GPT 判断文本是 ham 还是 spam。"""
    model.eval()
    input_ids = tokenizer.encode(text)

    supported_context_length = model.pos_emb.weight.shape[0]
    input_ids = input_ids[: min(max_length or supported_context_length, supported_context_length)]

    if max_length is not None:
        input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]

    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"


# ==================== ch07: 指令微调 ====================


def download_and_load_json_file(file_path, url=None):
    """读取指令微调 JSON 数据；本地不存在且提供 url 时才下载。"""
    file_path = Path(file_path)
    if not file_path.exists():
        if url is None:
            raise FileNotFoundError(file_path)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        file_path.write_text(response.text, encoding="utf-8")

    return json.loads(file_path.read_text(encoding="utf-8"))


def split_instruction_data(data, train_frac=0.85, test_frac=0.10):
    """ch07 使用 85% train、10% test、5% validation。"""
    train_end = int(len(data) * train_frac)
    test_end = train_end + int(len(data) * test_frac)
    train_data = data[:train_end]
    test_data = data[train_end:test_end]
    val_data = data[test_end:]
    return train_data, val_data, test_data


def format_instruction_input(entry):
    """把 instruction/input 格式化为 Alpaca 风格 prompt。"""
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


class InstructionDataset(Dataset):
    """预编码 instruction + input + response，供监督微调使用。"""

    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []

        for entry in data:
            prompt = format_instruction_input(entry)
            response = f"\n\n### Response:\n{entry['output']}"
            self.encoded_texts.append(tokenizer.encode(prompt + response))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def instruction_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu",
):
    """为 instruction finetuning 构造 shifted inputs/targets，并忽略多余 padding loss。"""
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)


def extract_instruction_response(generated_text, prompt):
    """从完整生成文本中切出 ### Response 后面的模型回答。"""
    return generated_text[len(prompt) :].replace("### Response:", "").strip()


def generate_instruction_response(
    entry,
    model,
    tokenizer,
    device,
    context_size,
    max_new_tokens=128,
    temperature=0.0,
    top_k=None,
    eos_id=50256,
):
    """对一条 instruction entry 生成回答，并返回去掉 prompt 的 response。"""
    prompt = format_instruction_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
    )
    generated_text = token_ids_to_text(token_ids.cpu(), tokenizer)
    return extract_instruction_response(generated_text, prompt)


def main():
    import argparse
    import contextlib
    import gc
    import io
    import re
    import sys
    import time
    from importlib.util import module_from_spec, spec_from_file_location

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    practice_dir = Path(__file__).resolve().parent
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Chapter summary script for LLMs from scratch.")
    parser.add_argument("--models-dir", type=Path, default=practice_dir / "gpt2")
    parser.add_argument("--spam-data-dir", type=Path, default=practice_dir / "spam_data")
    parser.add_argument("--output-dir", type=Path, default=practice_dir / "outputs")
    parser.add_argument("--ch06-epochs", type=int, default=5)
    parser.add_argument("--ch07-epochs", type=int, default=2)
    parser.add_argument("--use-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--eval-model", default="llama3.1:8b")
    parser.add_argument("--skip-ollama-eval", action="store_true")
    args = parser.parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.spam_data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    gpt2_files = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]

    def local_gpt2_files_ready(model_dir):
        return all((model_dir / name).exists() and (model_dir / name).stat().st_size > 0 for name in gpt2_files)

    def load_pretrained_gpt2(choose_model):
        config = base_config.copy()
        config.update(model_configs[choose_model])
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        model_dir = args.models_dir / model_size

        gpt_download_path = repo_root / "ch05" / "01_main-chapter-code" / "gpt_download.py"
        spec = spec_from_file_location("ch05_gpt_download", gpt_download_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {gpt_download_path}")

        gpt_download = module_from_spec(spec)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            spec.loader.exec_module(gpt_download)
            if local_gpt2_files_ready(model_dir):
                settings = json.loads((model_dir / "hparams.json").read_text(encoding="utf-8"))
                params = gpt_download.load_gpt2_params_from_tf_ckpt(str(model_dir / "model.ckpt"), settings)
            else:
                settings, params = gpt_download.download_and_load_gpt2(
                    model_size=model_size,
                    models_dir=args.models_dir,
                )

        model = GPTModel(config)
        load_weights_into_gpt(model, params)
        model.to(device)
        return model, config, settings

    def free_model(model):
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def query_ollama(prompt, model_name, url="http://localhost:11434/api/chat"):
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "seed": 123,
                "temperature": 0,
                "num_ctx": 2048,
            },
        }
        response_text = ""
        with requests.post(url, json=data, stream=True, timeout=120) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                response_json = json.loads(line)
                if "message" in response_json:
                    response_text += response_json["message"]["content"]
        return response_text

    def evaluate_instruction_responses(json_data, model_name):
        scores = []
        print("Ollama evaluation model:", model_name)
        for index, entry in enumerate(json_data, start=1):
            if entry["model_response"] == "":
                scores.append(0)
                continue

            prompt = (
                f"Given the input `{format_instruction_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry['model_response']}` "
                f"on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            score_text = query_ollama(prompt, model_name).strip()
            try:
                scores.append(int(score_text))
            except ValueError:
                print(f"Could not convert score for entry {index}: {score_text}")

            if index % 10 == 0:
                print(f"Scored {index}/{len(json_data)} entries")

        return scores

    print("Device:", device)
    print("=" * 50)

    print("Chapter 6: GPT-2 small spam classification")
    data_dir = args.spam_data_dir
    data_file_path = data_dir / "SMSSpamCollection.tsv"
    zip_path = data_dir / "sms_spam_collection.zip"

    try:
        download_and_unzip_spam_data(
            "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
            zip_path,
            data_dir,
            data_file_path,
        )
    except (requests.exceptions.RequestException, TimeoutError):
        download_and_unzip_spam_data(
            "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip",
            zip_path,
            data_dir,
            data_file_path,
        )

    spam_df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    spam_df = create_balanced_dataset(spam_df)
    spam_df["Label"] = spam_df["Label"].map({"ham": 0, "spam": 1})
    train_df, val_df, test_df = random_split(spam_df, train_frac=0.7, validation_frac=0.1)

    train_dataset = SpamDataset(train_df, tokenizer)
    val_dataset = SpamDataset(val_df, tokenizer, max_length=train_dataset.max_length)
    test_dataset = SpamDataset(test_df, tokenizer, max_length=train_dataset.max_length)

    ch06_model_name = "gpt2-small (124M)"
    ch06_model, ch06_config, _ = load_pretrained_gpt2(ch06_model_name)
    assert train_dataset.max_length <= ch06_config["context_length"]
    print("Loaded model:", ch06_model_name)
    print("Training set length:", len(train_dataset))
    print("Validation set length:", len(val_dataset))
    print("Test set length:", len(test_dataset))

    train_loader_cls = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=0)
    val_loader_cls = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=False, num_workers=0)
    test_loader_cls = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False, num_workers=0)

    torch.manual_seed(123)
    ch06_model = prepare_model_for_classification(
        model=ch06_model,
        emb_dim=ch06_config["emb_dim"],
        num_classes=2,
        train_last_block=True,
    )
    ch06_model.to(device)

    print("Initial accuracies")
    print("   Training accuracy:", calc_classification_accuracy_loader(train_loader_cls, ch06_model, device, num_batches=10))
    print("   Validation accuracy:", calc_classification_accuracy_loader(val_loader_cls, ch06_model, device, num_batches=10))
    print("   Test accuracy:", calc_classification_accuracy_loader(test_loader_cls, ch06_model, device, num_batches=10))

    start_time = time.time()
    optimizer = torch.optim.AdamW(ch06_model.parameters(), lr=5e-5, weight_decay=0.1)
    train_classifier_simple(
        model=ch06_model,
        train_loader=train_loader_cls,
        val_loader=val_loader_cls,
        optimizer=optimizer,
        device=device,
        num_epochs=args.ch06_epochs,
        eval_freq=50,
        eval_iter=5,
    )
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")

    print("Final accuracies")
    print("   Training accuracy:", calc_classification_accuracy_loader(train_loader_cls, ch06_model, device))
    print("   Validation accuracy:", calc_classification_accuracy_loader(val_loader_cls, ch06_model, device))
    print("   Test accuracy:", calc_classification_accuracy_loader(test_loader_cls, ch06_model, device))
    print("=" * 50)
    free_model(ch06_model)

    print("Chapter 7: GPT-2 medium instruction finetuning")
    instruction_path = repo_root / "ch07" / "01_main-chapter-code" / "instruction-data.json"
    instruction_data = download_and_load_json_file(
        instruction_path,
        url="https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json",
    )
    train_data, val_data, test_data = split_instruction_data(instruction_data)
    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    ch07_model_name = "gpt2-medium (355M)"
    ch07_model, ch07_config, _ = load_pretrained_gpt2(ch07_model_name)
    ch07_model.eval()
    print("Loaded model:", ch07_model_name)

    if args.use_lora:
        freeze_model_parameters(ch07_model)
        replace_linear_with_lora(ch07_model, rank=16, alpha=16)
        ch07_model.to(device)
        print("Appendix E LoRA: enabled")
    else:
        print("Appendix E LoRA: disabled; full finetuning")
    print(f"Total parameters: {count_parameters(ch07_model):,}")
    print(f"Trainable parameters: {count_parameters(ch07_model, only_trainable=True):,}")

    train_dataset_inst = InstructionDataset(train_data, tokenizer)
    val_dataset_inst = InstructionDataset(val_data, tokenizer)
    train_loader_inst = DataLoader(
        train_dataset_inst,
        batch_size=8,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=lambda batch: instruction_collate_fn(
            batch,
            allowed_max_length=ch07_config["context_length"],
            device=device,
        ),
    )
    val_loader_inst = DataLoader(
        val_dataset_inst,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=lambda batch: instruction_collate_fn(
            batch,
            allowed_max_length=ch07_config["context_length"],
            device=device,
        ),
    )

    print("Initial losses")
    with torch.no_grad():
        print("   Training loss:", calc_loss_loader(train_loader_inst, ch07_model, device, num_batches=5))
        print("   Validation loss:", calc_loss_loader(val_loader_inst, ch07_model, device, num_batches=5))

    start_time = time.time()
    optimizer = torch.optim.AdamW(
        (param for param in ch07_model.parameters() if param.requires_grad),
        lr=5e-5,
        weight_decay=0.1,
    )
    torch.manual_seed(123)
    warmup_steps = max(1, int(0.2 * len(train_loader_inst) * args.ch07_epochs))
    train_model_with_appendix_d(
        model=ch07_model,
        train_loader=train_loader_inst,
        val_loader=val_loader_inst,
        optimizer=optimizer,
        device=device,
        num_epochs=args.ch07_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=format_instruction_input(val_data[0]),
        tokenizer=tokenizer,
        warmup_steps=warmup_steps,
        initial_lr=3e-5,
        peak_lr=5e-5,
        min_lr=1e-6,
        max_grad_norm=1.0,
    )
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} minutes.")

    print("Generating responses")
    for entry in test_data:
        input_text = format_instruction_input(entry)
        token_ids = generate(
            model=ch07_model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=args.max_new_tokens,
            context_size=ch07_config["context_length"],
            eos_id=50256,
        )
        generated_text = token_ids_to_text(token_ids.cpu(), tokenizer)
        response_text = generated_text[len(input_text) :].replace("### Response:", "").strip()
        response_text = response_text.split("### Instruction:")[0].strip()
        response_text = response_text.split("<|endoftext|>")[0].strip()
        entry["model_response"] = response_text

    responses_path = args.output_dir / "instruction-data-with-response-standalone.json"
    with open(responses_path, "w", encoding="utf-8") as file:
        json.dump(test_data, file, indent=4, ensure_ascii=False)
    print(f"Responses saved as {responses_path}")

    suffix = "lora" if args.use_lora else "full"
    model_path = args.output_dir / f"{re.sub(r'[ ()]', '', ch07_model_name)}-sft-{suffix}-standalone.pth"
    torch.save(ch07_model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

    if args.skip_ollama_eval:
        return

    print("=" * 50)
    print("Chapter 7: Ollama/Llama evaluation")
    try:
        scores = evaluate_instruction_responses(test_data, args.eval_model)
        print(f"Number of scores: {len(scores)} of {len(test_data)}")
        if scores:
            print(f"Average score: {sum(scores) / len(scores):.2f}")
    except requests.exceptions.RequestException as exc:
        print("Ollama evaluation skipped.")
        print("Reason:", exc)
        print("Make sure Ollama is running and the model is pulled, for example: ollama pull llama3.1:8b")


if __name__ == "__main__":
    main()
