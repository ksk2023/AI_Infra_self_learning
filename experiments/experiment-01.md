# 实验 1：模型到底收到了什么？—— Chat 模板揭秘

> **改动**：在生成之前，把 tokenize 之后的输入重新解码并打印出来（新增 3 行）。
> **方法**：一次只改一处 → 预测输出变化 → 运行验证。
> **Commit**：`ad74f03`（实验 1~3 同属基线提交，git 仓库建立于实验 3 之后）

## 前置说明

在本实验之前有一个小改动：`tokenizer.decode(..., skip_special_tokens=True)` 改为 `False`。
作用是**保留特殊 token**，让我们能看到 `<|im_end|>` 这类"格式控制符"——它是后续实验 2、4 的观察基础。

## 改动点（Diff）

```diff
 print("3. Preparing input...")
 ...
 ).to("cuda")
 
+print("===== Model Input (chat template) =====")
+print(tokenizer.decode(inputs[0]))
+print("========================================")
+
 print("4. Generating response...")
```

## 核心代码

```python
print("===== Model Input (chat template) =====")
print(tokenizer.decode(inputs[0]))   # inputs 形状为 (1, seq_len)，取第一条序列
print("========================================")
```

- `apply_chat_template(..., return_tensors="pt")` 返回 `(batch=1, seq_len)` 的 token 张量；
- 把它**解码回文本**打印，等于偷看"模型真正读到的东西"；
- `inputs[0]` 取出该批次唯一一条序列。

## 运行方式

```bash
python3 scripts/first_inference.py
```

## 实验结果（实测）

脚本里只写了一条 `user` 消息，但模型实际收到的是：

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
请解释大语言模型推理中的 Prefill、Decode 和 KV Cache。<|im_end|>
<|im_start|>assistant
```

三个关键观察：

1. 出现了一条**脚本里没写过的 system prompt**；
2. 输入以 `<|im_start|>assistant\n` 结尾（`add_generation_prompt=True` 的作用）——这就是"请开始回答"的信号；
3. 所谓"对话"，本质是用特殊 token 拼出来的一段**连续文本**。

## 原理与结论

- Qwen2.5 的 chat 模板在**没有 system 消息时自动注入默认 system prompt**（"You are Qwen..."）。模型看到的输入和你在代码里写的消息列表并不一一对应——中间隔着一层模板；
- `<|im_start|>role\n...<|im_end|>` 是 Qwen 的对话格式控制符，模型在训练时见过海量这种结构，所以"知道"在哪里切换角色；
- 多轮对话 = 把历史消息全部按此格式拼进同一段文本，越拼越长。

## 延伸推论

- **调试 LLM 应用的第一件事**：把模板渲染结果打出来看。很多"模型不按格式回答"的问题，根源都是模板渲染和预期不符（比如忘了 `add_generation_prompt`，模型会以为该继续扮演 user）；
- 不同模型的模板不通用（Llama 用 `[INST]`、ChatML 系用 `<|im_start|>`），这就是 `apply_chat_template` 存在的意义——它读取模型自带的 `tokenizer_config.json` 里的模板定义。

<details>
<summary>📄 完整代码（本实验时刻，点击展开）</summary>

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/dev/infra/models/Qwen2.5-1.5B-Instruct"

print("1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("2. Loading model to RTX 4090...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True
).to("cuda").eval()

print("3. Preparing input...")

messages = [
    {
        "role": "user",
        "content": "请解释大语言模型推理中的 Prefill、Decode 和 KV Cache。"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

print("===== Model Input (chat template) =====")
print(tokenizer.decode(inputs[0]))
print("========================================")

print("4. Generating response...")

with torch.inference_mode():
    outputs = model.generate(
        inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

answer_tokens = outputs[0][inputs.shape[-1]:]
answer = tokenizer.decode(
    answer_tokens,
    skip_special_tokens=False
)

print("\n===== Qwen Response =====")
print(answer)

print("\n===== GPU Information =====")
print("GPU:", torch.cuda.get_device_name(0))
print(
    "Allocated memory:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GiB"
)
```

</details>

[← 返回总览](../README.md)
