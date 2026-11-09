# 实验 2：max_new_tokens=32 —— 截断 vs 自然停止

> **改动**：`max_new_tokens` 从 256 改为 32（单行）。
> **Commit**：`ad74f03`

## 改动点（Diff）

```diff
     outputs = model.generate(
         inputs,
-        max_new_tokens=256,
+        max_new_tokens=32,
         do_sample=False,
         pad_token_id=tokenizer.eos_token_id
     )
```

## 核心代码

```python
outputs = model.generate(
    inputs,
    max_new_tokens=32,      # 长度预算：最多生成 32 个新 token
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)
```

`max_new_tokens` 是**生成新 token 的预算上限**，与输入 prompt 长度无关（区别于 `max_length`）。

## 运行方式

```bash
python3 scripts/first_inference.py
```

## 实验结果（实测）

| 配置 | 回答结尾 |
|---|---|
| `max_new_tokens=256` | `...处理大规模的文本任务。` + **`<\|im_end\|>`**（完整结束） |
| `max_new_tokens=32` | `...Prefill：这是指在推理开始之前，模型会预先填`（**没有** `<\|im_end\|>`，拦腰截断） |

32 token 只够说一两句话，且结尾没有任何特殊 token。

## 原理与结论

- 生成结束有两种方式：**自然停止**（模型吐出 `<|im_end|>`，引擎的 eos 检查触发）或**长度截断**（预算用完被强制掐断）；
- 由于前置改动 `skip_special_tokens=False` 保留了特殊 token，**看一眼结尾有没有 `<|im_end|>` 就能立刻判断是哪种**：
  - 有 → 模型自己"说完了"
  - 没有 → 被截断了

一句话：**截断看上限，停止看 eos。**

## 延伸推论

- 这是生产环境的常用排查手段：用户反馈"回答突然断了"，先检查 `max_new_tokens` / `max_length` 是否太小，而非怀疑模型"不想说"；
- 截断的 token 数和回答语言有关：中文大约 1~1.5 字/token，估算预算时要按目标语言的 token 密度换算；
- `finish_reason`（stop / length）—— OpenAI 风格 API 返回的这个字段，就是引擎替你做了本实验的"看结尾"判断。

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
        max_new_tokens=32,
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
