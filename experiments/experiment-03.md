# 实验 3：去掉切片 —— generate 到底返回了什么

> **改动**：`outputs[0][inputs.shape[-1]:]` → `outputs[0]`（不再切掉输入部分，单行）。
> **Commit**：`ad74f03`

## 改动点（Diff）

```diff
-answer_tokens = outputs[0][inputs.shape[-1]:]
+answer_tokens = outputs[0]
```

## 核心代码

```python
# 原来：从已知边界切掉 prompt，只留新增部分
answer_tokens = outputs[0][inputs.shape[-1]:]
# 现在：整条序列一起解码（prompt + 生成内容）
answer_tokens = outputs[0]
```

## 运行方式

```bash
python3 scripts/first_inference.py
```

## 实验结果（实测）

`Qwen Response` 段变成**输入 + 输出连在一起**的完整文本：

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
请解释大语言模型推理中的 Prefill、Decode 和 KV Cache。<|im_end|>
<|im_start|>assistant
在大语言模型的推理过程中，Prefill、Decode和KV Cache是三个重要的概念。
...（后续被 max_new_tokens=32 截断）
```

三个关键观察：

1. `<|im_start|>assistant` 与回答正文**直接相连**，中间只有一个换行；
2. generate 返回的是 **prompt + 新生成** 的完整序列；
3. 本实验开头的 "Model Input" 段和 Response 段的**开头逐字一致**。

## 原理与结论

- **模型眼里没有"输入"和"输出"，只有一条文本流**。它做的唯一一件事是：给定前文，续写下一个 token；续写结果直接拼在前文后面；
- "切片"不是模糊的截断，而是**按已知边界精确摘取**——prompt 长度已知（`inputs.shape[-1]`），从那里一刀切下去，切出来的必然是纯回答；
- 一句话：**续写是模型的本质，切片是产品的礼貌。**

## 延伸推论

- 你在聊天产品里只看到回答，看不到模板和拼回去的问题——应用层做了同样的切片处理。若某产品输出里漏出了 `<|im_end|>` 或你的问题被复述一遍，就是这一步后处理没做干净；
- "多轮对话"在模型侧并不存在状态：每一轮都是把**全部历史**拼成一条长文本重新喂入（配合 KV Cache 复用，见实验 9/10）。所谓对话记忆，就是不断变长的 prompt。

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

answer_tokens = outputs[0]
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
