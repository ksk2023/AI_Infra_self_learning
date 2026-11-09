# 实验 4：eos_token_id=None —— 谁在让模型"闭嘴"

> **改动**：`generate()` 增加 `eos_token_id=None`（关闭停止符检查）；同时 `max_new_tokens` 32→128 恢复生成空间以便观察。
> **Commit**：`d33fcba`

## 改动点（Diff）

```diff
     outputs = model.generate(
         inputs,
-        max_new_tokens=32,
-        do_sample=False,
-        pad_token_id=tokenizer.eos_token_id
+        max_new_tokens=128,
+        do_sample=False,
+        pad_token_id=tokenizer.eos_token_id,
+        eos_token_id=None  # 实验四：不设停止符，模型无法"喊停"
     )
```

## 核心代码

```python
outputs = model.generate(
    inputs,
    max_new_tokens=128,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=None  # 告诉引擎：没有停止符，生成到 128 上限为止
)
```

`eos_token_id=None` 移除了引擎的停止条件——注意这是**引擎参数**，不是模型参数。

## 运行方式

```bash
python3 scripts/first_inference.py
```

## 实验结果（实测）

模型说完一段完整回答后**停不下来**，输出大致为：

```
...它们共同作用，使得模型能够高效地处理大规模的文本任务。<|im_end|>
<|im_start|>user
请再详细解释一下KV Cache的作用<|im_end|>
<|im_start|>assistant
好的，KV Cache的主要作用是...（持续到 128 token 上限）
```

模型生成完 `<|im_end|>` 后，开始**自己编造用户提问、自己回答**，无限循环下去。

## 原理与结论

1. `<|im_end|>` 对模型来说**只是一个普通的词**。它"知道"该在这里结束回合，是因为训练时见过无数次这种模式，并非内部有什么"停止开关"；
2. 真正让生成停下来的是 `generate()` 的**外部检查**：每生成一个 token 就问一次"这是不是 eos？是就停"。`eos_token_id=None` 把这个检查关掉了；
3. 停不下来之后，模型只能按训练数据里最可能的模式续写——而最可能的模式就是"**用户又问了新问题**"，于是自问自答。

一句话：**"停止"不是模型的能力，是推理引擎的纪律。**

## 延伸推论

- vLLM 等推理框架的 `stop_token_ids`（追加停止符）、`ignore_eos`（无视停止符）参数，管的都是同一件事：**停止纪律在引擎侧**；
- Qwen 的 `generation_config.json` 里 `eos_token_id` 是一个**列表** `[151645, 151643]`（`<|im_end|>` 与 `<|endoftext|>` 双停止符）——因为 Qwen 训练语料两种结束符都出现过；
- 安全视角：越狱类攻击有时会诱导模型"扮演"对话格式，本质就是干扰这套 token 级的纪律体系。

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
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=None  # 实验四：不设停止符，模型无法"喊停"
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
