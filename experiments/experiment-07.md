# 实验 7：top_k / top_p —— 两道保险丝

> **改动**：`top_k`、`top_p` 也变为命令行参数（`sys.argv[2]`/`sys.argv[3]`，默认 20/0.8 = Qwen 预置值；`0` 和 `1.0` 分别表示"不限制"）。
> **Commit**：`5eb3c37`

## 改动点（Diff）

```diff
 temperature = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
+top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 20        # 0 = 不限制
+top_p = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8     # 1.0 = 不限制
-print(f"4. Generating response (temperature={temperature})...")
+print(f"4. Generating response (temperature={temperature}, top_k={top_k}, top_p={top_p})...")
 
     outputs = model.generate(
         ...
         temperature=temperature,
+        top_k=top_k,
+        top_p=top_p,
         pad_token_id=tokenizer.eos_token_id
```

## 核心代码

```python
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 20     # 候选池：只留概率最高的 K 个
top_p = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8  # 候选池：留累计概率 p 以内
```

## 运行方式（A/B 对照）

```bash
# A组：高温 + 拆掉两道保险丝（top_k=0 不限候选数，top_p=1.0 不截断概率）
python3 scripts/first_inference.py 1.5 0 1.0

# B组（对照组）：同样高温，保险丝还在（默认 top_k=20, top_p=0.8）
python3 scripts/first_inference.py 1.5
```

## 实验结果（实测）

- **A 组**：语无伦次——话说一半突然跑题、用词诡异、中英夹杂、不合语法的拼接，偶尔蹦出正常句子又马上崩掉；
- **B 组**：放飞但仍然"像人话"——温度同样高，但骰子只能在受限候选池里摇。

## 原理与结论

两道保险丝都在**限定骰子的候选池**，发生在温度之后：

| 参数 | 机制 | 好比 |
|------|------|------|
| `top_k=20` | 只保留概率最高的 20 个候选再摇骰子 | 海选只留前 20 名 |
| `top_p=0.8` | 保留累计概率达 80% 的候选（数量随分布浮动） | 保底覆盖 80% 胜率 |

- 高温把分布压平后，排名 500 开外的冷门 token 概率不再近乎为零——A 组里它们开始中奖，这就是语无伦次的来源；
- **top_p 更"聪明"**：分布很尖时它可能只留两三个候选，分布很平时可能留几十个，自适应。实践中通常二选一或组合使用；Qwen 官方默认 `20 / 0.8` 双保险。

## 延伸推论

- 完整的采样流水线顺序：**logits → ÷T（温度）→ top_k 截断 → top_p 截断 → softmax → 掷骰子**；
- "模型放飞/胡言乱语"的线上事故，排查清单第一条就是检查采样参数（高温 + 宽松截断的组合）；
- 与贪心的取舍：贪心 = 把候选池缩到 1，绝对稳但呆板；采样三件套是在"稳"与"活"之间连续调节的旋钮。

<details>
<summary>📄 完整代码（本实验时刻，点击展开）</summary>

```python
import sys
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

temperature = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 20        # 0 = 不限制
top_p = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8     # 1.0 = 不限制
print(f"4. Generating response (temperature={temperature}, top_k={top_k}, top_p={top_p})...")

with torch.inference_mode():
    outputs = model.generate(
        inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
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
