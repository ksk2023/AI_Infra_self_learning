# 实验 8：固定随机种子 —— 混乱也可以复现

> **改动**：新增可选第 4 个参数 `seed`，传入则执行 `torch.manual_seed(seed)`。
> **Commit**：`522b78f`

## 改动点（Diff）

```diff
 top_p = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8     # 1.0 = 不限制
+seed = int(sys.argv[4]) if len(sys.argv) > 4 else None
+if seed is not None:
+    torch.manual_seed(seed)  # 实验八：固定随机种子，让骰子可复现
-print(f"4. Generating response (temperature={temperature}, top_k={top_k}, top_p={top_p})...")
+print(f"4. Generating response (temperature={temperature}, top_k={top_k}, top_p={top_p}, seed={seed})...")
```

## 核心代码

```python
seed = int(sys.argv[4]) if len(sys.argv) > 4 else None
if seed is not None:
    torch.manual_seed(seed)  # 固定 RNG 初始状态 → 固定后续"随机"数列
```

## 运行方式（用最狂野的参数，效果最震撼）

```bash
python3 scripts/first_inference.py 1.5 0 1.0 42   # 同种子跑两遍
python3 scripts/first_inference.py 1.5 0 1.0 42
python3 scripts/first_inference.py 1.5 0 1.0 43   # 换种子
```

## 实验结果（实测）

1. 前两次（seed=42）输出**一字不差**——哪怕参数是"高温 + 无保险丝"的完全体混沌，也被精确复现；
2. 第三次（seed=43）输出完全不同，但"疯狂程度"一致。

## 原理与结论

- 采样的"随机"数来自**伪随机数生成器（PRNG）**：从初始状态（种子）出发按固定算法推演出数列。固定种子 = 固定初始状态 = 每一摇的结果都被写死；
- 所谓随机性，从头到尾只是"**你不知道种子**"——与量子噪声那类真随机有本质区别；
- **精确化**：
  - **同环境**下：不是"大概率一样"，是**数学上 100% 一样**（纯确定性算法）；
  - **跨硬件/库版本**：浮点求和顺序不同 → logits 小数点后第 6~7 位差异 → **若两个候选 token 概率几乎打平则排名翻转** → 从那一枚 token 起整个轨迹分叉（蝴蝶效应）。

## 延伸推论

- **"要么逐字一致、要么完全分叉"，不存在"输出有少许差异"的中间态**——文本生成强序列依赖，一步错步步错；分叉点之前的文本仍然逐字相同；
- **分叉只发生在"模型犹豫"的位置**（分布平坦、前几名几乎并列），模型笃定的位置（尖峰分布）微小偏差无害；
- 工程用途：
  1. **可复现的 bug 报告**：生产环境记录每次请求的 `(prompt, 参数, seed, 模型版本)`，用户说"刚才输出很诡异"，随时精确重放；
  2. **Prompt 回归测试**：固定 seed 后改 prompt，输出变化只来自 prompt 本身，而非骰子噪声；
  3. 论文/实验复现；
- 严格的跨环境复现要固定四件套：**模型版本 + 库版本 + 硬件 + 种子**；
- 蝴蝶效应的实拍证据 → 见实验 10b（两次贪心解码因 kernel 路径不同而停在 428 vs 434 步）。

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
seed = int(sys.argv[4]) if len(sys.argv) > 4 else None
if seed is not None:
    torch.manual_seed(seed)  # 实验八：固定随机种子，让骰子可复现
print(f"4. Generating response (temperature={temperature}, top_k={top_k}, top_p={top_p}, seed={seed})...")

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
