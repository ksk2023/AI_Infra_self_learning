# 实验 5：do_sample=True —— 从确定到掷骰子

> **改动**：`do_sample=False` → `True`；同时移除实验 4 的 `eos_token_id=None`（恢复停止纪律，避免自问自答干扰观察）。
> **Commit**：`5cb9510`

## 改动点（Diff）

```diff
     outputs = model.generate(
         inputs,
         max_new_tokens=128,
-        do_sample=False,
-        pad_token_id=tokenizer.eos_token_id,
-        eos_token_id=None  # 实验四：不设停止符，模型无法"喊停"
+        do_sample=True,  # 实验五：采样生成，每次结果不同
+        pad_token_id=tokenizer.eos_token_id
     )
```

## 核心代码

```python
outputs = model.generate(
    inputs,
    max_new_tokens=128,
    do_sample=True,  # 按概率分布采样，而非永远取最大概率
    pad_token_id=tokenizer.eos_token_id
)
```

## 运行方式（本实验起需连跑多次对比）

```bash
python3 scripts/first_inference.py   # 连续运行 2~4 次
```

## 实验结果（实测）

连续运行 **4 次，4 次回答各不相同**（措辞、结构、举例都有差异）。同时两个附带观察：

1. **开头往往相似，越往后分歧越大**——前几个 token 概率分布很尖（模型很确定），后面越来越发散；
2. 顶部刷屏的三条警告**消失了**：`temperature is set to 0.7 -- this flag is only used in sample-based generation`（top_p/top_k 同理）。

## 原理与结论

| 模式 | 挑选下一个 token 的方式 | 特点 |
|------|----------------------|------|
| `do_sample=False`（贪心） | 永远选概率最大的（argmax） | 确定、稳定，但呆板、易重复 |
| `do_sample=True`（采样） | 按概率分布"掷骰子" | 多样、有创造性，但每次不同 |

- 贪心是确定性的：同输入跑 100 次一字不差（本系列实验 1~4 的输出全部可复现）；
- 警告消失的原因：Qwen 的 `generation_config.json` 预置了 `temperature=0.7, top_p=0.8, top_k=20`。贪心模式下这些参数用不到（所以 transformers 警告），采样模式下真正生效；
- 解码时模型输出的不是"一个词"，而是**全词表的概率分布**——贪心和采样只是两种从这个分布里取值的策略。

## 延伸推论

- Chat 产品的"重新生成"按钮给你不同答案、模型偶尔"发挥过头"，都来自这颗骰子；
- 什么时候用哪个：**代码/数学/事实抽取 → 贪心或低温采样**（要稳定）；**创意写作/起名/头脑风暴 → 采样**（要多样）；
- 采样随机性的可控与复现问题 → 见实验 6（temperature 调骰子偏心）与实验 8（固定种子）。

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
        do_sample=True,  # 实验五：采样生成，每次结果不同
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
