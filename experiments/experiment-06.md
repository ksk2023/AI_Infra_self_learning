# 实验 6：temperature —— 骰子的偏心旋钮

> **改动**：`temperature` 变为命令行参数（`sys.argv[1]`，默认 0.7 = Qwen 官方推荐值）；新增 `import sys`。
> **Commit**：`2e5e15e`

## 改动点（Diff）

```diff
+import sys
 import torch
 from transformers import AutoTokenizer, AutoModelForCausalLM
 ...
+temperature = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
+print(f"4. Generating response (temperature={temperature})...")
 
 with torch.inference_mode():
     outputs = model.generate(
         inputs,
         max_new_tokens=128,
         do_sample=True,
+        temperature=temperature,
         pad_token_id=tokenizer.eos_token_id
     )
```

## 核心代码

```python
temperature = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
# ...传给 generate
temperature=temperature,
```

把实验变量暴露成 CLI 参数后，无需改代码就能自由对比——这也是"受控实验"的工程化做法。

## 运行方式（两组各跑 2~3 次）

```bash
python3 scripts/first_inference.py 0.1   # 低温组
python3 scripts/first_inference.py 1.5   # 高温组
python3 scripts/first_inference.py       # 不传参 = 0.7
```

## 实验结果（实测）

| | `0.1` 低温 | `1.5` 高温 |
|---|---|---|
| 多次运行结果 | **几乎一字不差**（趋近贪心） | 每次差异巨大 |
| 语言风格 | 呆板、套话、车轱辘 | 跳跃、用词怪异、可能中英夹杂 |
| 内容质量 | 稳但无趣 | 偶有灵光，常出现讲不通的句子 |

## 原理与结论

temperature 在 softmax 之前把 logits 除以 T：

```
概率 = softmax(logits / T)
```

- **T < 1**：除以小数 → 大的更大 → 分布变**尖** → 骰子几乎总落在最高概率 token 上；T→0 退化为纯贪心；
- **T > 1**：除以大数 → 差距被压**平** → 冷门 token 也有机会中选 → 骰子变"公平"而疯狂。

温度不改排名（除法不改变大小顺序），改的是**排名之间的概率差距**。

## 延伸推论

- 业界惯例：代码/数学/事实问答用低温（0~0.3），通用对话 0.6~0.8，创意写作/头脑风暴 0.8~1.2；
- 低温的副作用是"车轱辘话"：分布太尖时，模型容易陷入 `所以说是的，所以说...` 之类的重复环（配合 `repetition_penalty` 可缓解）；
- 0.1 时多次运行"几乎"一致但理论上有极小概率翻车——想要**严格**可复现要靠固定种子（实验 8）。

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
print(f"4. Generating response (temperature={temperature})...")

with torch.inference_mode():
    outputs = model.generate(
        inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=temperature,
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
