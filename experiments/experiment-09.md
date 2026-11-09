# 实验 9：手写解码循环 —— 拆开 generate 的黑盒

> **改动**：不用 `model.generate`，自己写 for 循环逐 token 生成（贪心 argmax），每步打印输入 token 数、耗时、新 token；停止检查也自己写。新增 `import time`。
> **Commit**：`7d15555`

## 改动点（Diff）

在自动 generate（对照组，保留）之后新增约 25 行：

```diff
+import time
 ...
+print("\n===== Manual decode loop (Prefill + Decode + KV Cache) =====")
+
+im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
+stop_ids = {tokenizer.eos_token_id, im_end_id}
+
+step_inputs = inputs          # 第一次喂入整个 prompt（Prefill）
+past_kv = None                # KV Cache，从第二步开始复用
+with torch.inference_mode():
+    for step in range(128):
+        ...  # 详见下方核心代码
```

## 核心代码（全场的 4 个关键动作）

```python
step_inputs = inputs                  # 第一次喂入整个 prompt（Prefill）
past_kv = None                        # KV Cache 容器
for step in range(128):
    out = model(step_inputs, past_key_values=past_kv, use_cache=True)  # ① 一次前向
    next_id = int(out.logits[0, -1].argmax())   # ② 只看最后一个位置挑 token
    if next_id in stop_ids:                    # ③ 停止纪律由自己执行
        break
    past_kv = out.past_key_values              # ④ K/V 存入缓存
    step_inputs = inputs.new_tensor([[next_id]])  # ④' 下一步只喂这 1 个新 token
```

| 动作 | 行 | 说明 |
|---|---|---|
| ① 前向传播 | `model(step_inputs, past_key_values=past_kv, use_cache=True)` | step 0 吃整个 prompt（Prefill）；之后只吃 1 个 token，历史全靠 `past_kv` 携带（Decode + KV Cache 复用） |
| ② 取 token | `out.logits[0, -1].argmax()` | 输出里**只看最后一个位置**的概率分布；前面位置的 logits 直接丢弃——"续写"的全部 |
| ③ 停止 | `if next_id in stop_ids: break` | 实验 4 证明的"外部纪律"，这次由我们亲手执行 |
| ④ 喂食 | `past_kv = out.past_key_values` + 只喂新 token | 自我喂食循环：每步生成的 token 转身成为下一步的唯一输入 |

**注意**：`next_id`（本步新生成的 token）就是下一步的**全部输入**——模型从头到尾没"收到"过自己的完整回答，每步只见上一枚 token + 缓存中的全部历史。

## 运行方式

```bash
python3 scripts/first_inference.py
```

## 实验结果（实测，节选）

```
step   0 | input  40+ tok |   xx ms | ...     ← Prefill：整个 prompt 一次并行算完
step   1 | input   1 tok  |  17.2 ms | '在'
step   2 | input   1 tok  |  17.5 ms | '大'
...
step 116 | input   1 tok  |  17.9 ms | '。\n'
step 117 | input   1 tok  |  17.1 ms | '  '
step 118 | input   1 tok  |  17.4 ms | ' -'
step 119 | input   1 tok  |  17.5 ms | ' **'
```

关键观察：

1. **step 0：`input 40+ tok`** —— 整个 prompt 一次性喂入，这就是 **Prefill**（并行算完所有位置的 KV）；
2. **step 1 起：`input 1 tok` 恒定不变** —— 每步只喂 1 个新 token，这就是 **Decode**；
3. 每步约 **17~21 ms**，串行等待——第 k 步的输入依赖第 k-1 步的输出，**无法并行**；
4. `repr` 显示 token 粒度：中文常一字一 token，英文单词可能整个一 token，还有 `'。\n'`、`' **'` 这类混合 token；
5. step 116 之后出现 markdown 格式符（` - `、` **应用场景**`）——不是模型"知道"要排版，纯粹是训练数据里满是 markdown，贪心续写的最可能路径长这样；
6. 手写贪心循环的文本 ≈ `generate(do_sample=False)` 的输出——**generate 的内核就是这个循环**。

## 原理与结论：Prefill / Decode / KV Cache

| 阶段 | 定义 | 计算特征 |
|------|------|---------|
| **Prefill** | 第一次前向：整段 prompt **一次并行**处理，算出所有位置的 K/V 灌入缓存，**顺手产出第 1 枚新 token** | 一步大计算，矩阵大，**算力型**（GPU 吃得饱） |
| **Decode** | 从第 2 枚 token 起，**每步进 1 出 1** 的串行续写，历史以缓存形式参与 | N 次小计算，**访存型**（每步搬全部权重+缓存只为算 1 个 token） |
| **KV Cache** | 存下已算好的 K/V；每步新 token 的 K/V **追加**进缓存尾部 | 历史只读不重算；缓存每步长一截 |

记忆口诀：**Prefill = 一次并行读完整个考题，写下答案第一个字；Decode = 一个字一个字往下写。**
一次请求 = 1 次 Prefill + N 次 Decode。

常见误区（本系列第一天 Qwen 自己都答错了）：Prefill **不是**"把数据预先填充到内存"——填充缓存只是副产品，它的定义是**整段 prompt 的首次并行处理**。

## 延伸推论：TTFT 与 TPOT

两个阶段分别对应两项核心服务指标：

| 指标 | 全称 | 由谁决定 | 用户感受 |
|---|---|---|---|
| **TTFT** | Time To First Token（首 token 延迟） | Prefill（+排队+权重加载） | "它开始打字了"的快慢 |
| **TPOT** | Time Per Output Token（单 token 生成时间） | Decode | "打字速度" |

$$总生成耗时 \approx TTFT + N_{tokens} \times TPOT$$

**用本实验实测数据验算**：428 token × 20.2 ms/token ≈ **8.65 s**，基准实测 **8.66 s** ✓

- 优化 TTFT：更快的 Prefill（算子融合、chunked prefill、更大 batch 的并行度）；
- 优化 TPOT：更快的 Decode（CUDA Graph 消除 kernel 发射开销、连续批处理 continuous batching、投机解码）；
- 两者常互相冲突（如提高并发摊薄单卡效率会拉长 TTFT 但提升总吞吐）——这是推理服务调度的核心矛盾；
- **投机解码（speculative decoding）**：用小模型一次猜多个 token、大模型一次验证，部分打破 Decode 的串行依赖——因为自我喂食循环（本实验的核心观察）是串行的根源。

<details>
<summary>📄 完整代码（本实验时刻，点击展开）</summary>

```python
import sys
import time
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

print("\n===== Manual decode loop (Prefill + Decode + KV Cache) =====")

im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
stop_ids = {tokenizer.eos_token_id, im_end_id}

step_inputs = inputs          # 第一次喂入整个 prompt（Prefill）
past_kv = None                # KV Cache，从第二步开始复用
with torch.inference_mode():
    for step in range(128):
        t = time.perf_counter()
        out = model(step_inputs, past_key_values=past_kv, use_cache=True)
        dt_ms = (time.perf_counter() - t) * 1000

        next_id = int(out.logits[0, -1].argmax())   # 贪心：只看最后一个位置
        piece = tokenizer.decode(next_id)
        print(
            f"step {step:3d} | input {step_inputs.shape[-1]:3d} tok"
            f" | {dt_ms:6.1f} ms | {piece!r}",
            flush=True,
        )

        if next_id in stop_ids:
            print(f"-> 命中停止符，循环结束（这次'纪律'由我们自己执行）")
            break
        past_kv = out.past_key_values       # 把 K/V 存起来给下一步
        step_inputs = inputs.new_tensor([[next_id]])  # 之后每步只喂 1 个新 token（Decode）

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
