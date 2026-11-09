# 实验 10 / 10b：use_cache 基准 —— KV Cache 的价值实测

> **改动（实验 10）**：解码循环封装为 `decode_loop(use_cache, verbose)` 函数，有/无缓存各跑一遍并输出基准报告。无缓存模式每步把 `prompt + 已生成全部 token` 重新喂入整段重算。计时加入 `torch.cuda.synchronize()`（CUDA 异步，等 GPU 真正算完再停表）。
> **改动（实验 10b）**：循环上限 `128 → 512`（单行），让二次增长显形。
> **Commit**：`e8f7197`（实验 10）、`42f09a1`（10b）

## 改动点（Diff）

```diff
+def decode_loop(use_cache: bool, verbose: bool):
+    step_inputs = inputs   # 有缓存：step0 喂整个 prompt，之后 1 个
+    full_seq = inputs      # 无缓存：每步喂 prompt + 已生成的全部 token
+    past_kv = None
     ...
      for step in range(128):     # 10b: range(512)
          t = time.perf_counter()
          if use_cache:
              out = model(step_inputs, past_key_values=past_kv, use_cache=True)
          else:
              out = model(full_seq, use_cache=False)
+         torch.cuda.synchronize()  # CUDA 是异步的：等 GPU 真正算完再停表
          ...
          if use_cache:
              past_kv = out.past_key_values
              step_inputs = inputs.new_tensor([[next_id]])
+         full_seq = torch.cat([full_seq, inputs.new_tensor([[next_id]])], dim=1)
      return total_ms, steps_done
+
+t_cache, n_cache = decode_loop(use_cache=True, verbose=True)
+t_nocache, n_nocache = decode_loop(use_cache=False, verbose=False)
```

## 核心代码

```python
def decode_loop(use_cache: bool, verbose: bool):
    step_inputs = inputs   # 有缓存：step0 喂整个 prompt（Prefill），之后 1 个（Decode）
    full_seq = inputs      # 无缓存：每步喂 prompt + 已生成的全部 token
    past_kv = None
    with torch.inference_mode():
        for step in range(512):
            if use_cache:
                out = model(step_inputs, past_key_values=past_kv, use_cache=True)
            else:
                out = model(full_seq, use_cache=False)   # 整段重算
            torch.cuda.synchronize()  # CUDA 异步：等 GPU 算完再停表
            ...
            if use_cache:
                past_kv = out.past_key_values            # K/V 存入缓存
                step_inputs = inputs.new_tensor([[next_id]])  # 只喂 1 个新 token
            full_seq = torch.cat([full_seq, ...], dim=1) # 无缓存：序列逐增长
```

**计时细节**：CUDA kernel 是异步发射的，`model(...)` 返回时 GPU 可能还没算完。不加 `synchronize()` 直接停表，量到的可能是"发射时间"而非"执行时间"——GPU 基准测试的经典陷阱。

## 运行方式

```bash
python3 scripts/first_inference.py    # 约 1 分钟：自动 generate + 两遍手写循环 + 基准报告
```

## 实验结果（实测）

**实验 10（上限 128 步）：**

```
with cache   : 128 tokens |   2.73 s |  21.3 ms/token
without cache: 128 tokens |   2.98 s |  23.3 ms/token
slowdown     : x1.09
```

无缓存模式的 `input` 列从 40 涨到 168（`full_seq` 逐增长）。

**实验 10b（上限 512 步）：**

```
with cache   : 428 tokens |   8.66 s |  20.2 ms/token
without cache: 434 tokens |  11.62 s |  26.8 ms/token
slowdown     : x1.34
```

无缓存 `input` 列持续涨到 470+，单步耗时随之爬升；缓存版从头到尾持平。

**趋势汇总：**

| | 128 步 | ~430 步 | 变化 |
|---|---|---|---|
| 缓存版 ms/token | 21.3 | **20.2** | 持平（噪声内）✅ O(n) |
| 无缓存 ms/token | 23.3 | **26.8** | +15% 持续爬升 ✅ O(n²) 显形 |
| 每生成 1 token 的额外代价 | +2.0 ms | +6.6 ms | 3.3 倍 |
| 总 slowdown | x1.09 | x1.34 | 方向明确 |

**蝴蝶效应实拍（10b 隐藏彩蛋）**：两遍都是贪心解码，理论上应走出相同轨迹、同时命中停止符——实际停止步数 **428 ≠ 434**。缓存增量计算与整段重算走不同 kernel 路径 → 浮点求和顺序不同 → logits 小数点后几位差异 → 某个概率打平点翻转 → 轨迹分叉。实验 8 讲的理论被当场抓获。

## 原理与结论

1. **KV Cache 把 O(n²) 压成 O(n)**：无缓存时第 n 步要多算 (40+n) 个 token，总重算代价 Σn ∝ n²；有缓存时每步恒定只算 1 个新 token；
2. **为什么本例倍数温和（x1.09→x1.34）**：1.5B 模型太小，单步时间由 **kernel 发射开销 + 3GB 权重读取**主导（延迟/访存瓶颈），4090 算力大量闲置——额外算术几乎免费。换大模型、长序列，无缓存从"慢一点"变成"物理不可行"；
3. **KV Cache 是纯加速、不是近似**：与整段重算在数学上等价（428≠434 只是不同 kernel 路径的浮点副作用）；
4. **代价是显存**：缓存每步增长一截；长上下文时 KV Cache 显存可超过模型权重本身。

## 延伸推论：给 KV Cache 瘦身的技术

| 技术 | 思路 | 应用 |
|---|---|---|
| **GQA**（Grouped-Query Attention） | 多个 Q 头共享一组 K/V 头，缓存体积按比例缩小（Qwen2.5-1.5B：14 个 Q 头共享 2 个 KV 头） | Qwen/Llama3 等主流模型内置 |
| **MQA** | GQA 的极限：全部 Q 头共享 1 组 KV | 更老的方案 |
| **PagedAttention** | 借操作系统分页思想管理缓存，消除内部碎片、支持共享前缀 | vLLM 的核心技术 |
| **量化缓存** | K/V 存 fp8/int8 | 进一步省显存换少量精度 |

**时间与空间，工程永远在两头倒腾**——这是本系列的收束点。

**测量素养备忘**：
- 单次运行、无多轮取中位数时，若效应量 ≈ 噪声量（实验 10 的 +2ms vs ±10ms 步间波动），结论要谨慎——10b 靠拉长序列让效应显形；
- 想在 3 秒内生成 100 token → TPOT 需 <30ms（用 `总耗时 ≈ TTFT + N×TPOT` 做容量规划）。

<details>
<summary>📄 完整代码（实验 10b 最终时刻，点击展开）</summary>

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


def decode_loop(use_cache: bool, verbose: bool):
    """手写生成循环。use_cache=True：每步只喂 1 个新 token，历史由 KV Cache 携带；
    use_cache=False：每步重喂完整序列，前文全部重算。"""
    step_inputs = inputs   # 有缓存：step0 喂整个 prompt（Prefill），之后 1 个（Decode）
    full_seq = inputs      # 无缓存：每步喂 prompt + 已生成的全部 token
    past_kv = None
    total_ms = 0.0
    steps_done = 0
    with torch.inference_mode():
        for step in range(512):
            t = time.perf_counter()
            if use_cache:
                out = model(step_inputs, past_key_values=past_kv, use_cache=True)
            else:
                out = model(full_seq, use_cache=False)
            torch.cuda.synchronize()  # CUDA 是异步的：等 GPU 真正算完再停表
            dt_ms = (time.perf_counter() - t) * 1000
            total_ms += dt_ms

            next_id = int(out.logits[0, -1].argmax())   # 贪心：只看最后一个位置
            piece = tokenizer.decode(next_id)
            n_in = step_inputs.shape[-1] if use_cache else full_seq.shape[-1]
            if verbose or step % 16 == 0:
                print(
                    f"step {step:3d} | input {n_in:3d} tok"
                    f" | {dt_ms:6.1f} ms | {piece!r}",
                    flush=True,
                )

            if next_id in stop_ids:
                print("-> 命中停止符，循环结束（'纪律'由我们自己执行）")
                break
            if use_cache:
                past_kv = out.past_key_values        # K/V 存入缓存，交给下一步
                step_inputs = inputs.new_tensor([[next_id]])  # 之后每步只喂 1 个新 token
            full_seq = torch.cat([full_seq, inputs.new_tensor([[next_id]])], dim=1)
            steps_done = step + 1
    return total_ms, steps_done


t_cache, n_cache = decode_loop(use_cache=True, verbose=True)
t_nocache, n_nocache = decode_loop(use_cache=False, verbose=False)

print("\n===== KV Cache benchmark =====")
print(f"with cache   : {n_cache} tokens | {t_cache/1000:6.2f} s | {t_cache/max(n_cache,1):5.1f} ms/token")
print(f"without cache: {n_nocache} tokens | {t_nocache/1000:6.2f} s | {t_nocache/max(n_nocache,1):5.1f} ms/token")
print(f"slowdown     : x{t_nocache/t_cache:.2f}")

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
