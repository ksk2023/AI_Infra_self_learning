# 实验 11：固定工作量的 KV Cache A/B 基准

> **改动**：新增独立脚本 `scripts/exp11_kv_cache_ab.py`（不改动主脚本）。
> **核心问题**：相同输入、相同模型、相同生成长度下，开启 KV Cache 的耗时是否**稳定**低于关闭缓存？
> **实测答案：否 —— speedup 0.998x，统计上无差异**；同时修正了实验 10/10b 的测量偏差。
> **Commit**：`0229e43`（脚本）、本文档另行提交

## 实验设计

| 项目 | 设置 |
|---|---|
| 模型 / 精度 | 本地 Qwen2.5-1.5B-Instruct，BF16 |
| Batch size | 1 |
| 输入 | 同一条 Prompt（44 token） |
| 输出长度 | 两组固定 128 个新 token（`min_new_tokens=128` + `max_new_tokens=128`，杜绝提前 EOS 导致工作量不等） |
| 推理入口 | 两组都用 `model.generate()` |
| 对比变量 | `use_cache=True / False` |
| 预热 | 每组 2 次 |
| 正式测量 | 每组 6 次，交替运行顺序 |
| 统计 | 中位数、最小/最大值、输出吞吐率、峰值 allocated 显存 |

**相对实验 10 的五个方法学升级：**

| 升级 | 解决的问题 |
|---|---|
| 固定工作量（min=max=128）+ assert 自检 | 实验 10 两组各跑到自然停止（428 vs 434 步），工作量不相等；现在**必然恰好 128 步**，且代码验证 |
| 预热 ×2 | 排除首次 CUDA 调用的懒初始化等一次性开销 |
| 交替顺序（True,False / False,True 轮换） | 消除 GPU 频率漂移、温度爬升等时序偏差 |
| 6 轮取中位数 | 实验 10 的教训：单次测量下噪声（±10ms）大于效应（+2ms） |
| 计时前后 `synchronize()` + `reset_peak_memory_stats()` | 只对整次生成计时/计存，不逐步打断流水线 |

## 运行方式

```bash
cd /home/dev/infra
CUDA_VISIBLE_DEVICES=3 python3 scripts/exp11_kv_cache_ab.py
```

## 实验结果（实测全量数据）

```
round=1 cache=True  time=2.3967s tps=53.41 peak=2.894GiB | round=1 cache=False time=2.4005s tps=53.32 peak=2.896GiB
round=2 cache=True  time=2.3552s tps=54.35 peak=2.894GiB | round=2 cache=False time=2.3830s tps=53.71 peak=2.896GiB
round=3 cache=True  time=2.3587s tps=54.27 peak=2.894GiB | round=3 cache=False time=2.4972s tps=51.26 peak=2.896GiB
round=4 cache=True  time=2.5850s tps=49.52 peak=2.894GiB | round=4 cache=False time=2.3665s tps=54.09 peak=2.896GiB
round=5 cache=True  time=2.4859s tps=51.49 peak=2.894GiB | round=5 cache=False time=2.3708s tps=53.99 peak=2.896GiB
round=6 cache=True  time=2.3969s tps=53.40 peak=2.894GiB | round=6 cache=False time=2.4342s tps=52.58 peak=2.896GiB

===== Summary =====
cache=True  median=2.3968s range=[2.3552, 2.5850]s output_tps=53.40
cache=False median=2.3918s range=[2.3665, 2.4972]s output_tps=53.52
Cache speedup: 0.998x
```

**逐轮配对**（同轮比大小）：True 更快 4 轮（+3.8 / +27.8 / +138.5 / +37.3 ms），False 更快 2 轮（+218.5 / +115.1 ms）—— 胜负互现、幅度互现；两组 range 完全交叠，中位数差 5ms（0.2%）。

## 分析与结论

### 核心答案：否

在本工作负载（1.5B 模型 / batch 1 / 172 token 以内序列 / 4090）下，**开关 KV Cache 的耗时统计上无差异（0.998x），也谈不上"稳定更低"** —— 这是对实验开篇问题的直接否定回答。

### 为什么与实验 10（x1.09）/ 10b（x1.34）冲突 —— 诚实的修正

三个层面的原因：

1. **实验 10 是单次测量**：x1.09 落在当时的噪声带内（Limitations 里我们自己预警过"效应≈噪声"），那个数字本来就不该当作结论；
2. **手工循环的测量结构含有三项额外开销**，且它们对不同模式的惩罚不等：
   - 每步 `torch.cuda.synchronize()` —— 打断 CPU 发射 / GPU 执行的流水线，重算的计算无法被异步隐藏；
   - 每步 `tokenizer.decode()` + `print()` —— 逐步 CPU 工作夹在 GPU 步之间不被重叠；
   - 无缓存路径每步物化**完整 logits** `out.logits[1, seq, 151936]`（最大 ~50MB 的 LM Head 计算与分配）。
   量化证据：同为有缓存 128 步，`generate` 18.7ms/token vs 手工循环 21.3ms/token —— 手工结构自带约 2.6ms/token 的开销；
3. **`generate()` 有一个隐藏优化**（本实验实测溯源，`transformers 4.46.3 generation/utils.py:2065`）：自动设置 `num_logits_to_keep=1` —— **即使 `use_cache=False`，LM Head 也只算最后一个位置**。这同时解释了两个现象：无缓存峰值显存只多 2MB（而不是 ~50MB）；无缓存每步的重算负担比手工循环轻一大截。
   实验 10b 的 x1.34：在 ~430 步长序列下方向可能真实（重算量随长度增长），但幅度被上述测量结构放大，应视为上界而非准确值。

### 原理：batch-1 小模型 Decode 是访存（memory-bound）主导

- 每一步无论开不开缓存，都要**完整搬运一次约 3GB 的模型权重**——GEMM 按权重矩阵读数据，172 个 token 的矩阵乘和 1 个 token 读的是**同一份权重**，访存量几乎不变；
- 重算多出来的 FLOPs 落在 4090 **大量闲置的算力**上，藏在访存等待的"影子"里 → 近似免费；
- 显存端同理：KV Cache 约 5MB 量级 vs 权重 2.88GiB（实测峰值 2.894 vs 2.896 GiB）—— 时间和空间两端在这个规模下都看不见。

### 什么条件下答案会反转（KV Cache 何时拉开差距）

| 条件 | 机制 |
|---|---|
| 序列更长（千 token 级） | 每步重算量线性增长 + attention O(L²)；实验 10b 在 ~430 步时方向已显现 |
| 模型更大 | 算力相对 GPU 饱和，重算无法再藏进闲置算力 |
| batch / 并发更高 | 多请求拼 batch 后算力被打满 |
| 显存受限 | KV Cache 与重算的空间权衡才实际生效（GQA/PagedAttention 的用武之地） |

## 测量素养收尾

本实验用方法学正确的基准**推翻了自己仓库里前两个实验的数字** —— 这不是失败，恰恰是实验 10 Limitations 中预警的兑现：*单次测量、含杂质的测量结构只能给方向性参考；只有固定工作量、预热、交替顺序、多次测量、中位数统计，才配得上"结论"二字。*

## Limitations

1. 128 token 仍偏短（172 token 以内的序列重算代价太小）；batch=1；结论严格限定于本工作负载；
2. 6 个样本未做统计显著性检验（样本量下 t 检验意义有限），以逐轮配对 + range 交叠作定性判断；
3. GPU3 为共享卡（运行时有其他进程占用 ~4.6GB），背景负载可能增大方差，但交替顺序保证其对两组公平。

## 完整代码

见 [`scripts/exp11_kv_cache_ab.py`](../scripts/exp11_kv_cache_ab.py)（与运行版本完全一致）。

<details>
<summary>📄 点击展开完整脚本</summary>

```python
import time
import statistics

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/dev/infra/models/Qwen2.5-1.5B-Instruct"
NEW_TOKENS = 128
ROUNDS = 6

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to("cuda").eval()

messages = [
    {
        "role": "user",
        "content": "请解释大语言模型推理中的 Prefill、Decode 和 KV Cache。",
    }
]

batch = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda")

prompt_len = batch["input_ids"].shape[-1]

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("Input tokens:", prompt_len)
print("New tokens per run:", NEW_TOKENS)

# 两组使用相同生成配置，只切换 use_cache。
generation_args = dict(
    min_new_tokens=NEW_TOKENS,
    max_new_tokens=NEW_TOKENS,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.0,
    temperature=1.0,
    top_p=1.0,
    top_k=50,
    pad_token_id=tokenizer.pad_token_id,
    return_dict_in_generate=False,
)


@torch.inference_mode()
def run_once(use_cache):
    # 开始计时前，等此前的 GPU 工作完成。
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    output = model.generate(
        **batch,
        **generation_args,
        use_cache=use_cache,
    )

    # 等本次生成的 GPU 工作完成，再停止计时。
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    actual_new_tokens = output.shape[-1] - prompt_len
    assert actual_new_tokens == NEW_TOKENS, actual_new_tokens

    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    del output
    return elapsed, actual_new_tokens, peak_gib


print("\nWarming up...")
for flag in (True, False, False, True):
    run_once(flag)

times = {True: [], False: []}

print("\n===== Measurements =====")
for round_id in range(ROUNDS):
    # 交替顺序：先开后关、先关后开。
    order = (True, False) if round_id % 2 == 0 else (False, True)

    for flag in order:
        elapsed, count, peak = run_once(flag)
        times[flag].append(elapsed)

        print(
            f"round={round_id + 1} "
            f"cache={str(flag):5s} "
            f"tokens={count} "
            f"time={elapsed:.4f}s "
            f"output_tps={count / elapsed:.2f} "
            f"peak_allocated={peak:.3f}GiB"
        )

print("\n===== Summary =====")
for flag in (True, False):
    values = times[flag]
    median_s = statistics.median(values)

    print(
        f"cache={str(flag):5s} "
        f"median={median_s:.4f}s "
        f"range=[{min(values):.4f}, {max(values):.4f}]s "
        f"output_tps={NEW_TOKENS / median_s:.2f}"
    )

speedup = (
    statistics.median(times[False])
    / statistics.median(times[True])
)
print(f"Cache speedup: {speedup:.3f}x")
```

</details>

[← 返回总览](../README.md)
