# AI Infra 自学实验记录：亲手拆解 LLM 推理的十个实验

> 从一个最简单的 Qwen 推理脚本出发，通过 **10 个"每次只改一处"的受控实验**，亲手验证 LLM 推理的核心机制：
> Chat 模板、特殊 Token、停止纪律、采样三件套、随机种子、Prefill / Decode / KV Cache、TTFT / TPOT，以及浮点运算的"蝴蝶效应"。
>
> 所有数据均为 RTX 4090 单卡实测，实验过程逐 commit 记录（见文末 [Git 实验地图](#git-实验地图)）。

---

## 目录

- [环境与素材](#环境与素材)
- [仓库结构](#仓库结构)
- [快速开始](#快速开始)
- [实验总览](#实验总览)
- [实验详细记录](#实验详细记录)
- [核心概念速查表](#核心概念速查表)
- [实测数据汇总](#实测数据汇总)
- [诚实的测量说明（Limitations）](#诚实的测量说明limitations)
- [Git 实验地图](#git-实验地图)

---

## 环境与素材

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| 模型 | Qwen2.5-1.5B-Instruct（本地权重，bfloat16，约占 2.88 GiB 显存） |
| PyTorch | 2.5.1 + CUDA 12.1 |
| 运行环境 | Docker 容器（Linux, Python 3.10） |

## 仓库结构

```
.
├── README.md                  # 本文档：完整实验记录
├── .gitignore                 # 排除 models/（3GB 权重不入库）与 __pycache__/
├── models/                    # 本地模型权重（git 忽略，需自行下载）
│   └── Qwen2.5-1.5B-Instruct/
└── scripts/
    └── first_inference.py     # 实验主脚本（含全部实验的最终形态）
```

## 快速开始

```bash
# 默认：采样生成（temperature=0.7, top_k=20, top_p=0.8）+ 手写解码循环 A/B 基准
python3 scripts/first_inference.py

# 自定义采样参数：<temperature> <top_k> <top_p> [seed]
python3 scripts/first_inference.py 0.1              # 低温：多次运行结果几乎一致
python3 scripts/first_inference.py 1.5              # 高温：每次差异巨大
python3 scripts/first_inference.py 1.5 0 1.0        # 高温+拆掉 top_k/top_p 保险丝：语无伦次
python3 scripts/first_inference.py 1.5 0 1.0 42     # 再固定种子：混乱也可精确复现
```

脚本的 `generate()`（对照组）之后，会再跑一遍**手写的 token-by-token 解码循环**，
分别以 `use_cache=True / False` 各生成一遍并输出 KV Cache 基准报告。

---

## 实验总览

| # | 主题 | 一句话结论 | Commit |
|---|------|-----------|--------|
| 0 | 基线脚本 | 最朴素的 transformers 推理 | `ad74f03` |
| 1 | 打印 Chat 模板 | 脚本只写了 user 消息，模型却收到了一条没写过的 system prompt | `ad74f03` |
| 2 | max_new_tokens 256→32 | 截断的回答结尾**没有** `<\|im_end\|>`：截断看上限，停止看 eos | `ad74f03` |
| 3 | 去掉输出切片 | generate 返回的是"输入+输出"整条序列，模型的本质是续写 | `ad74f03` |
| 4 | eos_token_id=None | 停止不是模型的能力，是推理引擎的纪律；模型开始自问自答 | `d33fcba` |
| 5 | do_sample=True | 贪心=确定，采样=掷骰子；连续 4 次运行 4 种结果 | `5cb9510` |
| 6 | temperature 旋钮 | 0.1≈呆板可复现，1.5=放飞自我；温度重塑骰子的偏心程度 | `2e5e15e` |
| 7 | top_k / top_p 保险丝 | 拆掉后高温采样直接语无伦次；保险丝限定候选池 | `5eb3c37` |
| 8 | 固定随机种子 | 同种子+同参数=逐字一致，连"混乱"都可复现 | `522b78f` |
| 9 | 手写解码循环 | 亲眼看到 Prefill（input 40+ tok）与 Decode（input 1 tok）的形状差异 | `7d15555` |
| 10 | use_cache 基准（128 步） | 无缓存只慢 x1.09——小模型处于开销主导区间 | `e8f7197` |
| 10b | use_cache 基准（512 步） | x1.34 显形；并当场捕获浮点蝴蝶效应（428≠434） | `42f09a1` |

---

## 实验详细记录

### 实验 1：模型到底收到了什么？—— Chat 模板揭秘

**改动**：生成前把 `tokenizer.decode(inputs[0])` 打印出来。

**观察**：脚本里只有一条 user 消息，模型实际收到的却是：

```
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
请解释大语言模型推理中的 Prefill、Decode 和 KV Cache。<|im_end|>
<|im_start|>assistant
```

**结论**：
- Qwen2.5 的 chat 模板在**没有 system 消息时自动注入默认 system prompt**；
- 输入以 `<|im_start|>assistant\n` 结尾（`add_generation_prompt=True` 的作用），这就是"请开始回答"的信号；
- 所谓"多轮对话"，本质是用特殊 token 拼出来的一段连续文本。

### 实验 2：max_new_tokens=32 —— 截断 vs 自然停止

**改动**：`max_new_tokens` 从 256 改为 32。

**观察**：回答话说一半被拦腰截断，结尾**没有** `<|im_end|>`；而 256 上限时回答完整、结尾有 `<|im_end|>`。

**结论**：结合 `skip_special_tokens=False`（本组的前置改动），**看一眼结尾有没有 `<|im_end|>` 就能判断回答是自然结束还是被截断**——生产环境排查"回答突然断了"的常用手段。

> 截断看上限，停止看 eos。

### 实验 3：去掉切片 —— generate 到底返回了什么

**改动**：`outputs[0][inputs.shape[-1]:]` → `outputs[0]`（不切掉输入部分）。

**观察**：输出 = 完整 chat 模板 + 回答正文，且 `<|im_start|>assistant` 与回答正文直接相连，中间只有一个换行。

**结论**：
- **模型眼里没有"输入/输出"，只有一条文本流**，它做的唯一一件事是"给定前文，续写下一个 token"；
- generate 返回 prompt+新增的完整序列，切片只是人类展示的后处理；
- 一句话：**续写是模型的本质，切片是产品的礼貌。**

### 实验 4：eos_token_id=None —— 谁在让模型"闭嘴"

**改动**：`generate(..., eos_token_id=None)`（同时把 max_new_tokens 恢复到 128 以便观察）。

**观察**：模型说完一段完整回答后停不下来，输出大致为：

```
...处理大规模的文本任务。<|im_end|>
<|im_start|>user
（模型自己编造的新问题）<|im_end|>
<|im_start|>assistant
（模型自己回答自己）...
```

**结论**：
- `<|im_end|>` 对模型只是**一个普通的词**，它"知道"该停是训练出来的模式；
- 真正的停止是 generate 每生成一个 token 做一次的**外部检查**："这是不是 eos？是就停"；
- 关掉检查后，模型只能按最可能模式续写——而训练数据里最可能的模式就是"用户又问了新问题"；
- **"停止"不是模型的能力，是推理引擎的纪律。** vLLM 等框架的 `stop_token_ids` / `ignore_eos` 参数管的都是这件事。

### 实验 5：do_sample=True —— 从确定到掷骰子

**改动**：`do_sample=False` → `True`（并移除实验 4 的 eos 干扰）。

**观察**：连续运行 4 次，4 次回答各不相同；开头往往相似，越往后分歧越大；同时顶部三条 `temperature/top_p/top_k only used in sample-based generation` 警告消失——Qwen 预置的采样参数（0.7/0.8/20）开始真正生效。

**结论**：贪心=永远选概率最大（argmax，确定但呆板）；采样=按概率分布掷骰子（多样但不确定）。

### 实验 6：temperature —— 骰子的偏心旋钮

**改动**：temperature 变为命令行参数（`sys.argv[1]`，默认 0.7）。

**观察**：
- `0.1`：多次运行几乎一字不差（趋近贪心），语言呆板、套话；
- `1.5`：每次差异巨大，用词怪异、可能中英夹杂。

**结论**：`概率 = softmax(logits / T)`。T<1 分布变尖（笃定），T>1 分布压平（冒险），T→0 退化为贪心。
业界惯例：代码/数学/事实问答用低温（0~0.3），创意写作用高温（0.8~1.2）。

### 实验 7：top_k / top_p —— 两道保险丝

**改动**：top_k、top_p 也变成命令行参数（`sys.argv[2]`/`[3]`；0 与 1.0 分别表示"不限制"）。

**观察**（同样 temperature=1.5 下对比）：
- `1.5 0 1.0`（拆保险丝）：从约 15 万 token 的全词表里摇骰子，输出语无伦次、话说到一半跑题、语言混杂；
- `1.5`（保险丝在）：放飞但仍然"像人话"。

**结论**：
- `top_k=20`：只保留概率最高的 20 个候选再摇骰子（海选留前 20 名）；
- `top_p=0.8`：保留累计概率达 80% 的候选，数量随分布自适应（聪明的按比例截断）；
- 高温把分布压平后，"本来绝无可能"的冷门 token 也会中奖——保险丝挡的就是这个。

### 实验 8：固定种子 —— 混乱也可以复现

**改动**：新增可选 `seed` 参数（`sys.argv[4]`），传入则 `torch.manual_seed(seed)`。

**观察**：`1.5 0 1.0 42` 连跑两遍输出**逐字一致**（最狂野参数下的混乱被完美复现）；换成 43 则完全不同。

**结论**：
- 采样用的"随机数"是**伪随机**：由 RNG 从初始状态按固定算法推演，固定种子=固定数列；
- 工程用途：可复现的 bug 报告（记录 prompt+参数+seed+模型版本即可精确重放）、Prompt 回归测试、论文复现；
- 精确化：**同环境=数学保证的确定（100%），跨环境=碰运气**——不同硬件/算子的浮点差异可能翻转打平的候选，导致整条轨迹分叉（见实验 10b 的实拍）。

### 实验 9：手写解码循环 —— 拆开 generate 的黑盒

**改动**：不用 `model.generate`，自己写 for 循环逐 token 生成（贪心 argmax），每步打印输入 token 数、耗时、新 token；停止检查也自己写。

核心代码（约 20 行）：

```python
step_inputs = inputs                 # 第一次喂整个 prompt（Prefill）
past_kv = None                       # KV Cache 容器
for step in range(512):
    out = model(step_inputs, past_key_values=past_kv, use_cache=True)
    torch.cuda.synchronize()         # CUDA 异步，等 GPU 算完再停表
    next_id = int(out.logits[0, -1].argmax())   # 只看最后一个位置
    if next_id in stop_ids: break               # 停止纪律由自己执行
    past_kv = out.past_key_values               # K/V 存入缓存
    step_inputs = inputs.new_tensor([[next_id]])  # 之后每步只喂 1 个新 token
```

**观察**：
- `step 0 | input 40+ tok`（Prefill：整个 prompt 一次并行算完）；
- `step 1+ | input 1 tok`（Decode：每步只喂 1 个新 token，历史全靠 KV Cache 携带）；
- 每步约 17~21 ms，串行等待——第 k 步的输入依赖第 k-1 步的输出，无法并行；
- 手写贪心循环的文本与 `generate(do_sample=False)` 一致：generate 的内核就是这个循环。

**结论**：
- **Prefill = 一次并行读完整个考题，写下答案第一个字**（一步大计算，算力型）；
- **Decode = 一个字一个字往下写**（N 次小计算，访存型）；
- 模型从头到尾没"收到"过自己的完整回答——每步只见上一枚 token + 缓存中的全部历史。

### 实验 10 / 10b：use_cache 基准 —— KV Cache 的价值实测

**改动**：解码循环封装为 `decode_loop(use_cache, verbose)`，有/无缓存各跑一遍并输出基准报告。无缓存模式每步把 `prompt+已生成全部 token` 重新喂入整段重算（`full_seq` 逐增长）。10b 把循环上限从 128 提到 512。

**实测数据**：

| 配置 | tokens | 总耗时 | ms/token | slowdown |
|---|---|---|---|---|
| 128 步 · 有缓存 | 128 | 2.73 s | 21.3 | — |
| 128 步 · 无缓存 | 128 | 2.98 s | 23.3 | **x1.09** |
| 512 步上限 · 有缓存 | 428 | 8.66 s | 20.2 | — |
| 512 步上限 · 无缓存 | 434 | 11.62 s | 26.8 | **x1.34** |

**观察**：
- 无缓存模式 `input` 列从 40 涨到 168（128 步）/ 直到 470+（10b），单步耗时随之爬升；缓存版从头到尾持平；
- 缓存版 ms/token 对序列长度**完全免疫**（21.3→20.2，噪声范围内），无缓存版持续上涨；
- 每生成 1 个 token 的额外代价从 +2.0 ms 涨到 +6.6 ms（3.3 倍）；
- **蝴蝶效应实拍**：两遍都是贪心解码，理论上应走出相同轨迹、同时停止——实际停止步数 428 ≠ 434。缓存增量计算与整段重算走不同 kernel 路径 → 浮点求和顺序不同 → logits 小数点后几位的差异 → 某个概率打平点翻转 → 轨迹分叉。

**结论**：
- KV Cache 把 O(n²) 压成 O(n)：无缓存时第 n 步多算 (40+n) 个 token，总代价 ∝ n²；
- 本例 x1.09/x1.34 温和的原因：1.5B 模型太小，单步时间被 kernel 发射开销与权重读取主导，4090 算力大量闲置；换大模型/长序列，无缓存从"慢一点"变成"物理不可行"；
- 缓存是**纯加速而非近似**：与整段重算在数学上等价（428≠434 只是浮点副作用）；
- 代价是**显存**：缓存每步增长一截，长上下文时可能超过模型权重本身——GQA（Qwen2.5 在用）、PagedAttention（vLLM）都是给缓存瘦身的技术。

---

## 核心概念速查表

| 概念 | 一句话 |
|---|---|
| Chat 模板 | 把消息列表拼成带特殊 token 的单段文本，是"对话"的真实形态 |
| 特殊 token | `<\|im_start\|>` / `<\|im_end\|>` 等，模型的"格式控制符" |
| 停止纪律 | eos 检查由推理引擎执行，模型只是"倾向于"在 im_end 处收尾 |
| 贪心 vs 采样 | argmax（确定呆板）vs 按分布掷骰子（多样不确定） |
| temperature | softmax(logits/T)：调骰子偏心程度，T→0 退化为贪心 |
| top_k / top_p | 限定骰子候选池：固定前 K 个 / 累计概率 p 的自适应集合 |
| 种子 | 固定 RNG 初始状态=固定"随机"数列；同环境 100% 可复现 |
| Prefill | 整段 prompt 一次并行处理，填满 KV 缓存，产出第 1 个 token（算力型） |
| Decode | 每步进 1 出 1 的串行续写，历史以缓存形式参与（访存型） |
| KV Cache | 存下已算好的 K/V，历史只读不重算；省时间、吃显存 |
| TTFT / TPOT | 首 token 延迟（由 Prefill 决定）/ 单 token 生成时间（由 Decode 决定）；总耗时 ≈ TTFT + N × TPOT |
| 蝴蝶效应 | 浮点差异在概率打平点翻转 token 选择，后续轨迹整体分叉 |
| 自回归 | 每步生成的 token 转身成为下一步的唯一输入，环环相扣无法并行 |

**验证 TTFT/TPOT 公式**：428 token × 20.2 ms/token ≈ 8.65 s，实测 8.66 s ✓

---

## 实测数据汇总

- 模型显存占用（bfloat16）：**2.88 GiB**
- 贪心生成 256 token：回答完整，结尾 `<\|im_end\|>` 自然出现
- 贪心生成 32 token：截断，无 `<\|im_end\|>`
- 采样 4 连跑（temp=0.7）：4 种不同回答
- 采样 temp=0.1 多连跑：几乎逐字一致
- 采样 temp=1.5 + top_k=0 + top_p=1.0：语无伦次；固定 seed=42 后逐字复现
- 手写 Decode 单步：约 17~21 ms/token（含同步开销）
- KV Cache 基准：见实验 10/10b 表格（x1.09 → x1.34）

---

## 诚实的测量说明（Limitations）

1. **单次运行，无多轮取中位数**：实验 10 中步间噪声（±10 ms）大于缓存效应（+2 ms），x1.09 处于"效应≈噪声"区间；10b 通过拉长序列让效应显形（x1.34）；
2. **开销主导区间**：1.5B 模型 + 4090 的组合下，单步时间由 kernel 发射与权重读取主导，计算增量几乎免费——KV Cache 的真实价值在大模型/长序列下才完全展开；
3. **butterfly 观察（428≠434）为单次实拍**：分叉是否发生取决于是否撞上概率打平点，属概率事件，重复实验不一定每次复现；
4. 计时含 `torch.cuda.synchronize()` 同步开销，绝对值偏保守，相对比较有效。

---

## Git 实验地图

```text
42f09a1  实验 10b：解码循环 128→512 步，二次增长显形（x1.34，捕获蝴蝶效应）
e8f7197  实验 10：use_cache 开/关基准（重算 vs KV Cache，x1.09）
7d15555  实验 9：手写 token-by-token 解码循环（Prefill + KV Cache + Decode）
522b78f  实验 8：可选 seed 参数（torch.manual_seed 可复现性）
5eb3c37  实验 7：top_k/top_p 命令行参数（0/1.0 = 拆保险丝）
2e5e15e  实验 6：temperature 命令行参数（默认 0.7）
5cb9510  实验 5：do_sample=True（采样，非确定）
d33fcba  实验 4：eos_token_id=None（模型无法停止）
ad74f03  基线 + 实验 1~3（打印 chat 模板 / max_new_tokens=32 / 不切片）
```

> 方法论：**一次只改一处 → 预测输出变化 → 实际运行验证 → 概念闭环**。
> 每个 commit 即一次可回滚、可 diff 的受控实验。
