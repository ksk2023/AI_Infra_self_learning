# AI Infra 自学实验记录：亲手拆解 LLM 推理的十个实验

> 从一个最简单的 Qwen 推理脚本出发，通过 **10 个"每次只改一处"的受控实验**，亲手验证 LLM 推理的核心机制：
> Chat 模板、特殊 Token、停止纪律、采样三件套、随机种子、Prefill / Decode / KV Cache、TTFT / TPOT，以及浮点运算的"蝴蝶效应"。
>
> 所有数据均为 RTX 4090 单卡实测。每个实验的**完整代码、核心代码、细节、结果与推论**见 [`experiments/`](./experiments) 目录；本页是总览与改动一览。

---

## 目录

- [环境与素材](#环境与素材)
- [仓库结构](#仓库结构)
- [快速开始](#快速开始)
- [实验改动一览](#实验改动一览)
- [核心概念速查表](#核心概念速查表)（详细版：[experiments/concepts-guide.md](./experiments/concepts-guide.md)）
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
├── README.md                     # 本页：总览 + 改动一览
├── experiments/                  # ★ 每个实验的详细记录
│   ├── concepts-guide.md         # ★ 核心概念详解（是什么→怎么工作→实验证据→工程实践）
│   ├── experiment-01.md          # Chat 模板揭秘
│   ├── experiment-02.md          # 截断 vs 自然停止
│   ├── experiment-03.md          # generate 返回完整序列
│   ├── experiment-04.md          # 停止纪律（eos_token_id=None）
│   ├── experiment-05.md          # 采样 vs 贪心
│   ├── experiment-06.md          # temperature
│   ├── experiment-07.md          # top_k / top_p
│   ├── experiment-08.md          # 固定随机种子
│   ├── experiment-09.md          # 手写解码循环（Prefill/Decode/KV Cache/TTFT/TPOT）
│   └── experiment-10.md          # use_cache 基准（KV Cache 价值实测）
├── .gitignore                    # 排除 models/ 与 __pycache__/
├── models/                       # 本地模型权重（git 忽略，需自行下载）
│   └── Qwen2.5-1.5B-Instruct/
└── scripts/
    └── first_inference.py        # 实验主脚本（全部实验的最终形态）
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

---

## 实验改动一览

> 方法论：**一次只改一处 → 预测输出变化 → 实际运行验证 → 概念闭环**。每个 commit 即一次可回滚、可 diff 的受控实验。
> 点击实验名查看完整代码、核心代码与详细结论。

| # | 实验 | 改动（本处 diff） | 一句话结论 | Commit |
|---|------|------------------|-----------|--------|
| 前置 | — | `skip_special_tokens=True → False` | 保留特殊 token，后续观察的基础 | — |
| 1 | [Chat 模板揭秘](./experiments/experiment-01.md) | `+print(tokenizer.decode(inputs[0]))` ×3 行 | 脚本只写了 user 消息，模型却收到了没写过的 system prompt | `ad74f03` |
| 2 | [截断 vs 自然停止](./experiments/experiment-02.md) | `max_new_tokens: 256 → 32` | 截断看上限，停止看 eos（结尾有无 `<\|im_end\|>` 一眼可辨） | `ad74f03` |
| 3 | [generate 返回完整序列](./experiments/experiment-03.md) | `outputs[0][inputs.shape[-1]:] → outputs[0]` | 模型没有输入/输出之分，只有续写；切片是产品的礼貌 | `ad74f03` |
| 4 | [停止纪律](./experiments/experiment-04.md) | `+eos_token_id=None` | "停止"不是模型的能力，是推理引擎的纪律；关掉后模型自问自答 | `d33fcba` |
| 5 | [采样 vs 贪心](./experiments/experiment-05.md) | `do_sample: False → True` | 贪心=argmax（确定呆板），采样=掷骰子（多样不确定）；4 连跑 4 种结果 | `5cb9510` |
| 6 | [temperature](./experiments/experiment-06.md) | `+temperature=CLI参数` | softmax(logits/T)：0.1≈呆板可复现，1.5=放飞自我 | `2e5e15e` |
| 7 | [top_k / top_p](./experiments/experiment-07.md) | `+top_k/top_p=CLI参数` | 两道保险丝限定候选池；拆掉后高温采样语无伦次 | `5eb3c37` |
| 8 | [固定随机种子](./experiments/experiment-08.md) | `+torch.manual_seed(seed)` | 伪随机=纯数学：同种子逐字复现，连混乱都可以 | `522b78f` |
| 9 | [手写解码循环](./experiments/experiment-09.md) | `+约25行 for 循环替代 generate` | 亲眼看 Prefill（input 40+ tok）与 Decode（input 1 tok）；引出 TTFT/TPOT | `7d15555` |
| 10 | [use_cache 基准](./experiments/experiment-10.md) | `+decode_loop(use_cache)` 双跑计时 | 无缓存只慢 x1.09——小模型处于开销主导区间 | `e8f7197` |
| 10b | （并入实验 10 文档） | `range(128) → range(512)` | x1.34 显形，O(n²) 露头；并当场捕获蝴蝶效应（428≠434） | `42f09a1` |

---

## 核心概念速查表

> 每个概念的**详细解释（是什么 → 怎么工作 → 实验证据 → 工程实践）**见 **[概念详解](./experiments/concepts-guide.md)**，下表是快速索引。

| 概念 | 一句话 | 详解 |
|---|---|---|
| Chat 模板 | 把消息列表拼成带特殊 token 的单段文本，是"对话"的真实形态 | [链接](./experiments/concepts-guide.md#11-chat-模板) |
| 特殊 token | `<\|im_start\|>` / `<\|im_end\|>` 等，模型的"格式控制符" | [链接](./experiments/concepts-guide.md#12-特殊-token) |
| 自回归 | 每步生成的 token 转身成为下一步的唯一输入，环环相扣无法并行 | [链接](./experiments/concepts-guide.md#21-自回归autoregression) |
| 停止纪律 | eos 检查由推理引擎执行，模型只是"倾向于"在 im_end 处收尾 | [链接](./experiments/concepts-guide.md#22-停止纪律eos-discipline) |
| 贪心 vs 采样 | argmax（确定呆板）vs 按分布掷骰子（多样不确定） | [链接](./experiments/concepts-guide.md#31-贪心-vs-采样) |
| temperature | softmax(logits/T)：调骰子偏心程度，T→0 退化为贪心 | [链接](./experiments/concepts-guide.md#32-temperature) |
| top_k / top_p | 限定骰子候选池：固定前 K 个 / 累计概率 p 的自适应集合 | [链接](./experiments/concepts-guide.md#33-top-k--top-p) |
| 种子 | 固定 RNG 初始状态=固定"随机"数列；同环境 100% 可复现 | [链接](./experiments/concepts-guide.md#34-随机种子与伪随机) |
| Prefill | 整段 prompt 一次并行处理，填满 KV 缓存，产出第 1 个 token（算力型） | [链接](./experiments/concepts-guide.md#41-prefill) |
| Decode | 每步进 1 出 1 的串行续写，历史以缓存形式参与（访存型） | [链接](./experiments/concepts-guide.md#42-decode) |
| KV Cache | 存下已算好的 K/V，历史只读不重算；省时间、吃显存 | [链接](./experiments/concepts-guide.md#43-kv-cache) |
| TTFT / TPOT | 首 token 延迟（由 Prefill 决定）/ 单 token 生成时间（由 Decode 决定）；总耗时 ≈ TTFT + N × TPOT | [链接](./experiments/concepts-guide.md#44-ttft-与-tpot) |
| 蝴蝶效应 | 浮点差异在概率打平点翻转 token 选择，后续轨迹整体分叉 | [链接](./experiments/concepts-guide.md#51-浮点蝴蝶效应) |

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
- KV Cache 基准：128 步 x1.09 → 512 步 x1.34（详见[实验 10](./experiments/experiment-10.md)）

---

## 诚实的测量说明（Limitations）

1. **单次运行，无多轮取中位数**：实验 10 中步间噪声（±10 ms）大于缓存效应（+2 ms），x1.09 处于"效应≈噪声"区间；10b 通过拉长序列让效应显形（x1.34）；
2. **开销主导区间**：1.5B 模型 + 4090 的组合下，单步时间由 kernel 发射与权重读取主导，计算增量几乎免费——KV Cache 的真实价值在大模型/长序列下才完全展开；
3. **butterfly 观察（428≠434）为单次实拍**：分叉是否发生取决于是否撞上概率打平点，属概率事件，重复实验不一定每次复现；
4. 计时含 `torch.cuda.synchronize()` 同步开销，绝对值偏保守，相对比较有效。

---

## Git 实验地图

```text
fc19bc8  Add learning README（总览 + experiments/ 详细文档）
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
