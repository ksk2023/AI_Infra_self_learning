# Qwen3.6-35B-A3B @ 单卡 RTX 4090 解码优化日志

- 模型:`/home/dev/models/Qwen3.6-35B-A3B-GGUF-Q4K_S/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`(20.9G)
- 硬件:RTX 4090 24564 MiB,驱动 535.104.05(CUDA ≤12.2)
- 理论屋顶(roofline.py,短上下文 Q4_K_S + BF16 KV):**579 tok/s**
- 方法论(参考 T58 NPU 实战):单变量改动 → A/B 实测 → 数值/输出正确性检查 → 只保留有效项;每次记录当时 GPU 快照,忙卡数字只作参考

| 轮次 | 日期 | 卡 | GPU 快照(占用) |
|---|---|---|---|

## 验证跑(忙卡GPU1,仅链路验证)  (11-20 06:58:41, GPU1, 快照: 0, 21069 MiB, 99 %;1, 2293 MiB, 82 %;2, 17374 MiB, 0 %;3, 6792 MiB, 43 %;)
```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24217 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24217 MiB
error: invalid parameter for argument: -c
usage: /home/dev/tools/llama.cpp/build2/bin/llama-bench [options]

options:
  -h, --help
  --version                                   show version and build info
  --numa <distribute|isolate|numactl>         numa mode (default: disabled)
  -r, --repetitions <n>                       number of times to repeat each test (default: 5)
  --prio <-1|0|1|2|3>                         process/thread priority (default: 0)
  --delay <0...N> (seconds)                   delay between each test (default: 0)
  -o, --output <csv|json|jsonl|md|sql>        output format printed to stdout (default: md)
  -oe, --output-err <csv|json|jsonl|md|sql>   output format printed to stderr (default: none)
  --list-devices                              list available devices and exit
  -v, --verbose                               verbose output
  --progress                                  print test progress indicators
  --no-warmup                                 skip warmup runs before benchmarking
  -fitt, --fit-target <MiB>                   fit model to device memory with this margin per device in MiB (default: off)
  -fitc, --fit-ctx <n>                        minimum ctx size for --fit-target (default: 0)

test parameters:
  -m, --model <filename>                            (default: models/7B/ggml-model-q4_0.gguf)
  -hf, -hfr, --hf-repo <user>/<model>[:quant]       Hugging Face model repository; quant is optional, case-insensitive
                                                    default to Q4_K_M, or falls back to the first file in the repo if Q4_K_M doesn't exist.
                                                    example: ggml-org/GLM-4.7-Flash-GGUF:Q4_K_M
                                                    (default: unused)
  -hff, --hf-file <file>                            Hugging Face model file. If specified, it will override the quant in --hf-repo
                                                    (default: unused)
  -hft, --hf-token <token>                          Hugging Face access token
                                                    (default: value from HF_TOKEN environment variable)
  --offline                                         Offline mode: forces use of cache, prevents network access
                                                    (default: disabled)
  -p, --n-prompt <n>                                (default: 512)
  -n, --n-gen <n>                                   (default: 128)
  -pg <pp,tg>                                       (default: )
  -d, --n-depth <n>                                 (default: 0)
  -b, --batch-size <n>                              (default: 2048)
  -ub, --ubatch-size <n>                            (default: 512)
  -ctk, --cache-type-k <t>                          (default: f16)
  -ctv, --cache-type-v <t>                          (default: f16)
  -t, --threads <n>                                 (default: 32)
  -C, --cpu-mask <hex,hex>                          (default: 0x0)
  --cpu-strict <0|1>                                (default: 0)
  --poll <0...100>                                  (default: 50)
  -ngl, --n-gpu-layers <n>                          (default: -1)
  -ncmoe, --n-cpu-moe <n>                           (default: 0)
  -sm, --split-mode <none|layer|row|tensor>         (default: layer)
  -mg, --main-gpu <i>                               (default: 0)
  -nkvo, --no-kv-offload <0|1>                      (default: 0)
  -fa, --flash-attn <on|off|auto>                   (default: auto)
  -dev, --device <dev0/dev1/...>                    (default: auto)
  -lm, --load-mode <auto|none|mmap|mlock|mmap+mlock|dio> (default: auto)
  -lzm, --lazy-mode <on|auto|off>                   (default: auto)
  -embd, --embeddings <0|1>                         (default: 0)
  -ts, --tensor-split <ts0/ts1/..>                  (default: 0)
  -ot --override-tensor <tensor name pattern>=<buffer type>;...
                                                    (default: disabled)
  -nopo, --no-op-offload <0|1>                      (default: 0)
  --no-host <0|1>                                   (default: 0)
  --repack <0|1>                                    (default: 1)

Multiple values can be given for each parameter by separating them with ','
or by specifying the parameter multiple times. Ranges can be given as
'first-last' or 'first-last+step' or 'first-last*mult'.
```

## 验证跑(忙卡GPU1,仅链路验证)  (11-20 06:59:12, GPU1, 快照: 0, 21069 MiB, 98 %;1, 2293 MiB, 69 %;2, 17374 MiB, 53 %;3, 6792 MiB, 44 %;)
```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24217 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24217 MiB
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           pp512 |       6548.04 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           pp128 |       2582.19 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           tg256 |         76.27 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |            tg64 |         78.89 ± 0.00 |

build: unknown (0)
```

## A/B: fa off(忙卡GPU1)  (11-20 06:59:54, GPU1, 快照: 0, 21069 MiB, 99 %;1, 2293 MiB, 69 %;2, 17374 MiB, 99 %;3, 6792 MiB, 43 %;)
```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24217 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24217 MiB
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           pp512 |       8137.30 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           pp128 |       2045.80 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           tg256 |         77.19 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |            tg64 |         81.16 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   0 |           pp512 |       4760.11 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   0 |           pp128 |       4127.29 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   0 |           tg256 |         77.95 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   0 |            tg64 |         73.96 ± 0.00 |

build: unknown (0)
```

## A/B: KV q8_0(忙卡GPU1)  (11-20 07:00:30, GPU1, 快照: 0, 21069 MiB, 98 %;1, 2293 MiB, 81 %;2, 17374 MiB, 98 %;3, 6792 MiB, 77 %;)
```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24217 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24217 MiB
| model                          |       size |     params | backend    | ngl | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |   1 |           pp512 |       4649.75 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |   1 |           pp128 |       3591.25 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |   1 |           tg256 |         78.07 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |   1 |            tg64 |         76.72 ± 0.00 |

build: unknown (0)
```

## A/B: repack off(忙卡GPU1)  (11-20 07:00:42, GPU1, 快照: 0, 21069 MiB, 98 %;1, 2293 MiB, 100 %;2, 17374 MiB, 70 %;3, 6792 MiB, 54 %;)
```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24217 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24217 MiB
| model                          |       size |     params | backend    | ngl |  fa | rpk |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |   0 |           pp512 |       9487.62 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |   0 |           pp128 |       2141.30 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |   0 |           tg256 |         78.81 ± 0.00 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |   0 |            tg64 |         73.96 ± 0.00 |

build: unknown (0)
```

## 正确性验证(2026-11-20,GPU1 忙卡)
llama-cli -st --temp 0 中文提问 MoE 概念:输出连贯、内容正确(思考模式)、EXIT=0。
生成速度 69.2 t/s(n=100),与 llama-bench 忙卡数字一致 → 模型/量化/内核/模板全链路正常。

## 忙卡 A/B 初步结论(GPU1,队友占 80% 算力,±3% 噪声,tg64)
| 配置 | tok/s | 相对 |
|---|---|---|
| fa on(默认含 repack on) | 78.89 | 基准 |
| fa off | 73.96 | -6.2% |
| fa on + KV q8_0 | 76.72 | -2.8%(短上下文;长上下文待干净卡复测) |
| repack off(fa on) | 73.96 | -6.3% → repack 保留开启 |

结论:fa on + repack on 保留;其余待干净卡决策。

## 源码侦察与预注册预测(2026-11-21,GPU0 被占期间)
ggml-cuda v1.0.0 关键事实:
1. mmvq.cu 是解码 GEMV 主力(Q4_K,M≤8),内核参数(nwarps/rows_per_block)按 (类型,列数) 查表,上游已按架构调优
2. gated-delta-net-mma.cu 仅 n_tokens≥64(预填充)启用;解码走 gated_delta_net.cu 的 ar 原子归约路径
3. GGML_CUDA_MMV_X/DMMV_X 只影响 dequant+matvec 路径(mmvf.cu),不影响 mmvq
4. 已有图级融合:gated_delta_net 状态快照 cpy 融合;--repack 默认开(实测+6%)

**预注册预测(待 GPU0 空闲证伪/证实)**:
- E1 MMV_X=64 / MMV_X=16 变体:解码 tg 预期 ±0%(不影响 mmvq 路径);若 >±2% 则预测错误
- B KV q8_0 短上下文:预期 -2%~-4%(反量化开销 > KV 节省)
- C/D KV q8_0 @32K:预期 +5%~+15%(KV 读取减半效应显现)
- 干净卡基线 tg256 预期 85~95 tok/s(忙卡 76.3 + 去争用)

## 纯源码分析轮(2026-11-21,GPU0 被占,用户选定模式)
- 产出《解码算子分析报告》:operator-analysis.md
- 关键结论:上游已融合 MoE 路由/GLU/共享专家;CUDA Graphs 默认启用;PDL 仅 Hopper(4090 无缘);剩余空间=微参数+系统层
- 新变体 mmvqw8(GENERIC 表 ncols=1 nwarps 4→8)构建完成,补丁已还原,已加入矩阵队列
- 变体总数:mmvqw8 / mmvx64 / mmvx16,矩阵将依次实测

## 预注册预测修正(2026-11-21 源码二次核查)
- E-repack:GPU 全卸载下预期 ±0%(repack=CPU_REPACK 缓冲区,与 GPU 无关);忙卡 -6.3% 判定为噪声
- E-rms融合:确认存在 rms_norm+mul+rope 图级融合,报告已更新

## 监视政策更新(2026-11-21 06:45)
- 观察:GPU0 任务循环占用(PID 20123 结束后被 11010 顶替,显存相当)→ 组内有任务队列
- 风险:间隙抢测可能让下一个排队任务 OOM(20G+8.7G>24G),危害队友任务
- 新政策 idle_quick_suite.sh:5 分钟空闲阈值 + 每步碰撞守卫(外来进程→中止让卡)+ 只跑精简 3 项(基线/KV长上下文AB/mmvqw8)
- 完整 7 步矩阵 + serve_bench 并发聚合:保留给用户协调的专属窗口

## 基线-fa-on(GPU0窗口)  (11-23 08:23:16, GPU0, 快照: 0, 2 MiB, 0 %;1, 5275 MiB, 34 %;2, 2 MiB, 0 %;3, 1545 MiB, 0 %;)
```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24217 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24217 MiB
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           pp512 |    10775.90 ± 576.97 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           pp512 |    11482.15 ± 168.40 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           tg256 |        195.37 ± 0.16 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           tg256 |        195.42 ± 0.14 |

build: unknown (0)
```

## KV-q8_0-长32K  (11-23 08:23:38, GPU0, 快照: 0, 2 MiB, 86 %;1, 5275 MiB, 76 %;2, 2 MiB, 0 %;3, 1545 MiB, 0 %;)
```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24217 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24217 MiB
| model                          |       size |     params | backend    | ngl | type_k | type_v |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | -----: | -----: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |   1 |           pp512 |    10686.01 ± 413.69 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |   1 |         pp32768 |      9692.68 ± 16.27 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |   1 |           tg256 |        192.88 ± 0.26 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   q8_0 |   q8_0 |   1 |           tg128 |        191.76 ± 0.46 |

build: unknown (0)
```

## KV-f16-长32K  (11-23 08:24:06, GPU0, 快照: 0, 2 MiB, 100 %;1, 5275 MiB, 25 %;2, 2 MiB, 0 %;3, 1545 MiB, 0 %;)
```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24217 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24217 MiB
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           pp512 |    10921.18 ± 672.29 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |         pp32768 |      9822.57 ± 10.30 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           tg256 |        194.87 ± 0.28 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           tg128 |        193.70 ± 0.37 |

build: unknown (0)
```

## 内核变体-MMVQW8  (11-23 08:24:34, GPU0, 快照: 0, 2 MiB, 0 %;1, 5275 MiB, 45 %;2, 2 MiB, 0 %;3, 1545 MiB, 0 %;)
```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24217 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24217 MiB
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           pp512 |    10920.64 ± 470.96 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           pp512 |    11411.19 ± 174.16 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           tg256 |        173.95 ± 0.15 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           tg256 |        174.08 ± 0.10 |

build: unknown (0)
```

## 还原复测  (11-23 08:24:58, GPU0, 快照: 0, 2 MiB, 0 %;1, 5275 MiB, 42 %;2, 2 MiB, 0 %;3, 1545 MiB, 0 %;)
```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24217 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24217 MiB
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           pp512 |    11109.79 ± 534.63 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           pp512 |    11414.67 ± 166.05 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           tg256 |        194.93 ± 0.23 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           tg256 |        195.00 ± 0.14 |

build: unknown (0)
```

## fa-off(GPU0干净)  (11-23 08:26:25, GPU0, 快照: 0, 2 MiB, 0 %;1, 5275 MiB, 72 %;2, 2 MiB, 0 %;3, 1545 MiB, 0 %;)
```
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24217 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24217 MiB
| model                          |       size |     params | backend    | ngl |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | --: | --------------: | -------------------: |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           pp512 |    11207.10 ± 589.59 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           pp512 |    11451.91 ± 172.45 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           tg256 |        195.21 ± 0.19 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   1 |           tg256 |        195.27 ± 0.15 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   0 |           pp512 |    10943.88 ± 432.88 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   0 |           pp512 |    11115.65 ± 203.06 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   0 |           tg256 |        194.92 ± 0.17 |
| qwen35moe 35B.A3B Q4_K - Small |  19.45 GiB |    34.66 B | CUDA       |  99 |   0 |           tg256 |        194.97 ± 0.24 |

build: unknown (0)
```

## 决策级结果分析(2026-11-23 08:23-08:30,GPU0 干净窗口,实测)
| 配置 | tg256 (tok/s) | vs 基线 | 预注册预测 | 判定 |
|---|---|---|---|---|
| **基线(fa on,默认)** | **195.21~195.42** | — | 85~95(严重保守) | **实测为预测的 2.2 倍;屋顶达成率 195/579=33.7%** |
| fa off(短上下文) | 194.92 | -0.2% | (忙卡曾 -6%,系噪声) | 中性,保留 fa on(兼容 KV 量化/长上下文) |
| KV q8_0 @32K | 192.88 | -1.3% | +5~15% | **证伪!** 32K 时 KV 流量仅占 27%,减半收益被反量化开销吃掉 |
| KV f16 @32K | 194.87 | -0.3% | — | 与短上下文基线持平(32K 拐点未到) |
| 内核变体 mmvqw8 | 173.95 | **-11.0%** | ±3% | **证伪!** nwarps=4 上游调优正确,8 反而差 |
| 还原复测 | 194.93~195.00 | ✓ | — | 变体换装无污染,方法论闭环 |

**决策**:三项优化候选全部否决;当前最优配置 = fa on + 默认参数(195.4 tok/s)。
**教训**:①忙卡数据噪声比想象大(fa -6% 实为 0);②上游微参数已充分调优;③真正剩余空间在聚合并发(测试中)与量化档位(IQ4_XS)。

## 32K 长上下文附加数据
pp32768 ≈ 9693-9823 tok/s(KV q8_0 与 f16 相当);tg128 @32K 上下文 ≈ 191.8-193.7。

## 并发聚合实测(2026-11-26,GPU0 干净,单次测量)
serve_bench 首三轮作废(503 当就绪 / wait 挂死 / 墙钟口径含 prefill),第四轮两种口径分开报。

| 并发 | 纯生成聚合 tok/s | 每路 tok/s | 端到端聚合 tok/s |
|---:|---:|---:|---:|
| 1 | 176.7 | 176.7 | 160.0 |
| 2 | 278.7 | 139.4 | 256.0 |
| 4 | 389.4 | 97.4 | 353.1 |
| 8 | 530.3 | 66.3 | 487.6 |
| 16 | 672.9 | 42.1 | 611.3 |

限制:单次测量、无预热、每路仅 256 token;未验证并发下的输出正确性。
结论:16 路聚合为单流 3.8 倍,未测到封顶点。这是调度层收益,不是算子优化。
**本轮未找到任何能提升单流解码速度的优化项。**
