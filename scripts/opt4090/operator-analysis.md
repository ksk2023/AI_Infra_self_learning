# Qwen3.6-35B-A3B @ RTX 4090 解码算子分析报告

日期:2026-11-21 | 基于 llama.cpp v1.0.0(自编译,sm_89,CUDA Graphs ON)
方法:源码静态分析(ggml/src/ggml-cuda/*),对照 T58 NPU 优化方法论
状态:GPU0 被占,纯源码分析轮;所有结论待真机证伪/证实

## 一、解码热路径(batch=1,每 token)

```
40 层 × [MoE(全部 40 层)] + 30 层 × [DeltaNet] + 10 层 × [全注意力] + lm_head
```

| 算子 | 内核文件 | 启动配置(Q4_K,ncols=1) | 备注 |
|---|---|---|---|
| MoE 专家 GEMV | mmvq.cu `mul_mat_vec_q_moe` | GENERIC 表:nwarps=4, rows/block=1 | **上游已融合**:路由 gate/GLU/偏置/共享专家(虚拟专家切片)全部并入 up 投影内核 |
| token→专家重排 | mmid.cu `mm_ids_helper<8>` | 每专家 1 warp | 8 专家走模板特化 ✓ |
| DeltaNet(解码) | gated_delta_net.cu ar 路径 | 原子归约 | mma 路径仅 n_tokens≥64(预填充);状态快照 cpy 已图级融合 |
| 全注意力 | fattn-tile.cu | — | 只读当前长度以内 KV(T58 曾修的同款问题,上游已正确) |
| lm_head | mmvq.cu `mul_mat_vec_q` | nwarps=4, grid=248320 blocks | 单个最大 GEMV(K=2048→N=248320) |
| 启动开销 | — | CUDA Graphs **已启用**(USE_CUDA_GRAPH 实证) | PDL 仅 Hopper+,4090 无缘 |

## 二、与 T58 NPU 手写优化对照

| T58 手写项(57.9 tok/s 的功臣) | ggml-cuda v1.0.0 对应物 |
|---|---|
| MoE 两次汇聚合一次(moe1) | ✓ 单内核 `mul_mat_vec_q_moe` |
| 专家 GLU 融合(triglux) | ✓ gate/GLU/bias 融合进 up 投影 |
| 共享专家同算(trishare) | ✓ 虚拟专家切片(shared_up) |
| add+rms 融合(addrms) | ✓ rms_norm+mul+rope 三算子图级融合(ggml-cuda.cu:3425) |
| DeltaNet 双核交错(dn17) | 对应物为 mma 路径,仅预填充 |
| flash 越界读修复(41dafcd) | ✓ 上游已只读有效长度 |
| KV 清零(kvzero) | ✓ 运行时按需处理 |
| ELF 级手工补丁 | 无对应物(GPU 由驱动调度) |

**结论:算法级融合上游已做完大半;剩余空间在微参数与系统层。**

## 三、可动手清单(按预期收益排序)

| # | 优化项 | 类型 | 预期收益 | 状态 |
|---|---|---|---|---|
| 1 | 连续批处理并发(llama-server -np N) | 系统层 | 聚合 ×2~3 | serve_bench.sh 就绪 |
| 2 | KV q8_0 @长上下文 | 运行时 | +5~15% | 矩阵含 C/D |
| 3 | MMVQ 参数表(nwarps 4→8 / rows 1→2) | 内核微参数 | ±3% | **本轮新变体 mmvqw8** |
| 4 | vdr 反量化展开宽度 | 内核微参数 | ±2% | 待评估 |
| 5 | MMV_X/DMMV_X | 内核微参数 | **预期 0**(不作用于 mmvq) | 变体已建,预注册 |
| 6' | --repack 开关 | 运行时 | **修正:预期 0**(repack 是 CPU 布局重排,全 GPU 卸载无关;忙卡 -6.3% 系噪声,fa-off/repack-off 同为 74.0 即噪声地板证据) | 预注册更新 |
| 6 | 换 IQ4_XS 权重(4.2bpw) | 量化档位 | 屋顶 +14%(需另下权重) | 未开始 |

## 四、预注册预测(不变,见 log.md)

- mmvx 变体解码 ±0%;KV q8_0 短 -2~-4%、长 +5~15%;干净基线 tg256 85~95 tok/s

## 五、验证计划(GPU0 窗口一到)

矩阵(idle_full_matrix.sh)+ 新增 mmvqw8 变体 + serve_bench 并发聚合,全部带 GPU 快照与正确性抽检。
