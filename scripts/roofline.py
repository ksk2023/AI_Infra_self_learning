#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3.6-35B-A3B 单卡解码 Roofline 计算器
----------------------------------------
每 token 显存流量模型:
    T(N) = W + S + kv_per_pos * (N+1)
    屋顶速度 = 有效带宽 / T(N)

用法示例:
    python3 roofline.py                          # 默认: 4090 + Q4_K_S + BF16 KV
    python3 roofline.py --bpw 4.2                # 换 IQ4_XS
    python3 roofline.py --kv-bytes 1             # KV cache 量化到 FP8/INT8
    python3 roofline.py --mtp 2                  # MTP 投机解码,平均接受长度 2
    python3 roofline.py --ctx 0 32768 131072     # 自定义评估的上下文长度
    python3 roofline.py --state-bytes 2          # DeltaNet 状态用 bf16
"""
import argparse

GiB = 1024 ** 3


def human(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


def main() -> None:
    p = argparse.ArgumentParser(description="单卡解码 Roofline 计算器")
    # 硬件
    p.add_argument("--bw", type=float, default=1008e9, help="显存带宽 B/s(4090=1008e9)")
    p.add_argument("--vram", type=float, default=24564 * 2**20, help="显存总字节(4090=24564MiB)")
    p.add_argument("--overhead", type=float, default=1.4 * GiB, help="运行时预留(CUDA上下文/计算缓冲)")
    # 模型激活参数量
    p.add_argument("--act", type=float, default=2.69e9, help="每 token 激活参数量")
    p.add_argument("--bpw", type=float, default=4.8, help="权重平均比特/权重(Q4_K_S≈4.8, IQ4_XS≈4.2)")
    p.add_argument("--weights", type=float, default=20_893_015_008, help="权重文件字节数(算 VRAM 用)")
    # KV cache(全注意力层)
    p.add_argument("--attn-layers", type=int, default=10)
    p.add_argument("--kv-heads", type=int, default=2)
    p.add_argument("--head-dim", type=int, default=256)
    p.add_argument("--kv-bytes", type=float, default=2, help="每个 KV 元素字节数(bf16=2, fp8=1)")
    # DeltaNet 状态
    p.add_argument("--delta-layers", type=int, default=30)
    p.add_argument("--state-mib", type=float, default=2.0, help="每层状态大小 MiB(fp32)")
    p.add_argument("--state-bytes", type=float, default=4, help="状态元素字节数(fp32=4, bf16=2)")
    # 输出控制
    p.add_argument("--ctx", type=int, nargs="+", default=[0, 32768, 65536, 78600, 131072, 262144],
                   help="评估的上下文长度列表")
    p.add_argument("--mtp", type=float, default=1.0, help="MTP 平均接受长度(1=关闭)")
    p.add_argument("--rates", type=float, nargs="+", default=[0.10, 0.25, 0.50],
                   help="展示的屋顶达成率")
    a = p.parse_args()

    W = a.act * a.bpw / 8.0
    S = a.delta_layers * a.state_mib * 2**20 * (a.state_bytes / 4.0) * 2  # 读+写
    kv = a.attn_layers * 2 * a.kv_heads * a.head_dim * a.kv_bytes          # 每位置字节

    print("=" * 78)
    print("每 token 流量构成")
    print(f"  W  激活权重   {a.act/1e9:.2f}B × {a.bpw}bpw      = {human(W)}")
    print(f"  S  DeltaNet态 {a.delta_layers}层 × {a.state_mib}MiB × {a.state_bytes}B × 2(读写) = {human(S)}")
    print(f"  kv 每位置     {a.attn_layers}层 × 2×{a.kv_heads}头×{a.head_dim}维 × {a.kv_bytes:g}B = {kv/1024:.1f} KiB/pos")
    n_cross = (W + S) / kv
    print(f"  拐点: N ≈ {n_cross:,.0f}(KV 读取追平 W+S;超过后 KV 是主要矛盾)")
    print("=" * 78)

    hdr = f"{'N':>9} | {'T(N)':>8} | {'屋顶 tok/s':>10} | " + " | ".join(f"@{int(r*100)}%".rjust(7) for r in a.rates) + " | MTP×%g" % a.mtp
    print(hdr); print("-" * len(hdr))
    for n in a.ctx:
        T = W + S + kv * (n + 1)
        roof = a.bw / T
        cells = " | ".join(f"{roof*r:7.0f}" for r in a.rates)
        print(f"{n:>9,} | {human(T):>8} | {roof:>10.0f} | {cells} | {roof*a.mtp:7.0f}")

    # VRAM 约束下的最大上下文
    kv_budget = a.vram - a.weights - a.overhead - a.delta_layers * a.state_mib * 2**20
    if kv_budget > 0:
        n_max = kv_budget / kv
        print("-" * len(hdr))
        print(f"VRAM 约束: 权重 {human(a.weights)} + 预留 {human(a.overhead)} → KV 预算 {human(kv_budget)}")
        print(f"  → 单卡可容纳最大上下文 N ≈ {n_max:,.0f}(超出则必须降 bpw / KV 量化 / 减预留)")
    else:
        print("!! 权重+预留已超显存,无法放置")


if __name__ == "__main__":
    main()
