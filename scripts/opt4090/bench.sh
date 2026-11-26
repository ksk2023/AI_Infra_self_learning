#!/usr/bin/env bash
# Qwen3.6-35B-A3B @ RTX4090 解码基准(单变量对照用)
# 用法: bash bench.sh <标签> [额外 llama-bench 参数...]
# 输出追加到 log.md;同时记录当时的 GPU 占用快照(诚实测量)。
set -uo pipefail
TAG=${1:?用法: bench.sh <标签> [llama-bench 参数...]}
shift
GPU=${BENCH_GPU:-0}                       # 只允许 GPU0(组内约定),勿改
MODEL=/home/dev/models/Qwen3.6-35B-A3B-GGUF-Q4K_S/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
BIN=/home/dev/tools/llama.cpp/build2/bin/llama-bench
LOG=/home/dev/opt4090/log.md
export LD_LIBRARY_PATH=/home/dev/tools/cuda-home/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

[ -x "$BIN" ] || { echo "llama-bench 不存在,先编译"; exit 1; }

SNAP=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')
echo "" >> "$LOG"
echo "## $TAG  ($(date '+%m-%d %H:%M:%S'), GPU$GPU, 快照: $SNAP)" >> "$LOG"

# 默认参数可被后续参数覆盖(llama-bench 取后者)
CUDA_VISIBLE_DEVICES=$GPU "$BIN" -m "$MODEL" -ngl 99 -fa on -p 512 -n 256 -r 3 "$@" 2>&1 | tee /tmp/bench_last.out
echo '```' >> "$LOG"; cat /tmp/bench_last.out >> "$LOG"; echo '```' >> "$LOG"
