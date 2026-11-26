#!/usr/bin/env bash
# 还原 build2 的 libggml-cuda 为默认编译版
set -euo pipefail
MAIN=/home/dev/tools/llama.cpp/build2
cmake --build "$MAIN" --target ggml -j 60 > /dev/null 2>&1 || true
echo "已还原(从 build2 缓存重建默认 libggml-cuda)"
