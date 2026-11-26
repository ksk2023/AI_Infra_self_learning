#!/usr/bin/env bash
# 4090 优化工作环境(编译/运行 llama.cpp 都要 source 这个)
export CUDA_HOME=/home/dev/tools/cuda-home
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# ld 解析 .so 传递依赖时需要 rpath-link(不读 LD_LIBRARY_PATH)
export LDFLAGS_RPATH="-Wl,-rpath-link,$CUDA_HOME/lib64"
