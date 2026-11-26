#!/usr/bin/env bash
# 编译期算子开关 A/B:用给定 ggml 编译开关构建变体,替换 build2 的 libggml-cuda,bench 后可还原
# 用法: bash build_variant.sh <名称> [cmake -D 参数...]
set -euo pipefail
source /home/dev/opt4090/env.sh
NAME=$1; shift
SRC=/home/dev/tools/llama.cpp
BD=$SRC/build-$NAME
MAIN=$SRC/build2

cmake -B "$BD" -S "$SRC" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCUDAToolkit_ROOT=$CUDA_HOME -DLLAMA_CURL=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS_RPATH" -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS_RPATH" "$@" > /dev/null
cmake --build "$BD" --target ggml -j 60 > "$BD.build.log" 2>&1

# 替换主构建的 ggml 库(备份当前)
mkdir -p /home/dev/opt4090/ggml-backup
for f in libggml-cuda.so libggml-cuda.so.1 libggml-cuda.so.1.0.0; do
  [ -e "$MAIN/bin/$f" ] && cp -a "$MAIN/bin/$f" /home/dev/opt4090/ggml-backup/$NAME-$f
  cp -a "$BD/bin/$f" "$MAIN/bin/$f"
done
echo "变体 [$NAME] 已装入 build2/bin"
