#!/usr/bin/env bash
# 组装用户态 CUDA 工具链:/home/dev/tools/cuda-home
#   bin     ← NVIDIA deb 完整 nvcc 前端(12.4)
#   nvvm    ← pip wheel 的 libdevice/libnvvm
#   include ← pip cuda_runtime + cuda_nvcc(crt) + cccl
#   lib64   ← pip cudart + cublas
set -euo pipefail
SP=/home/dev/.local/lib/python3.10/site-packages/nvidia
DEB=/home/dev/tools/nvcc-deb/usr/local/cuda-12.4
CH=/home/dev/tools/cuda-home

NVVM_DEB=/home/dev/tools/nvvm-deb/usr/local/cuda-12.4
CUDART_DEB=/home/dev/tools/cudart-dev-deb/usr/local/cuda-12.4
CUDART_RT_DEB=/home/dev/tools/cudart-deb/usr/local/cuda-12.4

rm -rf "$CH"; mkdir -p "$CH/bin"
for t in "$DEB"/bin/*; do ln -s "$t" "$CH/bin/$(basename "$t")"; done
mv "$CH/bin/nvcc" "$CH/bin/nvcc.real"
cat > "$CH/bin/nvcc" <<'EOF'
#!/usr/bin/env bash
# 包装器:cicc 在 nvvm/bin,nvcc 经 sh 调用时需要 PATH 里可见
DIR="$(cd "$(dirname "$0")" && pwd)"
export PATH="$DIR/../nvvm/bin:$PATH"
exec "$DIR/nvcc.real" "$@"
EOF
chmod +x "$CH/bin/nvcc"
ln -s "$NVVM_DEB/nvvm"              "$CH/nvvm"

mkdir -p "$CH/include" "$CH/lib64"
# merge_include <源目录>:拷贝除 Python 元文件外的所有条目
merge_include() {
  local src=$1
  [ -d "$src" ] || return 0
  find "$src" -mindepth 1 -maxdepth 1 ! -name '__init__.py' ! -name '__pycache__' -printf '%f\n' |
  while read -r e; do
    [ -e "$CH/include/$e" ] || cp -rs "$src/$e" "$CH/include/"
  done
}
merge_include "$SP/cuda_runtime/include"
merge_include "$SP/cuda_nvcc/include"
merge_include "$SP/cuda_cccl/include"
merge_include "$SP/cublas/include"
merge_include "$DEB/include"
merge_include "$CUDART_DEB/targets/x86_64-linux/include"
cp -s  "$CUDART_RT_DEB/targets/x86_64-linux/lib/"*.so* "$CH/lib64/" 2>/dev/null || true
cp -s  "$SP/cublas/lib/"*.so*       "$CH/lib64/" 2>/dev/null || true
cp -s  "$CUDART_DEB/targets/x86_64-linux/lib/"*.a "$CH/lib64/" 2>/dev/null || true
# 生成开发用符号链接(libcudart.so 等)
for f in "$CH/lib64"/*.so.*; do
  base=$(echo "$f" | sed 's/\.so\..*/.so/')
  [ -e "$base" ] || ln -s "$(basename "$f")" "$base"
done

echo '== 编译冒烟测试 =='
cat > /tmp/smoke.cu <<'EOF'
#include <cuda_runtime.h>
#include <cstdio>
__global__ void k(float* x){ x[threadIdx.x] *= 2.f; }
int main(){
  float h[8]; for (int i=0;i<8;i++) h[i]=(float)i;
  float* d; cudaMalloc(&d,32); cudaMemcpy(d,h,32,cudaMemcpyHostToDevice);
  k<<<1,8>>>(d); cudaError_t e=cudaDeviceSynchronize();
  if (e!=cudaSuccess){ printf("kernel err %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(h,d,32,cudaMemcpyDeviceToHost); cudaFree(d);
  bool ok = h[3]==6.f; printf("h[3]=%g -> %s\n", h[3], ok?"PASS":"WRONG");
  return ok?0:1;
}
EOF
"$CH/bin/nvcc" --cudart=shared -arch=sm_89 -I"$CH/include" -L"$CH/lib64" -o /tmp/smoke /tmp/smoke.cu && LD_LIBRARY_PATH="$CH/lib64" /tmp/smoke; rc=$?
[ $rc -eq 0 ] && echo "SMOKE OK" || echo "SMOKE FAIL rc=$rc"
