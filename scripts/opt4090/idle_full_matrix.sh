#!/usr/bin/env bash
# 等到空闲 GPU 后,自动执行全套决策级 A/B 矩阵(单变量、带快照、全记录)
set -uo pipefail
source /home/dev/opt4090/env.sh
OPT=/home/dev/opt4090
MODEL_MB=21500
IDLE_NEED=3

echo "[$(date '+%T')] 只监视 GPU0(组内约定,实验仅用 0 号卡)..."
idle_count=0; picked=0
while true; do
  line=$(nvidia-smi -i 0 --query-gpu=utilization.gpu,memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' -v need=$MODEL_MB '$1<=5 && ($2-$3)>=need {print "idle"; exit}')
  if [ "$line" = "idle" ]; then
    idle_count=$((idle_count+1))
    echo "[$(date '+%T')] GPU0 空闲 ($idle_count/$IDLE_NEED)"
    [ $idle_count -ge $IDLE_NEED ] && break
  else
    [ $idle_count -gt 0 ] && echo "[$(date '+%T')] GPU0 忙,重置"
    idle_count=0
  fi
  sleep 60
done
export BENCH_GPU=$picked
echo "[$(date '+%T')] === 选用 GPU$picked,开始矩阵 ==="

# 1) 干净基线(fa on + 默认 repack on)
bash $OPT/bench.sh "基线-fa-on(干净GPU$picked)" -p 512 -n 256 -r 5

# 2) fa off
bash $OPT/bench.sh "A-fa-off" -fa off -p 512 -n 256 -r 5

# 3) KV q8_0 短上下文
bash $OPT/bench.sh "B-kv-q8_0-短" -ctk q8_0 -ctv q8_0 -p 512 -n 256 -r 5

# 4) KV q8_0 长上下文(32K prompt,KV 读取压力测试)
bash $OPT/bench.sh "C-kv-q8_0-长32K" -ctk q8_0 -ctv q8_0 -p 32768 -n 128 -r 3
bash $OPT/bench.sh "D-kv-f16-长32K" -p 32768 -n 128 -r 3

# 5) 内核变体:逐个换 libggml-cuda 实测 → 还原
for v in mmvqw8 mmvx64 mmvx16; do
  bd=/home/dev/tools/llama.cpp/build-$v
  [ -f "$bd/bin/libggml-cuda.so" ] || { echo "变体 $v 未构建,跳过"; continue; }
  mkdir -p $OPT/ggml-backup
  for f in libggml-cuda.so libggml-cuda.so.1 libggml-cuda.so.1.0.0; do
    cp -a /home/dev/tools/llama.cpp/build2/bin/$f $OPT/ggml-backup/orig-$f
    cp -a "$bd/bin/$f" /home/dev/tools/llama.cpp/build2/bin/$f
  done
  tag=$(echo $v | tr 'a-z' 'A-Z')
  bash $OPT/bench.sh "E-内核变体-$tag" -p 512 -n 256 -r 5
  # 还原
  for f in libggml-cuda.so libggml-cuda.so.1 libggml-cuda.so.1.0.0; do
    cp -a $OPT/ggml-backup/orig-$f /home/dev/tools/llama.cpp/build2/bin/$f
  done
done

# 6) 收尾复测(确认还原后回到基线水平)
bash $OPT/bench.sh "F-还原复测" -p 512 -n 256 -r 5

# 7) 并发聚合吞吐(权重读取摊薄效应)
for conc in 1 2 4 8; do
  bash $OPT/serve_bench.sh $conc -c 8192 2>&1 | tail -3 | tee -a $OPT/log.md
done
echo "[$(date '+%T')] === 矩阵+聚合全部完成,详见 $OPT/log.md ==="
