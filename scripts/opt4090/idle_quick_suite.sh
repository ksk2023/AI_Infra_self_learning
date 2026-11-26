#!/usr/bin/env bash
# 等到 GPU0 空闲后自动执行精简高价值套件(带碰撞守卫,防与组内任务队列冲突)
# 政策(2026-11-21 更新,观察到 GPU0 任务循环占用):
#   - 连续 5 分钟空闲才开测(排除任务切换间隙)
#   - 每步开始前检查 GPU0 无外来进程;发现外来进程立即中止并退出(让出显存)
#   - 只跑 3 项最高价值测试;完整矩阵需用户协调窗口
set -uo pipefail
source /home/dev/opt4090/env.sh
OPT=/home/dev/opt4090
IDLE_NEED=5

gpu0_foreign() {  # GPU0 上除本脚本外是否有计算进程 → 有则输出进程数
  nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null |
    grep -c "$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)"
}
gpu0_free() {  # 空闲判定:无计算进程 且 利用率<=5
  u=$(nvidia-smi -i 0 --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  [ "$(gpu0_foreign)" -eq 0 ] && [ "$u" -le 5 ]
}
guard() {  # 碰撞守卫:每步测试前调用
  if [ "$(gpu0_foreign)" -gt 0 ]; then
    echo "[$(date '+%T')] !! GPU0 出现外来进程,中止剩余测试,让出显存"
    exit 42
  fi
}

echo "[$(date '+%T')] 监视 GPU0(5 分钟空闲阈值 + 碰撞守卫)..."
idle=0
while true; do
  if gpu0_free; then
    idle=$((idle+1)); echo "[$(date '+%T')] 空闲 $idle/$IDLE_NEED"
    [ $idle -ge $IDLE_NEED ] && break
  else
    [ $idle -gt 0 ] && echo "[$(date '+%T')] 又忙,重置"
    idle=0
  fi
  sleep 60
done
echo "[$(date '+%T')] GPU0 已空闲 ${IDLE_NEED} 分钟,开始精简套件(每步带守卫)"

export BENCH_GPU=0
guard && bash $OPT/bench.sh "基线-fa-on(GPU0窗口)" -p 512 -n 256 -r 5
guard && bash $OPT/bench.sh "KV-q8_0-长32K" -ctk q8_0 -ctv q8_0 -p 32768 -n 128 -r 3
guard && bash $OPT/bench.sh "KV-f16-长32K" -p 32768 -n 128 -r 3

# mmvqw8 变体(换装→测→还原)
guard && {
  mkdir -p $OPT/ggml-backup
  for f in libggml-cuda.so libggml-cuda.so.1 libggml-cuda.so.1.0.0; do
    cp -a /home/dev/tools/llama.cpp/build2/bin/$f $OPT/ggml-backup/orig-$f
    cp -a /home/dev/tools/llama.cpp/build-mmvqw8/bin/$f /home/dev/tools/llama.cpp/build2/bin/$f
  done
  bash $OPT/bench.sh "内核变体-MMVQW8" -p 512 -n 256 -r 5
  for f in libggml-cuda.so libggml-cuda.so.1 libggml-cuda.so.1.0.0; do
    cp -a $OPT/ggml-backup/orig-$f /home/dev/tools/llama.cpp/build2/bin/$f
  done
}
guard && bash $OPT/bench.sh "还原复测" -p 512 -n 256 -r 5
echo "[$(date '+%T')] === 精简套件完成,详见 $OPT/log.md ==="
