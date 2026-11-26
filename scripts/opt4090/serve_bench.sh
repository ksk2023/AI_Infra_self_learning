#!/usr/bin/env bash
# llama-server 并发聚合吞吐测试:单流 vs N 路并发的总 tok/s
# 用法: bash serve_bench.sh <并发数> [server 额外参数...]
set -uo pipefail
source /home/dev/opt4090/env.sh
CONC=${1:?用法: serve_bench.sh <并发数> [server 参数...]}
shift
MODEL=/home/dev/models/Qwen3.6-35B-A3B-GGUF-Q4K_S/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
SRV=/home/dev/tools/llama.cpp/build2/bin/llama-server
PORT=18321
NTOK=256

CUDA_VISIBLE_DEVICES=0 "$SRV" -m "$MODEL" -ngl 99 -fa on -c 8192 -np "$CONC" \
  --port $PORT --host 127.0.0.1 "$@" > /tmp/serve_bench.log 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT

# 等服务就绪(必须 HTTP 200,503=模型还在加载)
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health")
  [ "$code" = "200" ] && break
  sleep 5; [ $i -eq 120 ] && { echo "server 启动超时(最后状态码 $code)"; exit 1; }
done
echo "server 就绪(并发槽=$CONC),开始 $CONC 路请求 × $NTOK token"

T0=$(date +%s.%N)
for i in $(seq 1 "$CONC"); do
  curl -s http://127.0.0.1:$PORT/v1/completions -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"用中文详细介绍一下第${i}个主题:人工智能的发展历史。\",\"max_tokens\":$NTOK,\"temperature\":0}" \
    > /tmp/serve_out_$i.json &
done
wait
T1=$(date +%s.%N)

TOT=0
for i in $(seq 1 "$CONC"); do
  n=$(python3 -c "import json;d=json.load(open('/tmp/serve_out_$i.json'));print(d['usage']['completion_tokens'])" 2>/dev/null || echo 0)
  TOT=$((TOT+n))
done
WALL=$(python3 -c "print(f'{$T1-$T0:.1f}')")
python3 -c "print(f'并发={$CONC} 总token={$TOT} 墙钟={$WALL}s 聚合吞吐={$TOT/$WALL:.1f} tok/s 单流等价={$TOT/$WALL/$CONC:.1f} tok/s')"
grep -oE 'eval time.*tokens per second.*' /tmp/serve_bench.log | tail -1
