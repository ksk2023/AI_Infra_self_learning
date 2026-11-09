import sys
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/dev/infra/models/Qwen2.5-1.5B-Instruct"

print("1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

print("2. Loading model to RTX 4090...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True
).to("cuda").eval()

print("3. Preparing input...")

messages = [
    {
        "role": "user",
        "content": "请解释大语言模型推理中的 Prefill、Decode 和 KV Cache。"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

print("===== Model Input (chat template) =====")
print(tokenizer.decode(inputs[0]))
print("========================================")

temperature = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 20        # 0 = 不限制
top_p = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8     # 1.0 = 不限制
seed = int(sys.argv[4]) if len(sys.argv) > 4 else None
if seed is not None:
    torch.manual_seed(seed)  # 实验八：固定随机种子，让骰子可复现
print(f"4. Generating response (temperature={temperature}, top_k={top_k}, top_p={top_p}, seed={seed})...")

with torch.inference_mode():
    outputs = model.generate(
        inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id
    )

answer_tokens = outputs[0]
answer = tokenizer.decode(
    answer_tokens,
    skip_special_tokens=False
)

print("\n===== Qwen Response =====")
print(answer)

print("\n===== Manual decode loop (Prefill + Decode + KV Cache) =====")

im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
stop_ids = {tokenizer.eos_token_id, im_end_id}


def decode_loop(use_cache: bool, verbose: bool):
    """手写生成循环。use_cache=True：每步只喂 1 个新 token，历史由 KV Cache 携带；
    use_cache=False：每步重喂完整序列，前文全部重算。"""
    step_inputs = inputs   # 有缓存：step0 喂整个 prompt（Prefill），之后 1 个（Decode）
    full_seq = inputs      # 无缓存：每步喂 prompt + 已生成的全部 token
    past_kv = None
    total_ms = 0.0
    steps_done = 0
    with torch.inference_mode():
        for step in range(512):
            t = time.perf_counter()
            if use_cache:
                out = model(step_inputs, past_key_values=past_kv, use_cache=True)
            else:
                out = model(full_seq, use_cache=False)
            torch.cuda.synchronize()  # CUDA 是异步的：等 GPU 真正算完再停表
            dt_ms = (time.perf_counter() - t) * 1000
            total_ms += dt_ms

            next_id = int(out.logits[0, -1].argmax())   # 贪心：只看最后一个位置
            piece = tokenizer.decode(next_id)
            n_in = step_inputs.shape[-1] if use_cache else full_seq.shape[-1]
            if verbose or step % 16 == 0:
                print(
                    f"step {step:3d} | input {n_in:3d} tok"
                    f" | {dt_ms:6.1f} ms | {piece!r}",
                    flush=True,
                )

            if next_id in stop_ids:
                print("-> 命中停止符，循环结束（'纪律'由我们自己执行）")
                break
            if use_cache:
                past_kv = out.past_key_values        # K/V 存入缓存，交给下一步
                step_inputs = inputs.new_tensor([[next_id]])  # 之后每步只喂 1 个新 token
            full_seq = torch.cat([full_seq, inputs.new_tensor([[next_id]])], dim=1)
            steps_done = step + 1
    return total_ms, steps_done


t_cache, n_cache = decode_loop(use_cache=True, verbose=True)
t_nocache, n_nocache = decode_loop(use_cache=False, verbose=False)

print("\n===== KV Cache benchmark =====")
print(f"with cache   : {n_cache} tokens | {t_cache/1000:6.2f} s | {t_cache/max(n_cache,1):5.1f} ms/token")
print(f"without cache: {n_nocache} tokens | {t_nocache/1000:6.2f} s | {t_nocache/max(n_nocache,1):5.1f} ms/token")
print(f"slowdown     : x{t_nocache/t_cache:.2f}")

print("\n===== GPU Information =====")
print("GPU:", torch.cuda.get_device_name(0))
print(
    "Allocated memory:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GiB"
)
