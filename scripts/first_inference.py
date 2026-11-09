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

step_inputs = inputs          # 第一次喂入整个 prompt（Prefill）
past_kv = None                # KV Cache，从第二步开始复用
with torch.inference_mode():
    for step in range(128):
        t = time.perf_counter()
        out = model(step_inputs, past_key_values=past_kv, use_cache=True)
        dt_ms = (time.perf_counter() - t) * 1000

        next_id = int(out.logits[0, -1].argmax())   # 贪心：只看最后一个位置
        piece = tokenizer.decode(next_id)
        print(
            f"step {step:3d} | input {step_inputs.shape[-1]:3d} tok"
            f" | {dt_ms:6.1f} ms | {piece!r}",
            flush=True,
        )

        if next_id in stop_ids:
            print(f"-> 命中停止符，循环结束（这次'纪律'由我们自己执行）")
            break
        past_kv = out.past_key_values       # 把 K/V 存起来给下一步
        step_inputs = inputs.new_tensor([[next_id]])  # 之后每步只喂 1 个新 token（Decode）

print("\n===== GPU Information =====")
print("GPU:", torch.cuda.get_device_name(0))
print(
    "Allocated memory:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GiB"
)
