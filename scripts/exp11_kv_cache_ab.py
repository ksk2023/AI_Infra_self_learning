import time
import statistics

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/dev/infra/models/Qwen2.5-1.5B-Instruct"
NEW_TOKENS = 128
ROUNDS = 6

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
).to("cuda").eval()

messages = [
    {
        "role": "user",
        "content": "请解释大语言模型推理中的 Prefill、Decode 和 KV Cache。",
    }
]

batch = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to("cuda")

prompt_len = batch["input_ids"].shape[-1]

print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("Input tokens:", prompt_len)
print("New tokens per run:", NEW_TOKENS)

# 两组使用相同生成配置，只切换 use_cache。
generation_args = dict(
    min_new_tokens=NEW_TOKENS,
    max_new_tokens=NEW_TOKENS,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.0,
    temperature=1.0,
    top_p=1.0,
    top_k=50,
    pad_token_id=tokenizer.pad_token_id,
    return_dict_in_generate=False,
)


@torch.inference_mode()
def run_once(use_cache):
    # 开始计时前，等此前的 GPU 工作完成。
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()

    output = model.generate(
        **batch,
        **generation_args,
        use_cache=use_cache,
    )

    # 等本次生成的 GPU 工作完成，再停止计时。
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    actual_new_tokens = output.shape[-1] - prompt_len
    assert actual_new_tokens == NEW_TOKENS, actual_new_tokens

    peak_gib = torch.cuda.max_memory_allocated() / 2**30

    del output
    return elapsed, actual_new_tokens, peak_gib


print("\nWarming up...")
for flag in (True, False, False, True):
    run_once(flag)

times = {True: [], False: []}

print("\n===== Measurements =====")
for round_id in range(ROUNDS):
    # 交替顺序：先开后关、先关后开。
    order = (True, False) if round_id % 2 == 0 else (False, True)

    for flag in order:
        elapsed, count, peak = run_once(flag)
        times[flag].append(elapsed)

        print(
            f"round={round_id + 1} "
            f"cache={str(flag):5s} "
            f"tokens={count} "
            f"time={elapsed:.4f}s "
            f"output_tps={count / elapsed:.2f} "
            f"peak_allocated={peak:.3f}GiB"
        )

print("\n===== Summary =====")
for flag in (True, False):
    values = times[flag]
    median_s = statistics.median(values)

    print(
        f"cache={str(flag):5s} "
        f"median={median_s:.4f}s "
        f"range=[{min(values):.4f}, {max(values):.4f}]s "
        f"output_tps={NEW_TOKENS / median_s:.2f}"
    )

speedup = (
    statistics.median(times[False])
    / statistics.median(times[True])
)
print(f"Cache speedup: {speedup:.3f}x")
