import os
import time

try:
    os.environ.setdefault("USE_FLAGGEMS", "0")
except Exception:
    pass

from vllm import LLM, SamplingParams

# Local smoke test (VLLM_* vars are exported in scripts/setup_wsl.sh).
# Model path is overridable via MINICPM_PATH for non-WSL runtimes.
DEFAULT_MODEL = "/mnt/c/Users/ASUS/Desktop/ahmed/Ai/10-projects/FlagOS/models/openbmb/MiniCPM5-2B"
MODEL = os.environ.get("MINICPM_PATH", DEFAULT_MODEL)

# NOTE: this smoke test loads the FP16 checkpoint with BitsAndBytes (NF4),
# an early experiment path. The production inference path is the AWQ 4-bit
# checkpoint served by vLLM (see README -> Quantization).

prompts = [
    "Hello, my name is",
    "The capital of France is",
    "def fibonacci(n):",
    "Explain quantum computing in one sentence.",
]


def main():
    llm = LLM(
        model=MODEL,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        dtype="float16",
        max_model_len=1024,
        gpu_memory_utilization=0.70,
        enforce_eager=True,
        max_num_batched_tokens=2048,
        max_num_seqs=16,
        trust_remote_code=True,
    )

    sp = SamplingParams(max_tokens=48, temperature=0.0)

    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    dt = time.time() - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(f"generated_tokens={total_tokens} elapsed={dt:.2f}s throughput={total_tokens/dt:.2f} tok/s")
    for o in outputs:
        print("PROMPT:", o.prompt[:40])
        print("GEN:", o.outputs[0].text.replace("\n", " ")[:80])


if __name__ == "__main__":
    main()