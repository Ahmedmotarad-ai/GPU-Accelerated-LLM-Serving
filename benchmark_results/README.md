# Benchmark Results

## Provenance

The numbers below are **historical, real Tesla T4 measurements** collected on a
Google Colab T4 (16 GB VRAM, compute capability 7.5, CUDA 13.0) runtime at
`openbmb/MiniCPM5-2B` / `openbmb/MiniCPM5-2B-GPTQ`.

The raw CSV artifacts (`raw_runs_*.csv`, `summary_*.csv`) from that runtime were
**lost when the Colab runtime expired**. The repository therefore does **not**
ship reproducible raw benchmark artifacts for these runs.

## Historical T4 results

| Configuration            | Total tok/s | Output tok/s | TTFT      | TPOT      |
| ------------------------ | ----------: | -----------: | --------: | --------: |
| vLLM FP16 baseline       |       50.28 |        25.14 |  92.35 ms |   39.8 ms |
| FlagOS + SiLU            |       26.25 |       ~13.13 | 174.04 ms | ~76.1 ms  |
| FlagOS without SiLU      |       31.93 |       ~15.97 | 141.62 ms | ~62.4 ms  |
| AWQ 4-bit + FlagOS       |       27.61 |       ~13.81 | 171.93 ms | ~72 ms    |

## Interpretation

> In the tested Tesla T4 configuration, the FlagOS-based configurations
> measured lower throughput than the baseline vLLM FP16 configuration.

This is a statement about the tested configurations on the Tesla T4, not a
general claim that FlagOS is universally slower.

## Methodology (summary)

- Workload: 1024 input / 1024 output tokens, concurrency 1, 4 prompts.
- Metric definitions: TTFT = time to first token; TPOT = time per output token;
  token/s measured as aggregate output throughput.
- Tool: `vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py` against the
  vLLM OpenAI-compatible endpoint (port 9033).
- Configurations differed by the FlagOS operator dispatch whitelist
  (`VLLM_FL_FLAGOS_WHITELIST`) and by FP16 vs AWQ 4-bit quantization.

## Re-validation

New benchmark artifacts should be produced on a fresh T4 runtime using the
unchanged benchmark workflow:

```bash
python vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model "<path to local checkpoint or HF cache snapshot>" \
  --port 9033 \
  --served-model-name openbmb/MiniCPM5-2B-GPTQ \
  --test-cases '[[1024,1024,1,4]]'
```

Raw CSVs will be written under `vllm-plugin-FL/benchmark_results/`.