# GPU-Accelerated LLM Inference Server

Production-style LLM inference stack for **openbmb/MiniCPM5-2B** on a
**NVIDIA Tesla T4**, combining **vLLM**, the **FlagOS** operator-dispatch
plugin, **FlagGems** kernels, **AWQ 4-bit** quantization, and a
**Streamlit → FastAPI → vLLM** serving pipeline.

## 1. Project Overview

This repository demonstrates a full GPU LLM serving workflow:

- Inference engine: **vLLM** (OpenAI-compatible API)
- Operator/stencil acceleration: **FlagOS** (`vllm-plugin-FL`) + **FlagGems**
- Quantized serving: **AWQ 4-bit** (`openbmb/MiniCPM5-2B-GPTQ`, Marlin kernel)
- Serving layer: **FastAPI** gateway + **Streamlit** UI
- Benchmarking: real Tesla T4 measurements (historical run, see section 13)

## 2. Architecture

```text
Streamlit UI  (http://localhost:8501)
     |  HTTP  (API_BASE_URL, default http://127.0.0.1:8000)
     v
FastAPI Gateway  (port 8000)   /health  /generate  /metrics
     |  HTTP  (VLLM_BASE_URL, default http://127.0.0.1:9033)
     v
vLLM OpenAI-Compatible API  (port 9033)
     |  FlagOS plugin dispatch -> FlagGems kernels
     v
MiniCPM5-2B (AWQ 4-bit, Marlin)  ->  NVIDIA Tesla T4 GPU
```

The gateway **never loads the model**. vLLM remains the sole inference engine.
Streamlit communicates only with the gateway.

## 3. Target Hardware

| Property              | Value                     |
| --------------------- | ------------------------- |
| GPU                   | NVIDIA Tesla T4          |
| VRAM                  | 16 GB                    |
| Compute capability    | 7.5                      |
| CUDA                  | 13.0                     |
| Driver                | 580.82.07                |

The T4 cannot use FlashAttention 2; the tested configuration used the Triton
attention path (see section 9).

## 4. Software Stack

Runtime versions verified on the original Colab T4 runtime:

| Component      | Version / source                         |
| -------------- | ---------------------------------------- |
| Python         | 3.12.3                                   |
| PyTorch        | 2.11.0+cu130                            |
| vLLM           | 0.24.0                                   |
| FlagOS plugin  | submodule `vllm-plugin-FL` @ `fd5c727`   |
| FlagGems       | submodule `FlagGems` @ `f7c55cb`         |
| Model (FP16)   | `openbmb/MiniCPM5-2B`                    |
| Model (AWQ)    | `openbmb/MiniCPM5-2B-GPTQ`               |

## 5. Installation

### A. Inference runtime (GPU host / Colab T4)

```bash
git clone --recurse-submodules https://github.com/Ahmedmotarad-ai/GPU-Accelerated-LLM-Serving.git
cd GPU-Accelerated-LLM-Serving

# Optional venv (WSL / Linux)
python3 -m venv ~/flagos-env && source ~/flagos-env/bin/activate

# Base inference stack
pip install --upgrade pip
pip install "vllm==0.24.0"
pip install "huggingface_hub>=0.26"

# FlagOS plugin + FlagGems (editable installs from submodules)
bash scripts/setup_plugin.sh

# Verification
python scripts/verify_plugin.py
```

### B. GUI / gateway (any machine, CPU-only)

```bash
pip install -r requirements-gui.txt
```

## 6. Repository Structure

```text
GPU-Accelerated-LLM-Serving/
│
├── api/
│   ├── __init__.py
│   ├── client.py        # httpx client -> vLLM OpenAI API
│   ├── main.py          # FastAPI: /health /generate /metrics
│   └── schemas.py       # Pydantic request/response validation
│
├── tests/
│   └── test_api.py      # 12 gateway tests (mocked vLLM, no GPU)
│
├── scripts/
│   ├── setup_wsl.sh     # venv + vllm==0.24.0 install (WSL/Linux)
│   ├── setup_plugin.sh  # editable install of vllm-plugin-FL + FlagGems
│   ├── download_model.py# HF snapshot download helper
│   ├── smoke_minicpm5.py# BnB smoke path (experiment; see note in file)
│   ├── verify_plugin.py # import checks: flag_gems, vllm_fl, vllm
│   ├── test_cuda.py     # torch CUDA matmul check
│   └── test_bnb.py      # bitsandbytes linear4bit check
│
├── benchmark_results/
│   └── README.md        # historical T4 results + methodology (provenance)
│
├── FlagGems/            # git submodule (@ f7c55cb, flagos-ai/FlagGems)
├── vllm-plugin-FL/      # git submodule (@ fd5c727, flagos-ai/vllm-plugin-FL)
│
├── requirements-gui.txt # FastAPI + Streamlit deps only
├── requirements-dev.txt # pytest (+ dev tools)
├── .env.example         # environment template (no secrets)
├── .gitignore           # excludes weights, caches, venvs, .env
├── .dockerignore        # Docker build-context exclusions (weights, .git, caches)
├── Dockerfile           # gateway/UI container image (never loads the model)
├── docker-compose.yml   # vllm (GPU) + api + streamlit services
├── streamlit_app.py     # Streamlit frontend
└── README.md
```

`models/` holds downloaded weights and is intentionally untracked (see model
setup). Submodule checkouts are cloned with `--recurse-submodules`.

## 7. Model Setup

Weights are **downloaded from Hugging Face**; they are never committed.

```bash
# FP16 checkpoint (needed only for non-quantized serving / smoke tests)
python scripts/download_model.py \
  --repo openbmb/MiniCPM5-2B \
  --local-dir models/openbmb/MiniCPM5-2B

# AWQ 4-bit checkpoint (primary serving path)
python scripts/download_model.py \
  --repo openbmb/MiniCPM5-2B-GPTQ \
  --local-dir models/openbmb/MiniCPM5-2B-GPTQ
```

vLLM can also load directly from the HF hub by model id, or from the HF cache
snapshot path.

## 8. vLLM Serving

Start the OpenAI-compatible server on `:9033`:

```bash
vllm serve openbmb/MiniCPM5-2B-GPTQ \
  --quantization awq \
  --dtype float16 \
  --gpu-memory-utilization 0.30 \
  --enforce-eager \
  --max-model-len 512 \
  --port 9033
```

Smoke test:

```bash
curl http://127.0.0.1:9033/v1/models
```

Expected log markers (T4 runtime): `Using MarlinLinearKernel for
AutoAWQMarlinLinearMethod`, quantization `auto_awq`, and the FlagOS
attention backend active (`Op 'attention_backend' using 'default.flagos'`).

## 9. FlagOS Configuration

These environment variables are the verified Tesla T4 FlagOS configuration
(inject them before launching vLLM):

```bash
export VLLM_VENDOR=cuda
export FLAGGEMS_VENDOR=nvidia
export VLLM_PLUGINS=fl
export VLLM_FL_PREFER=flagos
export VLLM_FL_PREFER_ENABLED=True
export VLLM_FL_USE_FLAGGEMS_ATTN=0
export VLLM_FL_FLAGOS_WHITELIST="attention_backend"
```

- `VLLM_FL_USE_FLAGGEMS_ATTN=0` keeps the attention backend on the Triton/
  vLLM path. The **T4 is compute capability 7.5 and cannot use FlashAttention 2**;
  the tested configuration therefore used the Triton attention backend.
- `VLLM_FL_FLAGOS_WHITELIST` selects which operators FlagOS dispatches.

### SiLU experiment

To route SiLU+GELU through FlagOS as well (the tested SiLU configuration):

```bash
export VLLM_FL_FLAGOS_WHITELIST="silu_and_mul,attention_backend"
```

## 10. FlagGems Integration

`FlagGems` provides the GPU kernel library used by the FlagOS plugin. It is
pinned as a git submodule (`flagos-ai/FlagGems` @ `f7c55cb`) and installed
editable:

```bash
bash scripts/setup_plugin.sh   # pip install --no-build-isolation -e ./FlagGems
python -c "import flag_gems; print('flag_gems OK')"
```

## 11. Quantization

The served checkpoint is `openbmb/MiniCPM5-2B-GPTQ`. Although the Hugging Face
repository name contains **GPTQ**, vLLM consumes the checkpoint through its
**AWQ** runtime path:

| Property            | Value         |
| ------------------- | ------------- |
| quant_method        | awq           |
| bits                | 4             |
| group_size          | 128           |
| zero_point          | true          |
| desc_act           | false         |
| checkpoint_format   | gemm          |
| vLLM kernel         | Marlin (`MarlinLinearKernel`) |

The runtime/benchmark configuration is therefore described as
**AWQ 4-bit (Marlin)** inference, not generic GPTQ inference.

## 12. Benchmark Methodology

- Workload: 1024 input / 1024 output tokens, concurrency 1, 4 prompts
  (`--test-cases '[[1024,1024,1,4]]'`).
- Metrics: TTFT (time to first token), TPOT (time per output token),
  aggregate output token/s.
- Tool: `vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py` against the
  vLLM endpoint (`--port 9033 --served-model-name openbmb/MiniCPM5-2B-GPTQ`).

```bash
python vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model "<path to local checkpoint or HF cache snapshot>" \
  --port 9033 \
  --served-model-name openbmb/MiniCPM5-2B-GPTQ \
  --test-cases '[[1024,1024,1,4]]'
```

Run this **after** the vLLM server (section 8) is up. Results are written under
`vllm-plugin-FL/benchmark_results/`.

## 13. Historical T4 Benchmark Results

Measured on a live Colab Tesla T4 during the original runtime. The raw CSV
artifacts were lost when that runtime expired; the numbers below are preserved
as historical measurements (see `benchmark_results/README.md`).

| Configuration            | Total tok/s | Output tok/s | TTFT      | TPOT      |
| ------------------------ | ----------: | -----------: | --------: | --------: |
| vLLM FP16 baseline       |       50.28 |        25.14 |  92.35 ms |   39.8 ms |
| FlagOS + SiLU            |       26.25 |       ~13.13 | 174.04 ms | ~76.1 ms  |
| FlagOS without SiLU      |       31.93 |       ~15.97 | 141.62 ms | ~62.4 ms  |
| AWQ 4-bit + FlagOS       |       27.61 |       ~13.81 | 171.93 ms | ~72 ms    |

> In the tested Tesla T4 configuration, the FlagOS-based configurations
> measured lower throughput than the baseline vLLM FP16 configuration.

This is a statement about these tested configurations on the Tesla T4, not a
general claim about FlagOS performance.

## 14. FastAPI Gateway

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint    | Method | Description                                     |
| ----------- | ------ | ----------------------------------------------- |
| `/health`   | GET    | Gateway + downstream vLLM reachability          |
| `/generate` | POST   | Chat completion via vLLM with measured latency  |
| `/metrics`  | GET    | In-process counters (requests, tokens, uptime)  |

```bash
curl -s http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain GPU quantization in simple terms.", "max_tokens": 256, "temperature": 0.2}'
```

```json
{
  "response": "...",
  "model": "openbmb/MiniCPM5-2B-GPTQ",
  "latency_ms": 1234.5,
  "completion_tokens": 120,
  "tokens_per_second": 97.2
}
```

Environment variables (see `.env.example`):

| Variable          | Default                    | Used by         |
| ----------------- | -------------------------- | --------------- |
| `VLLM_BASE_URL`   | `http://127.0.0.1:9033`    | FastAPI gateway |
| `VLLM_MODEL_NAME` | `openbmb/MiniCPM5-2B-GPTQ` | FastAPI gateway |
| `VLLM_TIMEOUT`    | `120`                      | FastAPI gateway |
| `API_BASE_URL`    | `http://127.0.0.1:8000`    | Streamlit       |
| `REQUEST_TIMEOUT` | `120`                      | Streamlit       |

## 15. Streamlit UI

```bash
python -m streamlit run streamlit_app.py --server.port 8501
```

Open http://localhost:8501. The UI shows model connection status badges,
prompt/max-tokens/temperature controls, the generated response, and latency /
output-tokens / tokens-per-second metrics.

## 16. Testing

```bash
pip install -r requirements-dev.txt
pytest -q tests/test_api.py
```

Expected: `12 passed`. The vLLM client is mocked, so the suite runs on any
machine without a GPU. GPU integration and benchmark runs require a T4 runtime.

## 17. Docker Deployment

Three-container deployment. The `vllm` service requires an NVIDIA GPU; `api`
and `streamlit` are lightweight CPU containers.

```text
docker-compose
   ├── vllm      (vllm/vllm-openai, NVIDIA GPU, host :9033)
   ├── api       (FastAPI gateway, host :8000)
   └── streamlit (host :8501)

streamlit http://localhost:8501
      |  API_BASE_URL=http://api:8000
      v
api http://localhost:8000  (/health /generate /metrics)
      |  VLLM_BASE_URL=http://vllm:8000
      v
vllm http://localhost:9033  (internal :8000 -> host :9033)
      |  GPU via deploy.resources.reservations.devices
      v
MiniCPM5-2B-GPTQ (AWQ 4-bit, downloaded from Hugging Face on first start)
```

Files:

- `Dockerfile` — gateway/UI image (python:3.12-slim; installs only
  `requirements-gui.txt`; **never loads a model**).
- `docker-compose.yml` — `vllm`, `api`, `streamlit` services, health checks,
  GPU reservation, `hf-cache` volume.
- `.dockerignore` — keeps weights/caches/git/submodules out of the build
  context.

### Prerequisites

1. Docker Engine with the Compose v2 plugin (`docker compose version`), or
   Docker Desktop.
2. NVIDIA driver (CUDA-capable) on the host.
3. **NVIDIA Container Toolkit** (`nvidia-container-toolkit`) installed and
   configured for the container runtime.
4. GPU: NVIDIA Tesla T4 (16 GB) — the validated target for this stack.

### Build and start

```bash
docker compose up --build -d
```

First startup downloads the model weights into the `hf-cache` volume
(several GB) and loads them onto the GPU before vLLM starts serving.

```bash
docker compose logs -f vllm      # model download + engine init
docker compose ps                # service health
```

### Stop

```bash
docker compose down              # stop containers (keep hf-cache volume)
docker compose down --volumes    # also delete the cached model weights
```

### URLs after startup

| Service    | URL                                       |
| ---------- | ----------------------------------------- |
| vLLM API   | http://localhost:9033/v1/models           |
| FastAPI    | http://localhost:8000/health              |
| Streamlit  | http://localhost:8501                     |

### GPU requirement

The vLLM service does not run without GPU access. It is reserved in compose:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

If the NVIDIA Container Toolkit is missing or misconfigured, the `vllm`
container crashes on startup (see Docker Troubleshooting below).

### Model download / cache behavior

- Weights are **not** in the repository. `openbmb/MiniCPM5-2B-GPTQ` is pulled
  from Hugging Face into the named volume `hf-cache`
  (`HF_HOME=/root/.cache/huggingface`) on first start.
- For gated models, create a local `.env` with `HF_TOKEN=hf_...` (never
  committed; `.env` is gitignored). Compose passes it to the `vllm` service.
- Reset the cache: `docker compose down --volumes`, then `docker compose up -d`.

### Configuration (compose interpolation via `.env`)

| Variable             | Default                                   |
| -------------------- | ----------------------------------------- |
| `VLLM_IMAGE`         | `vllm/vllm-openai:v0.24.0`                |
| `MODEL_ID`           | `openbmb/MiniCPM5-2B-GPTQ`                |
| `SERVED_MODEL_NAME`  | `openbmb/MiniCPM5-2B-GPTQ`                |
| `VLLM_GPU_MEM_UTIL`  | `0.30`                                    |
| `VLLM_MAX_MODEL_LEN` | `512`                                     |
| `VLLM_TIMEOUT`       | `120`                                     |
| `REQUEST_TIMEOUT`    | `120`                                     |
| `HF_TOKEN`           | (unset)                                   |

The AWQ runtime configuration is preserved end-to-end: `--quantization awq
--dtype float16 --enforce-eager`, with the pinned model id and served model
name (see sections 8 and 11).

### Example inference request

```bash
curl -s http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain GPU quantization in simple terms.", "max_tokens": 256, "temperature": 0.2}'
```

### FlagOS / FlagGems note

The Docker vLLM service uses the upstream `vllm/vllm-openai` image, which
includes the AWQ/Marlin path. The **FlagOS/FlagGems plugin stack is not
embedded in the container image**; it is built and run bare-metal via
`scripts/` (sections 5, 9 and 10). A containerized FlagOS variant would
require a custom vLLM image and T4 validation — tracked as future work.

### Docker Troubleshooting

| Symptom                              | Cause / fix                                                   |
| ------------------------------------ | ------------------------------------------------------------- |
| `vllm` container exits immediately   | No GPU access. Verify toolkit: `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` |
| Model never finishes loading         | First run downloads weights; check `docker compose logs -f vllm` and the `hf-cache` volume size |
| 503/502 from `api` (vllm unhealthy)  | Wait for model load (`start_period` is 300 s)                 |
| Image pull fails for the default tag | `VLLM_IMAGE` must match an existing upstream `vllm/vllm-openai` tag for your vLLM version |
| OOM on GPU                           | Lower `VLLM_GPU_MEM_UTIL` (0.30 fits the 2B AWQ on a 16 GB T4) |

### Verification status

- **Locally verified:** `docker compose config` parses; `Dockerfile`,
  `docker-compose.yml`, `.dockerignore` present; gateway unit tests pass.
- **Environment-dependent (not yet executed here):** image pull/build, vLLM GPU
  load, `/health` + `/generate` end-to-end, Streamlit availability, GPU
  visibility inside the vllm container. These require a Docker host with the
  NVIDIA Container Toolkit and a Tesla T4.

## 18. Troubleshooting

| Symptom                          | Cause / fix                                                     |
| -------------------------------- | --------------------------------------------------------------- |
| `/generate` -> 503/502 from gateway | vLLM unreachable or upstream error; check vLLM logs             |
| `/generate` -> 504                | Inference timeout; raise `VLLM_TIMEOUT`/`REQUEST_TIMEOUT`      |
| vLLM won't start, FlashAttention error | T4 is CC 7.5; migration path requires Triton attention (use verified env vars, section 9) |
| Benchmark `--model` path missing  | Pass the local checkpoint dir or the HF cache snapshot path     |
| CUDA OOM during load              | Keep `--gpu-memory-utilization` low (0.30 verified for 2B AWQ)  |
| Missing `vllm_fl` / `flag_gems`   | Re-run `scripts/setup_plugin.sh` (submodules must be checked out) |
| Submodule directory empty         | Clone with `--recurse-submodules` or run `git submodule update --init --recursive` |

## 19. Reproducibility

1. Clone with submodules:
   ```bash
   git clone --recurse-submodules https://github.com/Ahmedmotarad-ai/GPU-Accelerated-LLM-Serving.git
   ```
2. Reproduce the vLLM 0.24 + PyTorch 2.11 + CUDA 13 environment (section 5).
3. Download model checkpoints (section 7).
4. Export the FlagOS environment (section 9) and serve (section 8).
5. Run the gateway and UI (sections 14-15).

Pinned upstream dependencies:

```text
FlagGems      https://github.com/flagos-ai/FlagGems        @ f7c55cb
vllm-plugin-FL https://github.com/flagos-ai/vllm-plugin-FL @ fd5c727
```

## Not Yet Implemented (future phases)

- **Docker GPU validation**: the Docker files are implemented and
  configuration-validated, but an end-to-end container run on an NVIDIA
  Tesla T4 host still needs to be executed (see section 17).
- **Custom FlagOS vLLM image**: embedding the FlagOS/FlagGems plugin stack in
  a Docker image (currently served by the upstream `vllm/vllm-openai`
  container).
- **Re-created Colab notebook** (`notebooks/FlagOS_GPU_Inference.ipynb`). The
  original runtime notebook was lost; a fresh notebook with real execution
  output should be generated on a new T4 runtime.
- **Fresh benchmark run** to regenerate raw CSV artifacts.