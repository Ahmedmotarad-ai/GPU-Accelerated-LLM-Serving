# GPU-Accelerated LLM Inference Server

Production-style LLM serving stack on **NVIDIA Tesla T4 (16 GB)**:

- **vLLM 0.24.0** — OpenAI-compatible inference engine
- **FlagOS** + **FlagGems** — operator dispatch layers
- **MiniCPM5-2B** — AWQ/GPTQ 4-bit quantized (MarlinLinearKernel)
- **Triton attention backend** — compute capability 7.5 compatible

## Interactive Demo

A minimal three-tier demo: **Streamlit → FastAPI → vLLM → GPU**.

```text
Streamlit (UI)
     │  HTTP (API_BASE_URL)
     ▼
FastAPI (Gateway)          /health  /generate  /metrics
     │  HTTP (VLLM_BASE_URL)
     ▼
vLLM OpenAI-compatible API (port 9033)
     │
     ▼
MiniCPM5-2B AWQ 4-bit (Marlin)  →  NVIDIA T4
```

**Why Streamlit does not call vLLM directly:**

- Separation of concerns: the UI is only a frontend, vLLM only serves inference.
- The gateway owns request validation, latency/token metrics computation, and
  graceful error mapping — so the UI never sees raw vLLM errors or stack traces.
- One authoritative entry point for the model makes the stack easy to extend
  (auth, observability, routing) without touching the inference server.
- It mirrors how real inference platforms are deployed (client → gateway → engine).

### 1. Start vLLM (Terminal 1)

```bash
export VLLM_VENDOR=cuda
export FLAGGEMS_VENDOR=nvidia
export VLLM_PLUGINS=fl
export VLLM_FL_PREFER=flagos
export VLLM_FL_PREFER_ENABLED=True
export VLLM_FL_USE_FLAGGEMS_ATTN=0
export VLLM_FL_FLAGOS_WHITELIST=attention_backend

python -m vllm serve openbmb/MiniCPM5-2B-GPTQ \
  --quantization awq \
  --dtype float16 \
  --gpu-memory-utilization 0.30 \
  --enforce-eager \
  --max-model-len 512 \
  --port 9033
```

### 2. Start the FastAPI gateway (Terminal 2)

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Optionally `export VLLM_BASE_URL=...` first (defaults to `http://127.0.0.1:9033`).

### 3. Start Streamlit (Terminal 3)

```bash
python -m streamlit run streamlit_app.py --server.port 8501
```

Optionally `export API_BASE_URL=...` first (defaults to `http://127.0.0.1:8000`).
Open http://localhost:8501.

### API endpoints

| Endpoint   | Method | Description                                              |
| ---------- | ------ | -------------------------------------------------------- |
| `/health`  | GET    | Gateway + vLLM reachability check                        |
| `/generate`| POST   | Chat completion forwarded to vLLM with measured latency  |
| `/metrics` | GET    | In-process counters (requests, tokens generated, uptime) |

Example request:

```bash
curl -s http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain GPU quantization in simple terms.", "max_tokens": 256, "temperature": 0.2}'
```

Example response:

```json
{
  "response": "...",
  "model": "openbmb/MiniCPM5-2B-GPTQ",
  "latency_ms": 1234.5,
  "completion_tokens": 120,
  "tokens_per_second": 97.2
}
```

### Environment variables

| Variable          | Default                       | Used by         |
| ----------------- | ----------------------------- | --------------- |
| `VLLM_BASE_URL`   | `http://127.0.0.1:9033`       | FastAPI gateway |
| `VLLM_MODEL_NAME` | `openbmb/MiniCPM5-2B-GPTQ`    | FastAPI gateway |
| `VLLM_TIMEOUT`    | `120`                         | FastAPI gateway |
| `API_BASE_URL`    | `http://127.0.0.1:8000`       | Streamlit       |
| `REQUEST_TIMEOUT` | `120`                         | Streamlit       |

See `.env.example`.

### Testing the gateway (no GPU required)

```bash
pip install -r requirements-gui.txt pytest
python -m pytest tests/test_api.py -v
```

The downstream vLLM server is mocked, so the test suite does not need a GPU.

## Project Layout

```text
api/                    FastAPI gateway (schemas, client, main)
streamlit_app.py        Streamlit frontend
tests/test_api.py       Gateway API tests (mocked vLLM)
requirements-gui.txt    GUI/gateway dependencies only
.env.example            Configuration template
vllm-plugin-FL/         FlagOS plugin + benchmark scripts
FlagGems/               FlagGems kernel library
scripts/                Setup/smoke utilities
models/                 Local model checkpoints
```

## Benchmarking

The existing benchmark workflow is unchanged and lives under
`vllm-plugin-FL/benchmarks/`:

```bash
python vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model "<local model dir>" \
  --port 9033 \
  --served-model-name openbmb/MiniCPM5-2B-GPTQ \
  --test-cases '[[1024,1024,1,4]]'
```

Run the benchmark script **after** the vLLM server is up; it writes
`benchmark_results/raw_runs_*.csv` and `benchmark_results/summary_*.csv`.