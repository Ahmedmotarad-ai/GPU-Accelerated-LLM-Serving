"""FastAPI gateway for the GPU LLM Inference Server.

Routes:
    GET  /health     -> verify the gateway and the downstream vLLM server
    POST /generate   -> forward a chat completion request to vLLM
    GET  /metrics    -> lightweight in-process counters

The gateway never loads or serves the model; it only proxies to the
existing vLLM OpenAI-compatible endpoint.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.client import VLLMClient, VLLMClientError
from api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    MetricsResponse,
)

SERVICE_NAME = "llm-gateway"
_APP_STARTED = time.time()
_REQUESTS_TOTAL = 0
_TOKENS_GENERATED_TOTAL = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.vllm_client = VLLMClient()
    yield
    await app.state.vllm_client.close()


app = FastAPI(
    title="LLM API Gateway",
    description="Gateway between the Streamlit UI and the vLLM inference server.",
    version="0.1.0",
    lifespan=lifespan,
)


def _get_client(request: Request) -> VLLMClient:
    return request.app.state.vllm_client


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    client = _get_client(request)
    if not await client.is_reachable():
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="error", service=SERVICE_NAME, vllm="unreachable"
            ).model_dump(),
        )
    return HealthResponse(
        status="ok", service=SERVICE_NAME, vllm="reachable"
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(body: GenerateRequest, request: Request):
    global _REQUESTS_TOTAL, _TOKENS_GENERATED_TOTAL
    _REQUESTS_TOTAL += 1

    client = _get_client(request)
    start = time.perf_counter()
    try:
        text, model, completion_tokens = await client.chat(
            body.prompt, body.max_tokens, body.temperature
        )
    except VLLMClientError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - start) * 1000.0
    tokens_per_second = (
        completion_tokens / (latency_ms / 1000.0)
        if latency_ms > 0 and completion_tokens > 0
        else 0.0
    )
    _TOKENS_GENERATED_TOTAL += completion_tokens

    return GenerateResponse(
        response=text,
        model=model,
        latency_ms=round(latency_ms, 2),
        completion_tokens=completion_tokens,
        tokens_per_second=round(tokens_per_second, 2),
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics() -> MetricsResponse:
    return MetricsResponse(
        requests_total=_REQUESTS_TOTAL,
        tokens_generated_total=_TOKENS_GENERATED_TOTAL,
        uptime_seconds=round(time.time() - _APP_STARTED, 2),
    )