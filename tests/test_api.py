"""API gateway tests.

These tests do NOT require a GPU or a running vLLM server; the
downstream vLLM client is stubbed via app.state.vllm_client.
"""

import pytest
from fastapi.testclient import TestClient

from api.client import VLLMClientError
from api.main import app


class FakeVLLMClient:
    def __init__(
        self,
        reachable: bool = True,
        chat_error: Exception | None = None,
        text: str = "A test response from vLLM.",
        model: str = "openbmb/MiniCPM5-2B-GPTQ",
        completion_tokens: int = 5,
    ):
        self.reachable = reachable
        self.chat_error = chat_error
        self.text = text
        self.model = model
        self.completion_tokens = completion_tokens

    async def is_reachable(self) -> bool:
        return self.reachable

    async def chat(self, prompt, max_tokens, temperature):
        if self.chat_error is not None:
            raise self.chat_error
        self.last_prompt = prompt
        self.last_max_tokens = max_tokens
        self.last_temperature = temperature
        return self.text, self.model, self.completion_tokens

    async def close(self) -> None:
        pass


def test_health_ok():
    fake = FakeVLLMClient(reachable=True)
    with TestClient(app) as client:
        app.state.vllm_client = fake
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "llm-gateway",
        "vllm": "reachable",
    }


def test_health_vllm_unreachable():
    fake = FakeVLLMClient(reachable=False)
    with TestClient(app) as client:
        app.state.vllm_client = fake
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "error"
    assert response.json()["vllm"] == "unreachable"


def test_generate_ok():
    fake = FakeVLLMClient(completion_tokens=10)
    with TestClient(app) as client:
        app.state.vllm_client = fake
        response = client.post(
            "/generate",
            json={
                "prompt": "Explain GPU quantization in simple terms.",
                "max_tokens": 256,
                "temperature": 0.2,
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["response"] == fake.text
    assert body["model"] == "openbmb/MiniCPM5-2B-GPTQ"
    assert body["completion_tokens"] == 10
    assert body["latency_ms"] >= 0
    assert body["tokens_per_second"] > 0
    assert fake.last_prompt == "Explain GPU quantization in simple terms."
    assert fake.last_max_tokens == 256
    assert fake.last_temperature == 0.2


def test_generate_invalid_empty_prompt():
    fake = FakeVLLMClient()
    with TestClient(app) as client:
        app.state.vllm_client = fake
        response = client.post("/generate", json={"prompt": ""})
    assert response.status_code == 422


def test_generate_invalid_whitespace_prompt():
    fake = FakeVLLMClient()
    with TestClient(app) as client:
        app.state.vllm_client = fake
        response = client.post("/generate", json={"prompt": "   "})
    assert response.status_code == 422


def test_generate_invalid_max_tokens_zero():
    fake = FakeVLLMClient()
    with TestClient(app) as client:
        app.state.vllm_client = fake
        response = client.post(
            "/generate", json={"prompt": "Hi", "max_tokens": 0}
        )
    assert response.status_code == 422


def test_generate_invalid_temperature_out_of_range():
    fake = FakeVLLMClient()
    with TestClient(app) as client:
        app.state.vllm_client = fake
        high = client.post(
            "/generate", json={"prompt": "Hi", "temperature": 2.5}
        )
        negative = client.post(
            "/generate", json={"prompt": "Hi", "temperature": -0.1}
        )
    assert high.status_code == 422
    assert negative.status_code == 422


def test_generate_missing_prompt_field():
    fake = FakeVLLMClient()
    with TestClient(app) as client:
        app.state.vllm_client = fake
        response = client.post("/generate", json={"max_tokens": 1})
    assert response.status_code == 422


def test_generate_vllm_unreachable():
    fake = FakeVLLMClient(
        chat_error=VLLMClientError(
            "Could not reach the vLLM inference server.", status_code=503
        )
    )
    with TestClient(app) as client:
        app.state.vllm_client = fake
        response = client.post(
            "/generate", json={"prompt": "Hi"}
        )
    assert response.status_code == 503


def test_generate_vllm_timeout():
    fake = FakeVLLMClient(
        chat_error=VLLMClientError(
            "Inference request timed out.", status_code=504
        )
    )
    with TestClient(app) as client:
        app.state.vllm_client = fake
        response = client.post(
            "/generate", json={"prompt": "Hi"}
        )
    assert response.status_code == 504


def test_generate_vllm_upstream_error_keeps_detail():
    fake = FakeVLLMClient(
        chat_error=VLLMClientError(
            "The model 'x' does not exist. Please ensure it is defined...",
            status_code=502,
        )
    )
    with TestClient(app) as client:
        app.state.vllm_client = fake
        response = client.post(
            "/generate", json={"prompt": "Hi"}
        )
    assert response.status_code == 502
    assert "does not exist" in response.json()["detail"]


def test_metrics_ok():
    fake = FakeVLLMClient(completion_tokens=7)
    with TestClient(app) as client:
        app.state.vllm_client = fake
        client.post("/generate", json={"prompt": "Hello"})
        response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["requests_total"] >= 1
    assert body["tokens_generated_total"] >= 7
    assert body["uptime_seconds"] >= 0