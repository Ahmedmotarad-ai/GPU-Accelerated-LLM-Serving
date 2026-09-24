"""HTTP client for the downstream vLLM OpenAI-compatible server.

Configuration comes from environment variables (defaults shown):

    VLLM_BASE_URL     http://127.0.0.1:9033
    VLLM_MODEL_NAME   openbmb/MiniCPM5-2B-GPTQ
    VLLM_TIMEOUT      120
"""

import os
from typing import Tuple

import httpx

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:9033")
VLLM_MODEL_NAME = os.environ.get(
    "VLLM_MODEL_NAME", "openbmb/MiniCPM5-2B-GPTQ"
)
VLLM_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "120"))


class VLLMClientError(Exception):
    """Raised when vLLM cannot be reached or returns an upstream error."""

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


class VLLMClient:
    def __init__(
        self,
        base_url: str = VLLM_BASE_URL,
        model: str = VLLM_MODEL_NAME,
        timeout: float = VLLM_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def is_reachable(self) -> bool:
        """Verify vLLM is up via the OpenAI /v1/models endpoint."""
        try:
            response = await self._client.get("/v1/models")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def chat(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, str, int]:
        """Call vLLM /v1/chat/completions.

        Returns (text, model, completion_tokens).
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            response = await self._client.post(
                "/v1/chat/completions", json=payload
            )
        except httpx.TimeoutException as exc:
            raise VLLMClientError(
                "Inference request timed out.", status_code=504
            ) from exc
        except httpx.HTTPError as exc:
            raise VLLMClientError(
                "Could not reach the vLLM inference server.", status_code=503
            ) from exc

        if response.status_code >= 400:
            detail = ""
            try:
                body = response.json()
                error = body.get("error", {})
                detail = error.get("message", "") if isinstance(error, dict) else ""
                if not detail:
                    detail = body.get("detail", "")
            except ValueError:
                pass
            raise VLLMClientError(
                detail or f"vLLM returned HTTP {response.status_code}.",
                status_code=502,
            )

        data = response.json()
        text = data["choices"][0]["message"]["content"]
        model = data.get("model", self.model)
        completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
        return text, model, int(completion_tokens)