"""Pydantic request/response models for the LLM API Gateway."""

from pydantic import BaseModel, Field, field_validator

MAX_TOKENS_LIMIT = 4096


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User prompt sent to the LLM.")
    max_tokens: int = Field(
        256,
        ge=1,
        le=MAX_TOKENS_LIMIT,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )

    @field_validator("prompt")
    @classmethod
    def prompt_must_be_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty")
        return value


class GenerateResponse(BaseModel):
    response: str
    model: str
    latency_ms: float
    completion_tokens: int
    tokens_per_second: float


class HealthResponse(BaseModel):
    status: str
    service: str
    vllm: str


class MetricsResponse(BaseModel):
    requests_total: int
    tokens_generated_total: int
    uptime_seconds: float