"""Streamlit frontend for the GPU-Accelerated LLM Inference Server.

This UI never talks to vLLM directly. It calls the FastAPI gateway
(API_BASE_URL), which is the only component that talks to vLLM.
"""

import os

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
VLLM_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "openbmb/MiniCPM5-2B-GPTQ")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "120"))

st.set_page_config(
    page_title="GPU-Accelerated LLM Inference Server",
    page_icon="🤖",
    layout="wide",
)

st.title("GPU-Accelerated LLM Inference Server")
st.caption("vLLM • FlagOS • Triton • AWQ • NVIDIA T4")


def _badge(label: str, ok: bool) -> str:
    color = "#1fbf75" if ok else "#e74c3c"
    icon = "●" if ok else "○"
    return f'<span style="color:{color};font-weight:600;">{icon} {label}</span>'


def check_health() -> dict:
    """Return {'api': bool, 'vllm': bool, 'error': str|None}."""
    try:
        response = httpx.get(f"{API_BASE_URL}/health", timeout=10)
    except httpx.HTTPError:
        return {"api": False, "vllm": False, "error": "gateway_unreachable"}
    if response.status_code == 200:
        return {"api": True, "vllm": True, "error": None}
    return {"api": True, "vllm": False, "error": "vllm_unreachable"}


def format_validation_error(detail) -> str:
    """Turn a FastAPI 422 detail into a short human-readable message."""
    if isinstance(detail, list) and detail:
        first = detail[0]
        loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        msg = first.get("msg", "invalid input")
        return f"{loc}: {msg}" if loc else msg
    return str(detail)


with st.sidebar:
    st.header("Parameters")
    max_tokens = st.number_input(
        "Max Tokens",
        min_value=1,
        max_value=4096,
        value=256,
        step=32,
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.2,
        step=0.05,
    )

    st.divider()
    st.header("Server Status")
    health = check_health()
    col1, col2 = st.columns(2)
    col1.markdown(_badge("API", health["api"]), unsafe_allow_html=True)
    col2.markdown(_badge("vLLM", health["vllm"]), unsafe_allow_html=True)
    if not health["api"] and health["error"] == "gateway_unreachable":
        st.caption("FastAPI gateway is not reachable.")
    elif not health["vllm"]:
        st.caption("vLLM is not reachable through the gateway.")

    st.divider()
    st.caption(f"Model: `{VLLM_MODEL_NAME}`")
    st.caption("Quantization: `AWQ 4-bit`")


prompt = st.text_area(
    "Prompt",
    placeholder="Enter your prompt...",
    height=180,
)

generate = st.button("Generate", type="primary", use_container_width=True)

if generate:
    if not prompt.strip():
        st.warning("Please enter a prompt before generating.")
    else:
        payload = {
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        with st.spinner("Generating response..."):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/generate",
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
            except httpx.TimeoutException:
                st.error(
                    "⚠️ The inference server took too long to respond.\n"
                    "Try a shorter prompt or fewer max tokens, then retry."
                )
            except httpx.HTTPError:
                st.error(
                    "⚠️ Inference server is unavailable.\n"
                    "Make sure the FastAPI gateway and vLLM server are running."
                )
            else:
                if response.is_success:
                    data = response.json()
                    st.markdown("### Response")
                    st.markdown(f"> {data['response']}")

                    st.markdown("### Statistics")
                    c1, c2, c3 = st.columns(3)
                    latency_s = data["latency_ms"] / 1000.0
                    c1.metric("Latency", f"{latency_s:.2f} s")
                    c2.metric("Output Tokens", f"{data['completion_tokens']}")
                    c3.metric(
                        "Tokens/sec", f"{data['tokens_per_second']:.1f}"
                    )
                    st.markdown(
                        f"**Model:** `{data['model']}` · "
                        "**Quantization:** `AWQ 4-bit`"
                    )
                elif response.status_code == 422:
                    try:
                        detail = response.json().get("detail")
                    except ValueError:
                        detail = None
                    st.error(
                        "⚠️ Invalid request: "
                        f"{format_validation_error(detail) if detail else 'validation failed'}."
                    )
                elif response.status_code == 503:
                    st.error(
                        "⚠️ Inference server is unavailable.\n"
                        "Make sure the FastAPI gateway and vLLM server are running."
                    )
                elif response.status_code == 504:
                    st.error(
                        "⚠️ Inference server timed out.\n"
                        "Try again with fewer max tokens."
                    )
                else:
                    st.error(
                        f"⚠️ The server returned an error (HTTP {response.status_code})."
                    )
else:
    st.info("Enter a prompt above and click **Generate**.")