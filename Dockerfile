FROM python:3.12-slim

# App-tier image: FastAPI gateway + Streamlit UI.
# This container NEVER loads a model; it only talks to the vLLM GPU container.
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Application-tier dependencies only (see README -> FastAPI Gateway / Streamlit).
COPY requirements-gui.txt ./
RUN pip install --no-cache-dir -r requirements-gui.txt

COPY api ./api
COPY streamlit_app.py ./

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]