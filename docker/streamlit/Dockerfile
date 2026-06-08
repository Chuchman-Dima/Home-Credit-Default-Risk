# docker/streamlit/Dockerfile
# Streamlit dashboard — тільки runtime, без Jupyter

FROM python:3.11-slim

LABEL maintainer="Home Credit Default Risk"
LABEL description="Streamlit credit scoring dashboard"

# Системні залежності
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Встановити залежності
COPY requirements.txt .

# Streamlit-only install (без jupyter/ipykernel — зменшує образ)
RUN pip install --no-cache-dir --upgrade pip \
    && grep -v -E "^(jupyter|ipykernel|#)" requirements.txt \
       | pip install --no-cache-dir -r /dev/stdin \
    && pip install --no-cache-dir streamlit>=1.35

# Копіювати тільки потрібне для Streamlit
COPY src/            ./src/
COPY streamlit_app/  ./streamlit_app/

# Створити папки для volumes
RUN mkdir -p data/parquet models/plots

# Streamlit конфіг — вимкнути телеметрію та браузер
RUN mkdir -p /root/.streamlit
COPY docker/streamlit/config.toml /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]