# ─────────────────────────────────────────────────────────────────
#  Home Credit Default Risk — Streamlit App
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire src/ — містить app.py, db.py, features.py,
# dump_to_parquet.py, init_db.py, check_connection.py
COPY src/ ./src/

# check_connection.py і init_db.py також є в корені проєкту —
# копіюємо звідси щоб запускати як: python init_db.py
COPY check_connection.py .
COPY init_db.py          .

# Директорії для моделей і даних (монтуються через volumes)
RUN mkdir -p models/plots data/parquet data/home-credit-default-risk

# Streamlit config
RUN mkdir -p /root/.streamlit
COPY docker/streamlit_config.toml /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# app.py знаходиться в src/
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]