# docker/jupyter/Dockerfile
# Jupyter для EDA та тренування моделі

FROM python:3.11-slim

LABEL maintainer="Home Credit Default Risk"
LABEL description="Jupyter notebook server для EDA та тренування"

# Системні залежності
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Встановити Python залежності (шар кешується якщо requirements не змінились)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Копіювати вихідний код
COPY src/        ./src/
COPY notebooks/  ./notebooks/
COPY dump_to_parquet.py .
COPY init_db.py .
COPY check_connection.py .

# Створити папки
RUN mkdir -p data/parquet models/plots

EXPOSE 8888

# Jupyter запускається через docker-compose command