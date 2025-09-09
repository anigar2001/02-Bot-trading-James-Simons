FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl ca-certificates git cmake libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar TA-Lib desde fuente (para wrapper Python)
WORKDIR /tmp
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib && ./configure --prefix=/usr && make && make install \
    && cd / && rm -rf /tmp/ta-lib /tmp/ta-lib-0.4.0-src.tar.gz

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Puerto del dashboard Flask (opcional)
EXPOSE 8000

# Modo por defecto: bot en vivo
CMD ["python", "-m", "src.main", "--mode", "live"]
