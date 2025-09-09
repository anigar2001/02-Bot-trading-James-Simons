# Bot de Trading Cuantitativo (Testnet Binance)

Proyecto inspirado en principios cuantitativos (tipo Renaissance), adaptado a cripto y diseñado para operar en Binance Testnet. Incluye estrategias (Reversión a la Media, Momentum, Arbitraje de Pares), backtesting básico, componente de ML (RandomForest) y un dashboard web simple.

## Estructura

```
src/
  main.py                 # Entrada principal (live/backtest)
  strategies/
    base.py               # Clases base de señales/estrategia
    mean_reversion.py     # Reversión a la media
    momentum.py           # Momentum
    pairs_arbitrage.py    # Pares BTC/LTC
  utils/
    api.py                # Wrapper ccxt (Binance testnet)
    helpers.py            # Logging/utilidades
    indicators.py         # Indicadores (TA-Lib/pandas_ta fallback)
    risk.py               # Gestión de capital/posiciones
  backtests/
    backtest_strategies.py# Backtest rápido
  models/
    model_training.py     # Entrenamiento RandomForest
    predictor.py          # Carga/predicción del modelo
  dashboard/
    app.py                # Flask dashboard
    templates/index.html  # Vista
    templates/train.html  # Zona de entrenamiento
    templates/ai.html     # AI Portfolio Check

requirements.txt           # Dependencias
Dockerfile                 # Contenedor
.env.example               # Variables de entorno de ejemplo
```

## Modelo de Pares (convergencia BTC/LTC)

Entrenar un modelo que estime la probabilidad de convergencia del spread LTC/BTC (z-score) a corto plazo:

```
docker compose run --rm fetcher \
  python -m src.models.pairs_training \
    --symbols BTCUSDC,LTCUSDC \
    --timeframe 1h \
    --format parquet \
    --window 50 \
    --horizon 1

# Salida: src/models/pairs_model.pkl
# Contiene: modelo (logistic calibrado), lista de features y metadatos (símbolos, tf, ventana/horizonte)
```

Más adelante, el bot puede usarlo para filtrar/ponderar señales de arbitraje de pares (cuando la probabilidad de convergencia sea alta).

## Requisitos

- Python 3.10+
- Recomendado: entorno virtual `venv`
- Testnet de Binance (claves en `.env`)

Instalación local:

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edita .env con tus claves testnet
```

## Variables de Entorno

Ver `.env.example`. Importantes:

- `BINANCE_API_KEY` / `BINANCE_API_SECRET`
- `API_BASE` (por defecto testnet oficial: https://testnet.binance.vision)
- `INITIAL_CAPITAL` (300 por defecto)
- `RISK_PER_TRADE` (0.02 por defecto)
- `USE_ML` (1 para activar filtro del modelo entrenado)
- `ML_BUY_THRESH` / `ML_SELL_THRESH` (umbrales de probabilidad, 0.55/0.45 por defecto)

## Ejecución (Live / Testnet)

```
python src/main.py --mode live --symbols BTC/USDC,ETH/USDC,LTC/USDC --strategy all --timeframe 1m --interval 30
```

Usa `--dry` para simular órdenes sin enviarlas.

## Backtesting Rápido

```
python src/main.py --mode backtest --symbols BTC/USDC,ETH/USDC,LTC/USDC --timeframe 1h
```

El script `src/backtests/backtest_strategies.py` ejecuta un backtest simplificado para MR y Momentum.

## Entrenamiento del Modelo (ML)

```
python -m src.models.model_training --symbol BTC/USDT --timeframe 1h --limit 2000
```

## Forecaster Probabilístico (Load Quantile) y Estrategias Grid-like

Entrenar forecaster de quantiles (q10/q50/q90):

```
sudo docker compose run --rm fetcher \
  python -m src.models.train_load_forecaster \
    --symbols BTCUSDC,ETHUSDC,LTCUSDC \
    --timeframe 1h \
    --horizon 1d \
    --out src/models/quantile_models
```

Variables .env relevantes:

- `LOAD_QF_MODEL_PATHS` JSON con rutas pkl por TF, ej:
  `{ "1d":"src/models/quantile_models/BTCUSDC_1h_1d_load_qf.pkl", "4h":"...", "1h":"..." }`
- `LOAD_QF_THR_UP` / `LOAD_QF_THR_DOWN` (0.0005 / -0.0005 por defecto)
- `LOAD_QF_MAX_POS` (1.0)
- `PEAK_ENT_MAX` (1.8) y `PEAK_ADX_MIN` (18)

Estrategia Load Dispatch (swing):

```
python src/main.py --mode live --symbols BTC/USDC \
  --strategy load_dispatch_swing --timeframe 1h --interval 60
```

Estrategia Peak Shaving (scalping 1m):

```
python src/main.py --mode live --symbols BTC/USDC \
  --strategy peak_shaving_scalping --timeframe 1m --interval 30
```

Backtests rápidos:

```
python -c "from src.backtests.backtest_strategies import backtest_load_dispatch_swing; backtest_load_dispatch_swing('BTC/USDC','1h','1d',60)"
python -c "from src.backtests.backtest_strategies import backtest_peak_shaving_scalping; backtest_peak_shaving_scalping('BTC/USDC','1m',14)"
```

Genera `src/models/trained_model.pkl`. El dashboard y las estrategias pueden ampliarse para usar la probabilidad como filtro.

Entrenamiento desde particiones locales (reentrenos):

```
# Una vez tengas particiones (via backfill/resample), entrena desde esos datos
docker compose run --rm fetcher \
  python -m src.models.train_from_partitions --symbols BTCUSDC,ETHUSDC,LTCUSDC --timeframe 1h --format parquet

Walk-forward y modelos alternativos (LR, XGBoost, LightGBM):

```
# Recomendado: timeframes 15m o 1h para reducir ruido
docker compose run --rm fetcher \
  python -m src.models.walkforward_training \
    --symbols BTCUSDC,ETHUSDC,LTCUSDC \
    --timeframe 1h \
    --format parquet \
    --models lr,xgb,lgbm \
    --folds 5

# Genera: src/models/trained_lr.pkl, trained_xgb.pkl, trained_lgbm.pkl y best_model.pkl
# El bot puede usar el mejor con env: MODEL_PATH=src/models/best_model.pkl
```
```

## Dashboard

```
python -m src.dashboard.app
# Abre http://localhost:8000
```

Muestra el balance y las últimas operaciones a partir de `src/data/logs/trades.jsonl`.

## Descarga de datos históricos (Testnet)

```
python -m src.data.download_data --symbols BTCUSDT,ETHUSDT,LTCUSDT --timeframe 1h --limit 5000
```

Genera archivos CSV en `src/data/historical/`. Acepta símbolos con formato `BTCUSDT` (recomendado para testnet) o `BTC/USDT`.

Descarga por rango de fechas (prioritario a limit):

```
python -m src.data.download_data --symbols BTCUSDT --timeframe 1h --start 2024-01-01
```

Descarga en lote (símbolos y múltiples timeframes):

```
TRAIN_SYMBOLS=BTCUSDT,ETHUSDT,LTCUSDT TRAIN_TFS=1h,15m TRAIN_LIMIT=5000 \
python scripts/fetch_training_data.py
```

Desde contenedores (recomendado) y guardando CSVs en tu máquina:

```
# BTC en varios timeframes
docker compose run --rm fetcher python scripts/fetch_btc_data.py

# ETH en varios timeframes
docker compose run --rm fetcher python scripts/fetch_eth_data.py

# LTC en varios timeframes
docker compose run --rm fetcher python scripts/fetch_ltc_data.py

# O lote configurable por variables
docker compose run --rm fetcher \
  bash -lc "TRAIN_SYMBOLS=BTCUSDT,ETHUSDT,LTCUSDT TRAIN_TFS=1m,5m,15m,1h,4h TRAIN_LIMIT=5000 python scripts/fetch_training_data.py"
```

Backfill robusto por rango (particionado mensual, reintentable):

```
# Símbolos y timeframes, desde fecha (o horizonte por TF si omites --start)
docker compose run --rm fetcher \
  python -m src.data.backfill --symbols BTCUSDC,ETHUSDC,LTCUSDC --timeframes 1m,5m,15m,1h,4h --start 2024-01-01 --format parquet

# Notas:
# - Guarda en src/data/historical/partitioned/{SYM}/{TF}/YYYY-MM.parquet
# - Si ya existe, hace merge y deduplica por timestamp
# - Usa bloques de 1000 velas y pausa entre llamadas (ajustable)
# - Reanuda automáticamente desde la última vela guardada si vuelves a lanzarlo
```

Resampling local (construir 5m/15m/1h/4h a partir de 1m):

```
# Asumiendo que ya backfilleaste 1m en particiones parquet
docker compose run --rm fetcher \
  python -m src.data.resample --symbol BTCUSDC --source_tf 1m --targets 5m,15m,1h,4h --format parquet

# Repite para ETHUSDC y LTCUSDC si lo necesitas
```

## Docker

Construir y ejecutar:

```
docker build -t trading-bot .
docker run -d --name trading-bot --restart=always \
  -e BINANCE_API_KEY=xxx -e BINANCE_API_SECRET=yyy \
  -e API_BASE=https://testnet.binance.vision \
  trading-bot
```

Dashboard en contenedor (opcional):

```
docker run -d --name bot-dashboard --restart=always -p 8000:8000 \
  -e BINANCE_API_KEY=xxx -e BINANCE_API_SECRET=yyy \
  -e API_BASE=https://testnet.binance.vision \
  trading-bot python -m src.dashboard.app
```

## Despliegue continuo (servidor Linux)

### systemd

1. Copia el repo al servidor y configura `.env`.
2. Crea el servicio `/etc/systemd/system/crypto-bot.service`:

```
[Unit]
Description=Crypto Bot (Binance Testnet)
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/ruta/proyecto
EnvironmentFile=/ruta/proyecto/.env
ExecStart=/usr/bin/python /ruta/proyecto/src/main.py --mode live --strategy all --symbols BTC/USDC,ETH/USDC,LTC/USDC --timeframe 1m --interval 30
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

3. `sudo systemctl daemon-reload && sudo systemctl enable --now crypto-bot`.

Dashboard como servicio opcional:

```
[Service]
ExecStart=/usr/bin/python -m src.dashboard.app
```

### PM2 (alternativa)

```
pm2 start src/main.py --name crypto-bot --interpreter python -- --mode live --strategy all --symbols BTC/USDC,ETH/USDC,LTC/USDC --timeframe 1m --interval 30
pm2 save && pm2 startup
```

## Reentrenos automáticos

Servicio en docker-compose que reentrena periódicamente desde particiones locales y graba `src/models/trained_model.pkl`:

```
# Levantar el servicio (por defecto diario)
docker compose up -d trainer

# Variables útiles:
# TRAINER_SYMBOLS=BTCUSDT,ETHUSDT,LTCUSDT
# TRAINER_TF=1h
# TRAINER_FORMAT=parquet|csv
# TRAINER_INTERVAL_SECONDS=86400
```

## Notas de Riesgo

# Reentrenos automáticos
- El tamaño por trade es conservador y configurable por env.
- Se aplican stop-loss/take-profit gestionados por el bucle del bot.
- Cumple mínimos aproximados de Binance vía `ccxt` (precision/limits). Valida siempre en testnet.
# Ingesta desde Binance Vision (ZIP diarios)

Para grandes históricos de 1m, puedes ingerir los CSV diarios oficiales de Binance Vision sin API:

```
docker compose run --rm fetcher \
  python -m src.data.ingest_binance_archive \
    --symbols BTCUSDT,ETHUSDT,LTCUSDT \
    --timeframe 1m \
    --start 2018-01-01 --end 2018-12-31 \
    --format parquet \
    --alias_to_usdc

# Archivos origen: https://data.binance.vision/data/spot/daily/klines/{SYMBOL}/{TF}/{SYMBOL}-{TF}-YYYY-MM-DD.zip
# Salida: src/data/historical/partitioned/{SYM}/{TF}/YYYY-MM.parquet (merge + dedup)
```
