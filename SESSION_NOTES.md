# Sesión de trabajo – Bot-trading-James-Simons (2025-09-08)

Resumen de cambios, cómo ejecutar en contenedor, y notas de uso. Útil si cierras el editor.

## Cambios clave en el repo
- src/main.py: procesa primero señales multi‑leg (pares) antes de filtrar HOLD (pares en live ya se ejecuta).
- src/models/pairs_training.py: añade --start_date/--end_date; filtra datasets por rango.
- src/backtests/pairs_portfolio_check.py:
  - Añade --spot_mode y --spot_symbol (spot long‑only, por defecto LTCUSDC).
  - Añade fees aproximados para LTC (compra y venta) y los descuenta en aperturas/cierres (no en HOLDs).
  - Exporta en CSV: value_usdc, z, prob, action, fees_usdc_acum, long_ops_acum, short_ops_acum.
  - Muestra en stdout y en el título del PNG: ops long/short y fees acumulados.
- src/dashboard/app.py:
  - Train: parsea métricas de pairs_training (accuracy/f1) y auto‑inyecta --log_results a pairs (pairs_results.csv).
  - AI: añade soporte a spot_mode, guarda columnas ops_long, ops_short, fees_usdc en ai_runs.csv, y hace robusta la construcción de --symbols.
- src/dashboard/templates/ai.html:
  - Añade selector “Modo Spot (solo largos)”.
  - Explica qué es |z| y renombra/explica z_entry, z_exit, prob_thresh y window.

## Cómo ejecutar (PowerShell + Docker Compose)

1) Construir y levantar dashboard
```
docker compose build
docker compose up -d dashboard
# http://localhost:8000
```

2) Entrenar Pares (contenedor fetcher)
```
docker compose run --rm fetcher \
  python -m src.models.pairs_training \
    --symbols BTCUSDC,LTCUSDC \
    --timeframe 1h \
    --format parquet \
    --window 50 \
    --horizon 1 \
    --start_date 2025-07-01 --end_date 2025-09-01 \
    --out src/models/pairs_model.pkl
```

3) Backtest Pairs (market‑neutral)
```
docker compose run --rm fetcher \
  python -m src.backtests.pairs_portfolio_check \
    --symbols BTCUSDC,LTCUSDC \
    --timeframe 1h \
    --format parquet \
    --start_date 2025-07-01 --end_date 2025-09-01 \
    --model_path src/models/pairs_model.pkl \
    --z_entry 2.0 --z_exit 0.5 --prob_thresh 0.6 \
    --window 50 --initial_capital 1000
```

4) Backtest Pairs (spot long‑only en LTC/USDC)
```
docker compose run --rm fetcher \
  python -m src.backtests.pairs_portfolio_check \
    --symbols BTCUSDC,LTCUSDC \
    --timeframe 1h \
    --format parquet \
    --start_date 2025-07-01 --end_date 2025-09-01 \
    --model_path src/models/pairs_model.pkl \
    --z_entry 2.0 --z_exit 0.5 --prob_thresh 0.6 \
    --window 50 --initial_capital 1000 \
    --spot_mode --spot_symbol LTCUSDC
```

5) Live bot (opcional, testnet)
```
docker compose up -d bot
```

## Dashboard
- Train (/train):
  - Entrenador “pairs_training” ahora muestra Score y Positivos; guarda `src/data/training_logs/pairs_results.csv` automáticamente.
- AI (/ai):
  - Modo “Pairs”: añade “Modo Spot (solo largos)” y muestra en historial “Ops” (L/S) y “Fees”.

## Parámetros y definiciones
- window: velas para media y desviación del log-ratio (z-score). En 5m, 50 ⇒ ~4h10m.
- horizon: velas hacia delante para etiquetar convergencia (ej. 3 ⇒ 15m en 5m).
- |z|: |(log_ratio − media_rolling) / desv_rolling|. Desviaciones del ratio vs su media reciente.
- z_entry: umbral de entrada en |z| (cuán extremo debe ser el spread para abrir).
- z_exit: umbral de salida en |z| (vuelta a normalidad para cerrar).
- prob_thresh: probabilidad mínima de convergencia (modelo) para permitir la entrada.

## Fees (estimación aplicada en backtest)
- Compra de LTC con USDC: 0.00000628 LTC por cada 1 USDC gastado (reduce qty al abrir).
- Venta de LTC: 0.080427 USDC por cada 1 LTC vendido (se descuenta al cerrar).
- No se aplican fees en HOLD; sólo en aperturas/cierres.

## Salidas relevantes
- Pairs backtest: `src/data/training_logs/pairs_equity_*.{csv,png}`
  - CSV columnas: value_usdc, z, prob, action, fees_usdc_acum, long_ops_acum, short_ops_acum
- AI runs: `src/data/training_logs/ai_runs.csv` (incluye ops_long, ops_short, fees_usdc en modo pairs)
- Train runs: `src/data/training_logs/runs.csv` (parsea métricas de pairs_training) y `pairs_results.csv`.

## Notas
- Si runs.csv tenía cabecera antigua, el dashboard la archiva en `runs_legacy_<ts>.csv` y recrea con cabecera nueva.
- En live, pares ya se ejecuta gracias al reordenado de multi‑leg en src/main.py.

