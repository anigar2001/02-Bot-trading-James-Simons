#!/bin/sh
set -e

: "${TRAINER_SYMBOLS:=BTCUSDT,ETHUSDT,LTCUSDT}"
: "${TRAINER_TF:=1h}"
: "${TRAINER_FORMAT:=parquet}"
: "${TRAINER_INTERVAL_SECONDS:=86400}"

echo "[trainer] Starting retrain loop: symbols=${TRAINER_SYMBOLS} tf=${TRAINER_TF} format=${TRAINER_FORMAT} interval=${TRAINER_INTERVAL_SECONDS}s"

while true; do
  echo "[trainer] $(date -u) training..."
  python -m src.models.train_from_partitions --symbols "${TRAINER_SYMBOLS}" --timeframe "${TRAINER_TF}" --format "${TRAINER_FORMAT}" || true
  echo "[trainer] $(date -u) sleeping ${TRAINER_INTERVAL_SECONDS}s"
  sleep "${TRAINER_INTERVAL_SECONDS}"
done

