"""Entrenamiento desde particiones locales (parquet/csv) generadas por backfill/resample.

Combina datos de varios símbolos para un timeframe y entrena un RandomForest.
Guarda el modelo en src/models/trained_model.pkl (y copia con sufijo opcional).

Uso (desde contenedor):
  docker compose run --rm fetcher \
    python -m src.models.train_from_partitions --symbols BTCUSDT,ETHUSDT,LTCUSDT --timeframe 1h --format parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.models.model_training import build_features


def list_parts(symbol_key: str, tf: str, fmt: str) -> List[Path]:
    base = Path('src/data/historical/partitioned') / symbol_key / tf
    return sorted(base.glob(f'*.{fmt}'))


def read_concat(parts: List[Path], fmt: str) -> pd.DataFrame:
    frames = []
    for p in parts:
        if fmt == 'csv':
            df = pd.read_csv(p)
        else:
            df = pd.read_parquet(p)
        # timestamp puede venir en ms o iso
        if pd.api.types.is_numeric_dtype(df['timestamp']):
            idx = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        else:
            idx = pd.to_datetime(df['timestamp'], utc=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df.index = idx
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['open','high','low','close','volume'])
    out = pd.concat(frames, axis=0).sort_index()
    out = out[~out.index.duplicated(keep='last')]
    return out


def to_symbol_key(sym: str) -> str:
    s = sym.strip().upper()
    return s.replace('/', '')


def build_dataset(symbols: List[str], timeframe: str, fmt: str, horizon: int = 1) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        key = to_symbol_key(sym)
        parts = list_parts(key, timeframe, fmt)
        if not parts:
            print(f"Aviso: no hay particiones para {sym} {timeframe} ({fmt})")
            continue
        df = read_concat(parts, fmt)
        df_feat = build_features(df, horizon=horizon)
        df_feat['symbol'] = sym
        rows.append(df_feat)
    if not rows:
        raise SystemExit("No se construyó dataset: faltan particiones o están vacías")
    return pd.concat(rows, axis=0).sort_index()


def train_and_save(df_feat: pd.DataFrame, out_path: Path):
    feature_cols = ['ret1', 'ema_gap', 'rsi14', 'adx14', 'vol', 'roc3', 'roc10', 'ema20_slope', 'bbp20_2', 'atr14n', 'v_z20', 'vel3', 'vol_vel']
    X = df_feat[feature_cols].values
    y = df_feat['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=300, random_state=7, max_depth=None, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump({'model': model, 'features': feature_cols}, out_path)
    print(f"Modelo guardado en {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='BTCUSDC,ETHUSDC,LTCUSDC')
    ap.add_argument('--timeframe', default='1h')
    ap.add_argument('--format', choices=['csv', 'parquet'], default='parquet')
    ap.add_argument('--out', default='src/models/trained_model.pkl')
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    df_feat = build_dataset(symbols, args.timeframe, args.format)
    train_and_save(df_feat, Path(args.out))


if __name__ == '__main__':
    main()
