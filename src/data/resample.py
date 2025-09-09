"""Resample de OHLCV a timeframes superiores a partir de velas finas.

Lee particiones mensuales (csv/parquet) desde src/data/historical/partitioned/{SYM}/{TF}/
y genera particiones para los nuevos timeframes en la misma estructura.

Uso (desde contenedor fetcher):
  docker compose run --rm fetcher \
    python -m src.data.resample --symbol BTCUSDT --source_tf 1m --targets 5m,15m,1h,4h --format parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def tf_to_rule(tf: str) -> str:
    tf = tf.strip()
    if tf.endswith('m'):
        return f"{int(tf[:-1])}min"
    if tf.endswith('h'):
        return f"{int(tf[:-1])}h"
    if tf.endswith('d'):
        return f"{int(tf[:-1])}d"
    raise ValueError(f"Timeframe no soportado: {tf}")


def list_partitions(base: Path, symbol_key: str, tf: str, fmt: str) -> List[Path]:
    part_dir = base / 'partitioned' / symbol_key / tf
    if not part_dir.exists():
        return []
    return sorted(part_dir.glob(f"*.{fmt}"))


def read_concat(paths: List[Path], fmt: str) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    frames = []
    for p in paths:
        if fmt == 'csv':
            df = pd.read_csv(p)
        else:
            df = pd.read_parquet(p)
        # Acepta timestamp ms o ISO
        if pd.api.types.is_numeric_dtype(df['timestamp']):
            idx = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        else:
            idx = pd.to_datetime(df['timestamp'], utc=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df.index = idx
        frames.append(df)
    out = pd.concat(frames, axis=0).sort_index()
    out = out[~out.index.duplicated(keep='last')]
    return out


def to_symbol_key(sym: str) -> str:
    """Normaliza símbolos a la clave de partición: BTCUSDT o BTC/USDT -> BTCUSDT."""
    s = sym.strip().upper()
    if '/' in s:
        return s.replace('/', '')
    # Mantener tal cual si ya es BTCUSDT
    return s


def ohlcv_resample(df: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    rule = tf_to_rule(target_tf)
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    res = df.resample(rule, label='left', closed='left').agg(agg).dropna()
    res = res[res['volume'] >= 0]  # sanity
    return res


def write_partitioned(df: pd.DataFrame, base: Path, symbol_key: str, tf: str, fmt: str):
    out_dir = base / 'partitioned' / symbol_key / tf
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = df.copy()
    # Asegura índice datetime y columna timestamp sin tz para escritura
    ts_idx = tmp.index
    if getattr(ts_idx, 'tz', None) is not None:
        ts_idx = ts_idx.tz_convert('UTC').tz_localize(None)
    tmp['timestamp'] = ts_idx
    tmp['month'] = tmp['timestamp'].dt.strftime('%Y-%m')
    tmp['timestamp'] = tmp['timestamp'].astype('datetime64[ns]')
    for month, chunk in tmp.groupby('month'):
        path = out_dir / f"{month}.{fmt}"
        chunk = chunk.drop(columns=['month'])
        if fmt == 'csv':
            if path.exists():
                old = pd.read_csv(path)
                old['timestamp'] = pd.to_datetime(old['timestamp'])
                old = old.set_index('timestamp')
                merged = pd.concat([old, chunk.set_index('timestamp')], axis=0)
                merged = merged[~merged.index.duplicated(keep='last')].sort_index()
                merged.reset_index().to_csv(path, index=False)
            else:
                # chunk ya contiene columna 'timestamp'; no hacer reset_index para evitar duplicados
                chunk.to_csv(path, index=False)
        else:
            if path.exists():
                old = pd.read_parquet(path)
                old['timestamp'] = pd.to_datetime(old['timestamp'])
                old = old.set_index('timestamp')
                merged = pd.concat([old, chunk.set_index('timestamp')], axis=0)
                merged = merged[~merged.index.duplicated(keep='last')].sort_index()
                merged.reset_index().to_parquet(path, index=False)
            else:
                # chunk ya contiene columna 'timestamp'; no hacer reset_index para evitar duplicados
                chunk.to_parquet(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True, help='Símbolo: BTCUSDT, ETHUSDT...')
    ap.add_argument('--source_tf', required=True, help='TF de entrada, p.ej. 1m')
    ap.add_argument('--targets', required=True, help='TFs de salida coma-separados, p.ej. 5m,15m,1h,4h')
    ap.add_argument('--format', choices=['csv', 'parquet'], default='parquet')
    args = ap.parse_args()

    base = Path('src/data/historical')
    symbol_key = to_symbol_key(args.symbol)

    # Cargar todo el source particionado
    paths = list_partitions(base, symbol_key, args.source_tf, args.format)
    if not paths:
        raise SystemExit(f"No se encontró fuente particionada para {args.symbol} {args.source_tf} en {args.format}")
    df = read_concat(paths, args.format)
    if df.empty:
        raise SystemExit("Fuente vacía tras lectura")

    targets = [t.strip() for t in args.targets.split(',') if t.strip()]
    for tf in targets:
        out = ohlcv_resample(df, tf)
        write_partitioned(out, base, symbol_key, tf, args.format)
        print(f"Resample {args.symbol} {args.source_tf} -> {tf}: {len(out)} filas (particionado)")


if __name__ == '__main__':
    main()
