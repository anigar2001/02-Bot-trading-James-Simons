"""Ingesta de velas desde archivos diarios ZIP de Binance Vision (spot klines).

Descarga directa por día sin necesidad de API ni registro:
  https://data.binance.vision/data/spot/daily/klines/{SYMBOL}/{TF}/{SYMBOL}-{TF}-{YYYY-MM-DD}.zip

Este script:
- Itera un rango de fechas (por día) para uno o varios símbolos y timeframe.
- Lee el CSV dentro del ZIP en memoria (sin guardarlo) y extrae columnas OHLCV.
- Escribe en particiones mensuales (csv o parquet) en src/data/historical/partitioned/{SYM}/{TF}/YYYY-MM.{ext}
- Deduplica por timestamp al escribir (puede relanzarse sin duplicados).

Uso (desde contenedor 'fetcher'):
  docker compose run --rm fetcher \
    python -m src.data.ingest_binance_archive \
      --symbols BTCUSDT,ETHUSDT,LTCUSDT --timeframe 1m \
      --start 2018-01-01 --end 2018-12-31 --format parquet
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests


BASE_URL = os.getenv('BINANCE_VISION_BASE', 'https://data.binance.vision/data/spot/daily/klines')


def daterange(start: datetime, end: datetime) -> Iterable[datetime]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def symbol_key(sym: str) -> str:
    s = sym.strip().upper()
    return s.replace('/', '_') if '/' in s else s


def out_dir_for(sym_key: str, timeframe: str) -> Path:
    return Path('src/data/historical/partitioned') / sym_key / timeframe


def write_month_partition(df: pd.DataFrame, sym_key: str, timeframe: str, fmt: str):
    out_dir = out_dir_for(sym_key, timeframe)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    # Asegura columna timestamp datetime sin tz
    if pd.api.types.is_numeric_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df['month'] = df['timestamp'].dt.strftime('%Y-%m')
    for month, chunk in df.groupby('month'):
        path = out_dir / f"{month}.{fmt}"
        payload = chunk.drop(columns=['month']).sort_values('timestamp')
        if path.exists():
            if fmt == 'csv':
                old = pd.read_csv(path)
            else:
                old = pd.read_parquet(path)
            merged = pd.concat([old, payload], ignore_index=True)
            merged = merged.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        else:
            merged = payload
        if fmt == 'csv':
            merged.to_csv(path, index=False)
        else:
            merged.to_parquet(path, index=False)


def parse_csv_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ajusta columnas a OHLCV estándar: timestamp, open, high, low, close, volume.

    Binance klines CSV típicamente tiene 12 columnas:
    [ open_time, open, high, low, close, volume, close_time, quote_asset_volume,
      number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore ]
    """
    cols = list(df.columns)
    if len(cols) >= 6:
        # Detectar y saltar cabecera textual si existe
        first_cell = df.iloc[0, 0]
        if isinstance(first_cell, str) and any(k in first_cell.lower() for k in ('open_time', 'opentime')):
            df = df.iloc[1:].reset_index(drop=True)
        ts = pd.to_numeric(df.iloc[:, 0], errors='coerce')
        o = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        h = pd.to_numeric(df.iloc[:, 2], errors='coerce')
        l = pd.to_numeric(df.iloc[:, 3], errors='coerce')
        c = pd.to_numeric(df.iloc[:, 4], errors='coerce')
        v = pd.to_numeric(df.iloc[:, 5], errors='coerce')

        # Normalizar época: los CSV de Binance Vision pueden traer open_time en microsegundos.
        # Queremos milisegundos. Detectamos escala por orden de magnitud.
        ts_clean = ts.dropna()
        if not ts_clean.empty:
            med = float(ts_clean.median())
            if med > 1e17:  # nanosegundos
                ts = (ts // 1_000_000).astype('Int64')
            elif med > 1e14:  # microsegundos
                ts = (ts // 1_000).astype('Int64')
            else:  # milisegundos (o segundos)
                if med < 1e12:  # segundos -> a ms
                    ts = (ts * 1000).astype('Int64')
                else:
                    ts = ts.astype('Int64')

        out = pd.DataFrame({'timestamp': ts.astype('int64', errors='ignore'), 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
        out = out.dropna()
        # Sanidad del timestamp (ms epoch razonable: >= 2000-01-01 y <= ahora+2d)
        now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
        min_ms = 946684800000  # 2000-01-01
        max_ms = now_ms + 2 * 24 * 3600 * 1000
        out = out[(out['timestamp'] >= min_ms) & (out['timestamp'] <= max_ms)]
        # Cast a int64 seguro
        out['timestamp'] = out['timestamp'].astype('int64')
        # Filtrar OHLCV no válidos
        out = out[(out['high'] >= out['low']) & (out['open'] >= 0) & (out['close'] >= 0) & (out['volume'] >= 0)]
        return out
    raise ValueError('CSV con columnas inesperadas')


def fetch_day(symbol: str, timeframe: str, day: datetime) -> Optional[pd.DataFrame]:
    date_str = day.strftime('%Y-%m-%d')
    url = f"{BASE_URL}/{symbol}/{timeframe}/{symbol}-{timeframe}-{date_str}.zip"
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        # Buscar el primer CSV
        csv_names = [n for n in zf.namelist() if n.lower().endswith('.csv')]
        if not csv_names:
            return None
        with zf.open(csv_names[0]) as f:
            # Lee CSV sin asumir encabezado
            df = pd.read_csv(f, header=None)
    return parse_csv_df(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', required=True, help='BTCUSDT,ETHUSDT,LTCUSDT')
    ap.add_argument('--timeframe', default='1m', help='Timeframe (ej. 1m)')
    ap.add_argument('--start', default='2018-01-01', help='YYYY-MM-DD')
    ap.add_argument('--end', default=None, help='YYYY-MM-DD (si omites, hoy)')
    ap.add_argument('--format', choices=['csv', 'parquet'], default='parquet')
    ap.add_argument('--alias_to_usdc', action='store_true', help='Guardar en carpetas *USDC cuando el símbolo fuente es *USDT')
    args = ap.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc)

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    tf = args.timeframe

    for sym in symbols:
        sym_key = symbol_key(sym)
        if args.alias_to_usdc and sym_key.endswith('USDT'):
            sym_key = sym_key[:-4] + 'USDC'
        print(f"== Ingesta {sym} {tf} desde {start.date()} hasta {end.date()} ==")
        for day in daterange(start, end):
            try:
                df = fetch_day(sym, tf, day)
                if df is None or df.empty:
                    # Día ausente en el repositorio
                    continue
                write_month_partition(df, sym_key, tf, fmt=args.format)
                print(f"{sym} {tf} {day.date()}: +{len(df)} filas")
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 404:
                    continue
                print(f"HTTP error {sym} {day.date()}: {e}")
            except Exception as e:
                print(f"Error {sym} {day.date()}: {e}")


if __name__ == '__main__':
    main()
