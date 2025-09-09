"""Backfill robusto de OHLCV desde Binance Testnet (ccxt) con partición mensual.

Objetivo:
- Traer grandes rangos de velas en bloques y persistir en archivos particionados
  por mes para evitar CSV gigantes.
- Re-lanzable (resume): si ya existe parte del mes, se hace merge y se deduplica.

Uso típico (desde contenedor 'fetcher'):

  docker compose run --rm fetcher \
    python -m src.data.backfill --symbols BTCUSDT,ETHUSDT,LTCUSDT \
    --timeframes 1m,5m,15m,1h,4h --start 2024-01-01

Parámetros clave:
- --start y --end (YYYY-MM-DD o ISO). Si no das --end, usa ahora.
- --format csv|parquet (csv por defecto). Parquet recomendado para tamaño/rendimiento.
- --partition monthly (única opción por ahora) -> genera: src/data/historical/partitioned/{SYM}/{TF}/YYYY-MM.{ext}
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.utils.api import BinanceClient


def parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        if len(s) == 10:
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def tf_to_ms(tf: str) -> int:
    units = {'m': 60_000, 'h': 3_600_000, 'd': 86_400_000}
    return int(tf[:-1]) * units[tf[-1]]


def to_ccxt_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if '/' in s:
        return s
    for q in ("USDT", "BUSD", "USDC", "BTC", "ETH"):
        if s.endswith(q) and len(s) > len(q):
            base = s[: -len(q)]
            return f"{base}/{q}"
    return s


def to_symbol_key(sym: str) -> str:
    """Clave de símbolo para particiones: BTCUSDT o BTC/USDT -> BTCUSDT."""
    s = sym.strip().upper()
    return s.replace('/', '')


def month_key(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_partition(df: pd.DataFrame, out_dir: Path, symbol_key: str, tf: str, fmt: str):
    # Agrupar por mes y escribir/mergear
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['month'] = df['timestamp'].dt.strftime('%Y-%m')
    for month, chunk in df.groupby('month'):
        part_dir = out_dir / 'partitioned' / symbol_key / tf
        ensure_dir(part_dir)
        path = part_dir / f"{month}.{fmt}"
        chunk = chunk.drop(columns=['month']).sort_values('timestamp')
        if path.exists():
            if fmt == 'csv':
                old = pd.read_csv(path)
                old['timestamp'] = pd.to_datetime(old['timestamp'])
            else:
                old = pd.read_parquet(path)
            merged = pd.concat([old, chunk], ignore_index=True)
            merged = merged.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        else:
            merged = chunk
        if fmt == 'csv':
            merged.to_csv(path, index=False)
        else:
            merged.to_parquet(path, index=False)


def list_partitions(base: Path, symbol_key: str, tf: str, fmt: str) -> List[Path]:
    part_dir = base / 'partitioned' / symbol_key / tf
    if not part_dir.exists():
        return []
    return sorted(part_dir.glob(f'*.{fmt}'))


def read_last_saved_ts(base: Path, symbol_key: str, tf: str, fmt: str) -> Optional[int]:
    parts = list_partitions(base, symbol_key, tf, fmt)
    if not parts:
        return None
    last = parts[-1]
    try:
        if fmt == 'csv':
            df = pd.read_csv(last)
        else:
            df = pd.read_parquet(last)
        if df.empty:
            return None
        ts = pd.to_datetime(df['timestamp'])
        return int(pd.to_datetime(ts.iloc[-1]).timestamp() * 1000)
    except Exception:
        return None


def backfill_symbol(client: BinanceClient, symbol: str, tf: str, start: Optional[datetime], end: Optional[datetime], fmt: str = 'csv', batch: int = 1000, pause: float = 0.2):
    ccxt_symbol = to_ccxt_symbol(symbol)
    out_dir = Path('src/data/historical')
    ensure_dir(out_dir)

    since_ms = int(start.timestamp() * 1000) if start else None
    end_ms = int((end or datetime.now(timezone.utc)).timestamp() * 1000)
    tf_ms = tf_to_ms(tf)

    last_written_ms = None
    total = 0
    while True:
        try:
            ohlcv = client.exchange.fetch_ohlcv(ccxt_symbol, timeframe=tf, limit=batch, since=since_ms)
        except Exception as e:
            # Reintento simple
            time.sleep(1.0)
            try:
                ohlcv = client.exchange.fetch_ohlcv(ccxt_symbol, timeframe=tf, limit=batch, since=since_ms)
            except Exception:
                print(f"ERROR fetch {symbol} {tf} since={since_ms}: {e}")
                break

        if not ohlcv:
            break
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Cortar si excede end_ms
        if end_ms is not None:
            df = df[df['timestamp'] <= end_ms]
        if df.empty:
            break

        # Persistir por mes
        write_partition(df.copy(), out_dir, to_symbol_key(symbol), tf, fmt)

        total += len(df)
        last_ts = int(df['timestamp'].iloc[-1])
        # Avanza since
        if since_ms is None:
            since_ms = last_ts + tf_ms
        else:
            since_ms = max(since_ms + len(df) * tf_ms, last_ts + tf_ms)
        # Si ya llegamos al final
        if end_ms is not None and since_ms >= end_ms:
            break
        time.sleep(pause)

    print(f"Backfill {symbol} {tf}: {total} velas escritas en partitioned/")


def default_horizons_days(tf: str) -> int:
    return {
        '1m': 90,   # ~129,600 velas (3 meses)
        '5m': 365,  # ~105k
        '15m': 365 * 2,
        '1h': 365 * 3,
        '4h': 365 * 5,
        '1d': 365 * 8,
    }.get(tf, 365)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', required=True, help='Lista de símbolos coma-separados, p.ej. BTCUSDT,ETHUSDT')
    ap.add_argument('--timeframes', required=True, help='Lista de timeframes coma-separados, p.ej. 1m,5m,1h')
    ap.add_argument('--start', default=None, help='YYYY-MM-DD o ISO (si omites, usa horizonte por defecto por TF)')
    ap.add_argument('--end', default=None, help='YYYY-MM-DD o ISO (si omites, ahora)')
    ap.add_argument('--format', choices=['csv', 'parquet'], default='csv')
    ap.add_argument('--batch', type=int, default=1000)
    ap.add_argument('--pause', type=float, default=0.2)
    args = ap.parse_args()

    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    api_base = os.getenv('API_BASE', 'https://testnet.binance.vision')
    client = BinanceClient(api_key=api_key, api_secret=api_secret, api_base=api_base, enable_rate_limit=True, testnet=True, dry_run=True)
    client.load_markets()

    syms = [s.strip() for s in args.symbols.split(',') if s.strip()]
    tfs = [t.strip() for t in args.timeframes.split(',') if t.strip()]

    start_dt = parse_dt(args.start)
    end_dt = parse_dt(args.end) or datetime.now(timezone.utc)

    for s in syms:
        for tf in tfs:
            base = Path('src/data/historical')
            symbol_key = to_symbol_key(s)
            # Reanudar desde la última vela guardada si existe
            last_ms = read_last_saved_ts(base, symbol_key, tf, args.format)
            if last_ms is not None:
                sdt = datetime.fromtimestamp(last_ms / 1000.0, tz=timezone.utc) + timedelta(milliseconds=tf_to_ms(tf))
            else:
                sdt = start_dt
                if sdt is None:
                    days = default_horizons_days(tf)
                    sdt = end_dt - timedelta(days=days)
            backfill_symbol(client, s, tf, start=sdt, end=end_dt, fmt=args.format, batch=args.batch, pause=args.pause)


if __name__ == '__main__':
    main()
