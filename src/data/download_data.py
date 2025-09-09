"""Descargador de datos OHLCV desde Binance Testnet (vía ccxt).

Uso:
  python -m src.data.download_data --symbols BTCUSDT,ETHUSDT,LTCUSDT --timeframe 1h --limit 5000
  
  También puedes indicar una fecha de inicio (prioritaria a limit):
  python -m src.data.download_data --symbols BTCUSDT --timeframe 1h --start 2024-01-01

Notas:
- Acepta símbolos con o sin barra. Convierte a formato ccxt (BTC/USDT) automáticamente.
- Guarda CSV en src/data/historical/{SYMBOL}_{TF}.csv
- Lee BINANCE_API_KEY, BINANCE_API_SECRET, API_BASE del entorno.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone

import pandas as pd

from src.utils.api import BinanceClient


def to_ccxt_symbol(sym: str) -> str:
    s = sym.strip().upper()
    if '/' in s:
        return s
    # Inserta barra antes de USDT, BUSD, USDC si aparece al final
    for q in ("USDT", "BUSD", "USDC", "BTC", "ETH"):
        if s.endswith(q) and len(s) > len(q):
            base = s[: -len(q)]
            return f"{base}/{q}"
    # Fallback: intenta primera mitad/base / segunda
    if len(s) > 4:
        return f"{s[:-4]}/{s[-4:]}"
    return s


def tf_to_ms(tf: str) -> int:
    units = {
        'm': 60_000,
        'h': 3_600_000,
        'd': 86_400_000,
    }
    unit = tf[-1]
    num = int(tf[:-1])
    return num * units[unit]


def parse_start(start: Optional[str]) -> Optional[int]:
    if not start:
        return None
    # Acepta YYYY-MM-DD o ISO completo
    try:
        if len(start) == 10:
            dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(start)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def download_symbol(client: BinanceClient, symbol: str, timeframe: str, limit: int, start: Optional[str] = None):
    ccxt_symbol = to_ccxt_symbol(symbol)
    out_dir = Path("src/data/historical")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ccxt_symbol.replace('/', '_')}_{timeframe}.csv"

    batch = 1000
    df_all = []
    since = parse_start(start)
    remaining = limit
    # Si se especifica start, ignoramos 'limit' y descargamos hasta ahora en bloques
    if since is not None:
        step = batch * tf_to_ms(timeframe)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        while since < now_ms:
            ohlcv = client.exchange.fetch_ohlcv(ccxt_symbol, timeframe=timeframe, limit=batch, since=since)
            if not ohlcv:
                break
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_all.append(df)
            last_ts = int(df['timestamp'].iloc[-1])
            if last_ts <= since:
                since += step
            else:
                since = last_ts + 1
            if len(df) < batch:
                break
    else:
        while remaining > 0:
            this_limit = min(batch, remaining)
            ohlcv = client.exchange.fetch_ohlcv(ccxt_symbol, timeframe=timeframe, limit=this_limit, since=since)
            if not ohlcv:
                break
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_all.append(df)
            since = int(df['timestamp'].iloc[-1]) + 1
            remaining -= len(df)
            if len(df) < this_limit:
                break

    if not df_all:
        print(f"Sin datos para {ccxt_symbol}")
        return

    out = pd.concat(df_all, ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    out.to_csv(out_path, index=False)
    print(f"Guardado {len(out)} velas -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', default='BTCUSDT,ETHUSDT,LTCUSDT')
    parser.add_argument('--timeframe', default='1h')
    parser.add_argument('--limit', type=int, default=5000)
    parser.add_argument('--start', default=None, help='Fecha inicio YYYY-MM-DD o ISO (prioritaria a limit)')
    args = parser.parse_args()

    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    api_base = os.getenv('API_BASE', 'https://testnet.binance.vision')
    client = BinanceClient(api_key=api_key, api_secret=api_secret, api_base=api_base, enable_rate_limit=True, testnet=True, dry_run=True)
    client.load_markets()

    symbols: List[str] = [s.strip() for s in args.symbols.split(',') if s.strip()]
    for s in symbols:
        download_symbol(client, s, args.timeframe, args.limit, start=args.start)


if __name__ == '__main__':
    main()
