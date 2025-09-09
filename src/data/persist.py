from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _symbol_key(sym: str) -> str:
    """Clave de símbolo para particiones: BTC/USDT -> BTCUSDT, ETHUSDT -> ETHUSDT."""
    s = sym.strip().upper()
    return s.replace('/', '')


def _partition_dir(symbol_key: str, tf: str) -> Path:
    return Path('src/data/historical/partitioned') / symbol_key / tf


def _format() -> str:
    return os.getenv('PERSIST_FORMAT', 'parquet').lower()


def _list_month_files(symbol_key: str, tf: str, fmt: Optional[str] = None) -> List[Path]:
    fmt = fmt or _format()
    d = _partition_dir(symbol_key, tf)
    if not d.exists():
        return []
    return sorted(d.glob(f'*.{fmt}'))


def _read_last_ts(symbol_key: str, tf: str, fmt: Optional[str] = None) -> Optional[pd.Timestamp]:
    fmt = fmt or _format()
    files = _list_month_files(symbol_key, tf, fmt)
    if not files:
        return None
    last = files[-1]
    if fmt == 'csv':
        df = pd.read_csv(last)
        if df.empty:
            return None
        ts = pd.to_datetime(df['timestamp'])
    else:
        df = pd.read_parquet(last)
        if df.empty:
            return None
        ts = pd.to_datetime(df['timestamp'])
    return pd.to_datetime(ts.iloc[-1]).tz_localize('UTC') if ts.dt.tz is None else ts.iloc[-1]


def _month_str(ts: pd.Timestamp) -> str:
    return ts.strftime('%Y-%m')


def _write_partition(symbol_key: str, tf: str, df: pd.DataFrame, fmt: Optional[str] = None):
    fmt = fmt or _format()
    out_dir = _partition_dir(symbol_key, tf)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = df.copy()
    if not isinstance(tmp.index, pd.DatetimeIndex):
        raise ValueError('df must have DatetimeIndex')
    # Garantiza UTC naive en columna timestamp
    idx = tmp.index.tz_convert('UTC') if tmp.index.tz is not None else tmp.index.tz_localize('UTC')
    tmp['timestamp'] = idx.tz_localize(None)
    tmp['month'] = tmp['timestamp'].dt.strftime('%Y-%m')
    for month, chunk in tmp.groupby('month'):
        path = out_dir / f'{month}.{fmt}'
        payload = chunk.drop(columns=['month'])
        if fmt == 'csv':
            if path.exists():
                old = pd.read_csv(path)
                old['timestamp'] = pd.to_datetime(old['timestamp'])
                old = old.set_index('timestamp')
                merged = pd.concat([old, payload.set_index('timestamp')], axis=0)
                merged = merged[~merged.index.duplicated(keep='last')].sort_index()
                merged.reset_index().to_csv(path, index=False)
            else:
                payload.reset_index().to_csv(path, index=False)
        else:
            if path.exists():
                old = pd.read_parquet(path)
                old['timestamp'] = pd.to_datetime(old['timestamp'])
                old = old.set_index('timestamp')
                merged = pd.concat([old, payload.set_index('timestamp')], axis=0)
                merged = merged[~merged.index.duplicated(keep='last')].sort_index()
                merged.reset_index().to_parquet(path, index=False)
            else:
                payload.reset_index().to_parquet(path, index=False)


def append_1m(symbol: str, df_1m: pd.DataFrame):
    """Append de velas 1m cerradas a particiones mensuales.

    Espera df con columnas [open,high,low,close,volume] e índice DatetimeIndex UTC.
    Escribe sólo filas nuevas posteriores a la última guardada.
    """
    symbol_key = _symbol_key(symbol)
    last_ts = _read_last_ts(symbol_key, '1m')

    data = df_1m.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index, utc=True)

    # Usa sólo velas cerradas: excluye la última si es incompleta
    now = pd.Timestamp.utcnow().tz_localize('UTC')
    cutoff = now.floor('T')  # próxima vela en construcción
    data = data[data.index < cutoff]

    if last_ts is not None:
        data = data[data.index > last_ts]
    if data.empty:
        return 0
    _write_partition(symbol_key, '1m', data)
    return len(data)


def _resample(df_1m: pd.DataFrame, target_tf: str) -> pd.DataFrame:
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    rule = {'m': 'min', 'h': 'h', 'd': 'd'}[target_tf[-1]]
    step = int(target_tf[:-1])
    res = df_1m.resample(f'{step}{rule}', label='left', closed='left').agg(agg).dropna()
    return res


def append_resampled_from_1m(symbol: str, df_1m: pd.DataFrame, targets: List[str]):
    """Genera y añade velas agregadas a partir del 1m existente.

    Escribe sólo las velas de target_tf cuyo timestamp sea posterior al último guardado.
    """
    symbol_key = _symbol_key(symbol)
    base = df_1m.copy()
    if not isinstance(base.index, pd.DatetimeIndex):
        base.index = pd.to_datetime(base.index, utc=True)
    # Usa 1m cerradas
    now = pd.Timestamp.utcnow().tz_localize('UTC')
    base = base[base.index < now.floor('T')]

    for tf in targets:
        last_ts = _read_last_ts(symbol_key, tf)
        res = _resample(base, tf)
        if last_ts is not None:
            res = res[res.index > last_ts]
        if res.empty:
            continue
        _write_partition(symbol_key, tf, res)
