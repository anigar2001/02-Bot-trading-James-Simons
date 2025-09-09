"""Backtest sencillo de cartera con señales IA (solo largos spot).

Lee velas locales (parquet/csv) para símbolos USDC (ej. BTCUSDC, ETHUSDC, LTCUSDC),
aplica un clasificador binario (bundle joblib con `features`) para estimar probabilidad
de subida y ejecuta órdenes ficticias sin costes: compra cuando proba>=buy_thresh y
vende cuando proba<=sell_thresh. Registra el valor de cartera en USDC a lo largo del tiempo.

Uso ejemplo (desde contenedor fetcher):

  docker compose run --rm fetcher \
    python -m src.backtests.ai_portfolio_check \
      --symbols BTCUSDC,ETHUSDC,LTCUSDC \
      --timeframe 15m \
      --format parquet \
      --start_date 2024-01-01 --end_date 2025-09-01 \
      --initial_capital 300 \
      --buy_thresh 0.6 --sell_thresh 0.4 \
      --model_path src/models/best_model.pkl

Resultados:
  - CSV: src/data/training_logs/ai_portfolio_<tf>_<start>_<end>.csv
  - PNG: src/data/training_logs/ai_portfolio_<tf>_<start>_<end>.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load


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
        ts = df['timestamp']
        if pd.api.types.is_numeric_dtype(ts):
            idx = pd.to_datetime(ts, unit='ms', utc=True)
        else:
            idx = pd.to_datetime(ts, utc=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df.index = idx
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['open','high','low','close','volume'])
    out = pd.concat(frames, axis=0).sort_index()
    out = out[~out.index.duplicated(keep='last')]
    return out


def to_symbol_key(sym: str) -> str:
    return sym.strip().upper().replace('/', '')


@dataclass
class BundleModel:
    model: object
    features: List[str]


def load_bundle(path: Path) -> BundleModel:
    b = load(path)
    return BundleModel(model=b['model'], features=b['features'])


def build_row_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Replica mínima de features del predictor/model_training para 1 fila.
    Requiere ~200 velas para medias/volatilidades.
    """
    from src.utils.indicators import ema, rsi, adx

    tmp = df.copy().iloc[-300:]
    tmp['ret1'] = tmp['close'].pct_change()
    tmp['ema20'] = ema(tmp['close'], 20)
    tmp['ema50'] = ema(tmp['close'], 50)
    tmp['ema_gap'] = (tmp['ema20'] - tmp['ema50']) / tmp['ema50']
    tmp['rsi14'] = rsi(tmp['close'], 14)
    tmp['adx14'] = adx(tmp['high'], tmp['low'], tmp['close'], 14)
    tmp['vol'] = tmp['close'].pct_change().rolling(20).std()
    tmp['roc3'] = tmp['close'].pct_change(3)
    tmp['roc10'] = tmp['close'].pct_change(10)
    tmp['ema20_slope'] = tmp['ema20'].pct_change()
    ma20 = tmp['close'].rolling(20).mean()
    sd20 = tmp['close'].rolling(20).std()
    tmp['bbp20_2'] = (tmp['close'] - ma20) / (2 * sd20)
    tr = pd.concat([
        (tmp['high'] - tmp['low']).abs(),
        (tmp['high'] - tmp['close'].shift()).abs(),
        (tmp['low'] - tmp['close'].shift()).abs()
    ], axis=1).max(axis=1)
    tmp['atr14n'] = tr.rolling(14).mean() / tmp['close']
    vmean = tmp['volume'].rolling(20).mean()
    vstd = tmp['volume'].rolling(20).std()
    tmp['v_z20'] = (tmp['volume'] - vmean) / vstd
    tmp['vel3'] = tmp['close'].pct_change(3)
    tmp['vol_vel'] = tmp['v_z20'] * tmp['vel3']
    tmp = tmp.dropna()
    if tmp.empty:
        return pd.DataFrame(columns=feature_cols)
    last = tmp.iloc[[-1]]
    return last[feature_cols].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='BTCUSDC,ETHUSDC,LTCUSDC')
    ap.add_argument('--timeframe', default='1h')
    ap.add_argument('--format', choices=['csv','parquet'], default='parquet')
    ap.add_argument('--start_date', default=None)
    ap.add_argument('--end_date', default=None)
    ap.add_argument('--initial_capital', type=float, default=300.0)
    ap.add_argument('--buy_thresh', type=float, default=0.60)
    ap.add_argument('--sell_thresh', type=float, default=0.40)
    ap.add_argument('--model_path', default=None, help='Bundle joblib con {model,features}. Si omites, busca best_model.pkl')
    ap.add_argument('--out_prefix', default='src/data/training_logs/ai_portfolio')
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',') if s.strip()]
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        key = to_symbol_key(sym)
        parts = list_parts(key, args.timeframe, args.format)
        if not parts:
            print(f"[ai-check] no parts for {sym}")
            continue
        df = read_concat(parts, args.format)
        if args.start_date:
            df = df[df.index >= pd.to_datetime(args.start_date, utc=True)]
        if args.end_date:
            df = df[df.index < (pd.to_datetime(args.end_date, utc=True) + pd.Timedelta(days=1))]
        data[sym] = df
    if not data:
        raise SystemExit('[ai-check] no data loaded')

    # Índice conjunto: intersección para sincronía estricta
    common_index = None
    for df in data.values():
        idx = df.index
        common_index = idx if common_index is None else common_index.intersection(idx)
    # Recortar
    for sym in list(data.keys()):
        data[sym] = data[sym].loc[common_index]

    # Cargar modelo
    mdl_path = Path(args.model_path or 'src/models/best_model.pkl')
    if not mdl_path.exists():
        mdl_path = Path('src/models/trained_model.pkl')
    bundle = load_bundle(mdl_path)
    feat_cols = bundle.features

    # Estado de cartera
    cash = float(args.initial_capital)
    qty: Dict[str, float] = {sym: 0.0 for sym in symbols}
    value_rows = []

    # Warmup: empezar cuando cada símbolo tiene suficientes filas para features
    min_pos = max(210, 60)  # aproximación
    times = common_index[min_pos:]

    for ts in times:
        # señales por símbolo
        wants_buy = []
        wants_sell = []
        prices = {}
        for sym in symbols:
            df_sym = data[sym].loc[:ts]
            price = float(df_sym['close'].iloc[-1])
            prices[sym] = price
            X = build_row_features(df_sym, feat_cols)
            if X.empty:
                continue
            proba = float(bundle.model.predict_proba(X.values)[0][1])
            has_pos = qty.get(sym, 0.0) > 0
            if (not has_pos) and proba >= args.buy_thresh:
                wants_buy.append(sym)
            elif has_pos and proba <= args.sell_thresh:
                wants_sell.append(sym)

        # Ejecutar ventas primero (libera cash)
        for sym in wants_sell:
            if qty.get(sym, 0.0) > 0:
                cash += qty[sym] * prices[sym]
                qty[sym] = 0.0

        # Ejecutar compras: dividir cash disponible entre señales nuevas
        n = len([s for s in wants_buy if qty.get(s, 0.0) == 0.0])
        if n > 0 and cash > 0:
            alloc = cash / n
            for sym in wants_buy:
                if qty.get(sym, 0.0) == 0.0 and prices[sym] > 0:
                    spend = min(alloc, cash)
                    buy_q = spend / prices[sym]
                    qty[sym] = qty.get(sym, 0.0) + buy_q
                    cash -= spend

        # Valuación
        port_val = cash + sum(qty[s] * prices.get(s, 0.0) for s in symbols)
        value_rows.append({'timestamp': ts, 'value_usdc': port_val, 'cash': cash, **{f'qty_{s}': qty[s] for s in symbols}})

    if not value_rows:
        raise SystemExit('[ai-check] no samples to evaluate (check dates/timeframe/model)')

    out_dir = Path('src/data/training_logs')
    out_dir.mkdir(parents=True, exist_ok=True)
    start_tag = (args.start_date or (str(times[0].date()) if len(times)>0 else 'start')).replace('-', '')
    end_tag = (args.end_date or (str(times[-1].date()) if len(times)>0 else 'end')).replace('-', '')
    tag = f"{args.timeframe}_{start_tag}_{end_tag}"

    df_val = pd.DataFrame(value_rows).set_index('timestamp')
    csv_path = out_dir / f"ai_portfolio_{tag}.csv"
    df_val.to_csv(csv_path)

    # Plot
    plt.figure(figsize=(10, 4))
    (df_val['value_usdc']).plot()
    plt.title(f"AI Portfolio Value (start={args.initial_capital} USDC)\n{args.symbols} tf={args.timeframe} {args.start_date}->{args.end_date}")
    plt.ylabel('USDC')
    plt.grid(True, alpha=0.3)
    png_path = out_dir / f"ai_portfolio_{tag}.png"
    plt.tight_layout()
    plt.savefig(png_path)
    print(f"[ai-check] saved CSV: {csv_path}")
    print(f"[ai-check] saved PNG: {png_path}")


if __name__ == '__main__':
    main()

