"""Backtest sencillo de cartera de pares usando un modelo de convergencia (pairs_model.pkl).

Simula trades market-neutral (long/short) cuando |z|>z_entry y la probabilidad de convergencia
estimada por el modelo es >= prob_thresh. Cierra cuando |z|<z_exit o tras max_hold velas.

Genera CSV y PNG de la curva de equity (como AI Portfolio Check).

Uso:
  docker compose run --rm fetcher \
    python -m src.backtests.pairs_portfolio_check \
      --symbols BTCUSDC,LTCUSDC \
      --timeframe 1h --format parquet \
      --start_date 2024-01-01 --end_date 2025-09-01 \
      --model_path src/models/pairs_model.pkl \
      --z_entry 2.0 --z_exit 0.5 --prob_thresh 0.6 --window 50 --initial_capital 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

# Fees estimados para LTC (aprox)
LTC_BUY_FEE_PER_USDC = 0.00000628  # fee en LTC por cada 1 USDC gastado al comprar LTC
LTC_SELL_FEE_USDC_PER_LTC = 0.080427  # fee en USDC por cada 1 LTC vendido


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
        idx = pd.to_datetime(ts, utc=True) if not pd.api.types.is_numeric_dtype(ts) else pd.to_datetime(ts, unit='ms', utc=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df.index = idx
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['open','high','low','close','volume'])
    out = pd.concat(frames, axis=0).sort_index()
    out = out[~out.index.duplicated(keep='last')]
    return out


def build_pairs_features(df_a: pd.DataFrame, df_b: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    df = pd.concat([
        df_a['close'].rename('a'),
        df_b['close'].rename('b'),
        df_a['volume'].rename('a_vol'),
        df_b['volume'].rename('b_vol'),
    ], axis=1).dropna()
    if df.empty:
        return df
    df['ratio'] = df['b'] / df['a']
    df['log_ratio'] = np.log(df['ratio'])
    mu = df['log_ratio'].rolling(window).mean()
    sd = df['log_ratio'].rolling(window).std()
    df['z'] = (df['log_ratio'] - mu) / sd
    df['z_abs'] = df['z'].abs()
    df['z_lag1'] = df['z'].shift(1)
    df['z_abs_lag1'] = df['z_abs'].shift(1)
    df['z_change'] = df['z'] - df['z_lag1']
    df['ratio_vol20'] = df['log_ratio'].pct_change().rolling(20).std()
    a_vm = df['a_vol'].rolling(20).mean(); a_vs = df['a_vol'].rolling(20).std()
    b_vm = df['b_vol'].rolling(20).mean(); b_vs = df['b_vol'].rolling(20).std()
    df['a_vz'] = (df['a_vol'] - a_vm) / a_vs
    df['b_vz'] = (df['b_vol'] - b_vm) / b_vs
    return df.dropna().copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', required=True, help='A,B sin barra (ej. BTCUSDC,LTCUSDC)')
    ap.add_argument('--timeframe', default='1h')
    ap.add_argument('--format', choices=['csv','parquet'], default='parquet')
    ap.add_argument('--start_date', default=None)
    ap.add_argument('--end_date', default=None)
    ap.add_argument('--initial_capital', type=float, default=1000.0)
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--z_entry', type=float, default=2.0)
    ap.add_argument('--z_exit', type=float, default=0.5)
    ap.add_argument('--prob_thresh', type=float, default=0.6)
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--max_hold', type=int, default=240)
    ap.add_argument('--out_prefix', default='src/data/training_logs/pairs_portfolio')
    # Modo spot: sólo long en un símbolo (por defecto, LTCUSDC)
    ap.add_argument('--spot_mode', action='store_true', help='Simular sólo compras/ventas spot del símbolo elegido')
    ap.add_argument('--spot_symbol', default='LTCUSDC', help='Símbolo spot a usar en modo spot (default: LTCUSDC)')
    args = ap.parse_args()

    sym_a, sym_b = [s.strip().upper() for s in args.symbols.split(',')[:2]]
    pa = list_parts(sym_a, args.timeframe, args.format)
    pb = list_parts(sym_b, args.timeframe, args.format)
    if not pa or not pb:
        raise SystemExit('[pairs-portfolio] faltan particiones para A o B')
    df_a = read_concat(pa, args.format)
    df_b = read_concat(pb, args.format)
    if args.start_date:
        sd = pd.to_datetime(args.start_date, utc=True); df_a = df_a[df_a.index >= sd]; df_b = df_b[df_b.index >= sd]
    if args.end_date:
        ed = pd.to_datetime(args.end_date, utc=True) + pd.Timedelta(days=1); df_a = df_a[df_a.index < ed]; df_b = df_b[df_b.index < ed]

    feats = build_pairs_features(df_a, df_b, window=args.window)
    if feats.empty:
        raise SystemExit('[pairs-portfolio] sin datos tras construir features')

    # Cargar modelo y columnas
    bundle = load(args.model_path)
    model = bundle['model']; feature_cols = bundle['features']

    # Predecir prob converge por fila
    X_all = feats[feature_cols].copy()
    # Evitar warning de sklearn pasando arrays sin nombres de columnas
    prob = model.predict_proba(X_all.values)[:, 1]
    feats = feats.iloc[:len(prob)].copy()
    feats['prob'] = prob

    # Simulación de equity con pares market-neutral o spot long-only
    equity = []
    action = []
    z_vals = []
    prob_vals = []
    capital0 = float(args.initial_capital)
    position_open = False
    hold = 0
    # Datos para pnl
    entry_a = entry_b = 0.0
    units_long = 0.0
    units_short = 0.0
    long_is_b = False
    # Spot mode tracking
    spot_qty = 0.0
    # Métricas
    n_long = 0
    n_short = 0
    total_fees_usdc = 0.0
    fees_series = []
    long_series = []
    short_series = []

    for ts in feats.index:
        z = float(feats.at[ts, 'z'])
        p = float(feats.at[ts, 'prob'])
        z_vals.append(z)
        prob_vals.append(p)
        # Precios actuales
        price_a = float(df_a.loc[ts, 'close'])
        price_b = float(df_b.loc[ts, 'close'])

        # Modo spot: sólo operar largo en spot_symbol cuando la lógica indicaría long B
        if args.spot_mode:
            is_spot_b = args.spot_symbol.upper() == sym_b
            px_spot = price_b if is_spot_b else price_a
            # Entrada spot si: señal de entrada y sería long sobre ese símbolo
            desire_long_b = (z < -args.z_entry)
            should_open = (not position_open) and p >= args.prob_thresh and (
                (is_spot_b and desire_long_b) or ((not is_spot_b) and (z > args.z_entry))
            )
            act = 'HOLD'
            if should_open:
                spend = capital0
                spot_qty = spend / px_spot if px_spot > 0 else 0.0
                # Fee de compra si el símbolo spot es LTC
                if 'LTC' in args.spot_symbol.upper():
                    fee_qty = spend * LTC_BUY_FEE_PER_USDC
                    spot_qty = max(0.0, spot_qty - fee_qty)
                entry_b = px_spot
                position_open = True
                hold = 0
                act = 'OPEN'
                n_long += 1
            elif position_open:
                hold += 1
                pnl = spot_qty * (px_spot - entry_b)
                if abs(z) < args.z_exit or hold >= args.max_hold:
                    # Vender todo y registrar valor post-cierre
                    position_open = False
                    if 'LTC' in args.spot_symbol.upper():
                        fee_usdc = spot_qty * LTC_SELL_FEE_USDC_PER_LTC
                        total_fees_usdc += float(fee_usdc)
                        capital0 += pnl - float(fee_usdc)
                    else:
                        capital0 += pnl
                    spot_qty = 0.0
                    entry_b = 0.0
                    equity.append(capital0)
                    action.append('CLOSE')
                else:
                    equity.append(capital0 + pnl)
                    action.append('IN_POS')
                fees_series.append(total_fees_usdc)
                long_series.append(n_long)
                short_series.append(n_short)
                continue
            # Sin posición o justo abierta: equity es capital inicial
            equity.append(capital0)
            fees_series.append(total_fees_usdc)
            long_series.append(n_long)
            short_series.append(n_short)
            action.append(act)
            continue

        if not position_open:
            act = 'HOLD'
            # Evaluar apertura
            if abs(z) > args.z_entry and p >= args.prob_thresh:
                notional = capital0  # usa todo para el diferencial (simulación)
                long_is_b = (z < 0)
                if long_is_b:
                    # long B, short A
                    units_long = (notional / 2) / price_b
                    # Fee de compra LTC en apertura reduce cantidad
                    fee_qty = (notional / 2) * LTC_BUY_FEE_PER_USDC
                    units_long = max(0.0, units_long - fee_qty)
                    units_short = (notional / 2) / price_a
                else:
                    # long A, short B
                    units_long = (notional / 2) / price_a
                    units_short = (notional / 2) / price_b
                entry_a = price_a
                entry_b = price_b
                position_open = True
                hold = 0
                act = 'OPEN'
                n_long += 1; n_short += 1
            # Equity sin posición = capital inicial
            equity.append(capital0)
            fees_series.append(total_fees_usdc)
            long_series.append(n_long)
            short_series.append(n_short)
            action.append(act)
        else:
            # En posición: calcular PnL y equity
            hold += 1
            if long_is_b:
                pnl = units_long * (price_b - entry_b) + units_short * (entry_a - price_a)
            else:
                pnl = units_long * (price_a - entry_a) + units_short * (entry_b - price_b)
            equity.append(capital0 + pnl)
            fees_series.append(total_fees_usdc)
            long_series.append(n_long)
            short_series.append(n_short)
            # Evaluar cierre
            if abs(z) < args.z_exit or hold >= args.max_hold:
                position_open = False
                # Acumular PnL realizado al capital menos fees de cierre en el lado LTC
                if long_is_b:
                    # cerramos vendiendo LTC
                    fee_usdc = units_long * LTC_SELL_FEE_USDC_PER_LTC
                    total_fees_usdc += float(fee_usdc)
                    capital0 += pnl - float(fee_usdc)
                else:
                    # cerramos short LTC comprando LTC (fee de compra en LTC convertido a USDC)
                    buy_usdc = units_short * price_b
                    fee_qty = buy_usdc * LTC_BUY_FEE_PER_USDC
                    fee_usdc = fee_qty * price_b
                    total_fees_usdc += float(fee_usdc)
                    capital0 += pnl - float(fee_usdc)
                units_long = units_short = 0.0
                entry_a = entry_b = 0.0
                action.append('CLOSE')
            else:
                action.append('IN_POS')

    # Alinear índice
    idx = feats.index
    # Asegurar longitudes iguales
    n = len(idx)
    df_out = pd.DataFrame({
        'value_usdc': equity[:n],
        'z': z_vals[:n],
        'prob': prob_vals[:n],
        'action': action[:n],
        'fees_usdc_acum': fees_series[:n],
        'long_ops_acum': long_series[:n],
        'short_ops_acum': short_series[:n],
    }, index=idx)
    out_dir = Path('src/data/training_logs'); out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{sym_a}_{sym_b}_{args.timeframe}_{args.z_entry}_{args.z_exit}_{args.prob_thresh}"
    csv_path = out_dir / f"pairs_equity_{tag}.csv"; df_out.to_csv(csv_path)
    plt.figure(figsize=(10,4)); df_out['value_usdc'].plot(); plt.ylabel('USDC'); plt.grid(True, alpha=0.3)
    plt.title(f"Pairs Portfolio {sym_a}/{sym_b} tf={args.timeframe}\nops long={n_long} short={n_short} fees_usdc={total_fees_usdc:.4f}")
    plt.tight_layout(); png_path = out_dir / f"pairs_equity_{tag}.png"; plt.savefig(png_path)
    print(f"[pairs-portfolio] ops long={n_long} short={n_short} fees_usdc={total_fees_usdc:.4f}")
    print(f"[pairs-portfolio] saved CSV: {csv_path}")
    print(f"[pairs-portfolio] saved PNG: {png_path}")


if __name__ == '__main__':
    main()
