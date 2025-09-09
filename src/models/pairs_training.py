"""Entrenamiento de modelo de convergencia para pares (BTCUSDC/LTCUSDC).

Idea: modelar la probabilidad de que el spread/ratio entre LTC y BTC converja (disminuya su |z-score|)
en el próximo horizonte (h velas). Esto ayuda a decidir si abrir arbitraje estadístico.

Uso (desde contenedor):
  docker compose run --rm fetcher \
    python -m src.models.pairs_training \
      --symbols BTCUSDC,LTCUSDC \
      --timeframe 1h \
      --format parquet \
      --horizon 1

Salida: src/models/pairs_model.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


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


def build_pairs_features(df_a: pd.DataFrame, df_b: pd.DataFrame, window: int = 50, horizon: int = 1) -> pd.DataFrame:
    """Construye dataset de pares con ratio y z-score.

    - ratio = close_b / close_a
    - z = (ratio - media_rolling) / std_rolling
    Target: 1 si |z_{t+h}| < |z_t| (convergencia), 0 en caso contrario.
    """
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
    # Dinámica del z-score
    df['z_abs'] = df['z'].abs()
    df['z_lag1'] = df['z'].shift(1)
    df['z_abs_lag1'] = df['z_abs'].shift(1)
    df['z_change'] = df['z'] - df['z_lag1']
    # Volatilidad del ratio
    df['ratio_vol20'] = df['log_ratio'].pct_change().rolling(20).std()
    # Volumen relativo por activo
    a_vm = df['a_vol'].rolling(20).mean()
    a_vs = df['a_vol'].rolling(20).std()
    b_vm = df['b_vol'].rolling(20).mean()
    b_vs = df['b_vol'].rolling(20).std()
    df['a_vz'] = (df['a_vol'] - a_vm) / a_vs
    df['b_vz'] = (df['b_vol'] - b_vm) / b_vs
    # Target: convergencia del z-score a horizonte h
    h = max(1, int(horizon))
    df['target'] = (df['z_abs'].shift(-h) < df['z_abs']).astype(int)
    # Features finales
    feats = df[['z', 'z_abs', 'z_lag1', 'z_abs_lag1', 'z_change', 'ratio_vol20', 'a_vz', 'b_vz', 'target']].dropna().copy()
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='BTCUSDC,LTCUSDC', help='orden importa: A,B')
    ap.add_argument('--timeframe', default='1h')
    ap.add_argument('--format', choices=['csv', 'parquet'], default='parquet')
    ap.add_argument('--window', type=int, default=50)
    ap.add_argument('--horizon', type=int, default=1)
    ap.add_argument('--start_date', default=None, help='YYYY-MM-DD (opcional)')
    ap.add_argument('--end_date', default=None, help='YYYY-MM-DD (opcional)')
    ap.add_argument('--out', default='src/models/pairs_model.pkl')
    ap.add_argument('--log_results', default=None, help='Ruta CSV para guardar resultados de experimento')
    args = ap.parse_args()

    sym_a, sym_b = [s.strip() for s in args.symbols.split(',')[:2]]
    key_a, key_b = to_symbol_key(sym_a), to_symbol_key(sym_b)
    parts_a = list_parts(key_a, args.timeframe, args.format)
    parts_b = list_parts(key_b, args.timeframe, args.format)
    if not parts_a or not parts_b:
        raise SystemExit('No hay particiones suficientes para uno de los símbolos')

    df_a = read_concat(parts_a, args.format)
    df_b = read_concat(parts_b, args.format)
    # Filtrado por rango si se especifica
    if args.start_date:
        sd = pd.to_datetime(args.start_date, utc=True)
        df_a = df_a[df_a.index >= sd]
        df_b = df_b[df_b.index >= sd]
    if args.end_date:
        ed = pd.to_datetime(args.end_date, utc=True) + pd.Timedelta(days=1)
        df_a = df_a[df_a.index < ed]
        df_b = df_b[df_b.index < ed]
    # Alinear por índice (inner join en build_pairs_features)
    feats = build_pairs_features(df_a, df_b, window=args.window, horizon=args.horizon)
    if feats.empty:
        raise SystemExit('Dataset vacío tras construir features de pares')

    feature_cols = ['z', 'z_abs', 'z_lag1', 'z_abs_lag1', 'z_change', 'ratio_vol20', 'a_vz', 'b_vz']
    X = feats[feature_cols].values
    y = feats['target'].values

    # Split temporal 80/20 sin barajar
    n = len(feats)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    base = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf = CalibratedClassifierCV(base, method='sigmoid', cv=3)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    print(classification_report(y_test, y_pred, digits=3))
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    print(f"[pairs] model=lr score={acc:.4f} metric=accuracy f1={f1:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dump({
        'model': clf,
        'features': feature_cols,
        'meta': {
            'type': 'pairs_convergence',
            'symbols': [sym_a, sym_b],
            'timeframe': args.timeframe,
            'window': args.window,
            'horizon': args.horizon,
        }
    }, args.out)
    print(f"Modelo de pares guardado en {args.out}")

    # Log CSV opcional
    if args.log_results:
        from datetime import datetime
        Path(args.log_results).parent.mkdir(parents=True, exist_ok=True)
        header = ['timestamp','symbols','timeframe','window','horizon','accuracy','f1','precision','recall','n_test','positives','pos_pct']
        n_test = int(len(y_test))
        positives = int((y_test == 1).sum())
        pos_pct = f"{(positives / n_test * 100.0):.2f}" if n_test else ''
        row = [
            datetime.utcnow().isoformat(), f"{sym_a},{sym_b}", args.timeframe, args.window, args.horizon,
            f"{acc:.4f}", f"{f1:.4f}", f"{prec:.4f}", f"{rec:.4f}", n_test, positives, pos_pct
        ]
        import csv
        write_header = not Path(args.log_results).exists()
        with open(args.log_results, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)


if __name__ == '__main__':
    main()
