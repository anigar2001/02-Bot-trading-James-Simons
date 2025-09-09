"""Entrenamiento walk-forward con varios modelos y calibración de probabilidades.

Uso (desde contenedor fetcher):

  docker compose run --rm fetcher \
    python -m src.models.walkforward_training \
      --symbols BTCUSDC,ETHUSDC,LTCUSDC \
      --timeframe 1h \
      --format parquet \
      --models lr,xgb,lgbm \
      --folds 5

Genera:
  - src/models/trained_lr.pkl, trained_xgb.pkl, trained_lgbm.pkl
  - src/models/best_model.pkl (según accuracy media en folds)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:
    LGBMClassifier = None  # type: ignore

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


def build_dataset(symbols: List[str], timeframe: str, fmt: str, horizon: int = 1) -> pd.DataFrame:
    rows = []
    for sym in symbols:
        key = to_symbol_key(sym)
        parts = list_parts(key, timeframe, fmt)
        if not parts:
            print(f"[warn] no parts for {sym} {timeframe} {fmt}")
            continue
        df = read_concat(parts, fmt)
        if df.empty:
            continue
        feats = build_features(df, horizon=horizon)
        feats['symbol'] = sym
        rows.append(feats)
    if not rows:
        raise SystemExit("no data after reading partitions")
    return pd.concat(rows, axis=0).sort_index()


def time_folds(n: int, k: int) -> List[Tuple[int, int]]:
    """Devuelve k folds de validación, cada uno con un bloque del 1/k final del índice.
    Train = [0:idx_start), Test = [idx_start:idx_end)
    """
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else n
        folds.append((start, end))
    # para walk-forward: entrenar hasta start, validar en [start:end)
    return folds[1:]  # el primer bloque no tiene entrenamiento previo suficiente


def train_lr(X: np.ndarray, y: np.ndarray):
    lr = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf = CalibratedClassifierCV(lr, method='sigmoid', cv=3)
    clf.fit(X, y)
    return clf


def train_xgb(X: np.ndarray, y: np.ndarray):
    if XGBClassifier is None:
        return None
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        n_jobs=-1,
        reg_lambda=1.0,
        eval_metric='logloss',
    )
    xgb.fit(X, y)
    return xgb


def train_lgbm(X: np.ndarray, y: np.ndarray):
    if LGBMClassifier is None:
        return None
    lgbm = LGBMClassifier(
        n_estimators=600,
        max_depth=-1,
        num_leaves=63,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary',
        n_jobs=-1,
    )
    lgbm.fit(X, y)
    return lgbm


def evaluate_walkforward(df_feat: pd.DataFrame, models: List[str], k: int = 5) -> Dict[str, float]:
    feature_cols = ['ret1', 'ema_gap', 'rsi14', 'adx14', 'vol', 'roc3', 'roc10', 'ema20_slope', 'bbp20_2', 'atr14n', 'v_z20', 'vel3', 'vol_vel']
    X_all = df_feat[feature_cols].values
    y_all = df_feat['target'].values
    n = len(df_feat)
    folds = time_folds(n, k)
    scores = {m: [] for m in models}

    for (start, end) in folds:
        X_train, y_train = X_all[:start, :], y_all[:start]
        X_val, y_val = X_all[start:end, :], y_all[start:end]
        if len(y_train) == 0 or len(y_val) == 0:
            continue
        if 'lr' in models:
            clf = train_lr(X_train, y_train)
            y_hat = (clf.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
            scores['lr'].append(accuracy_score(y_val, y_hat))
        if 'xgb' in models:
            clf = train_xgb(X_train, y_train)
            if clf is not None:
                y_hat = (clf.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
                scores['xgb'].append(accuracy_score(y_val, y_hat))
        if 'lgbm' in models:
            clf = train_lgbm(X_train, y_train)
            if clf is not None:
                y_hat = (clf.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
                scores['lgbm'].append(accuracy_score(y_val, y_hat))

    return {m: float(np.mean(v)) if v else 0.0 for m, v in scores.items()}


def train_full_and_save(df_feat: pd.DataFrame, models: List[str]):
    feature_cols = ['ret1', 'ema_gap', 'rsi14', 'adx14', 'vol', 'roc3', 'roc10', 'ema20_slope', 'bbp20_2', 'atr14n', 'v_z20', 'vel3', 'vol_vel']
    X_all = df_feat[feature_cols].values
    y_all = df_feat['target'].values
    results = {}
    out_dir = Path('src/models')
    out_dir.mkdir(parents=True, exist_ok=True)

    if 'lr' in models:
        clf = train_lr(X_all, y_all)
        path = out_dir / 'trained_lr.pkl'
        dump({'model': clf, 'features': feature_cols}, path)
        results['lr'] = str(path)
    if 'xgb' in models and XGBClassifier is not None:
        clf = train_xgb(X_all, y_all)
        path = out_dir / 'trained_xgb.pkl'
        dump({'model': clf, 'features': feature_cols}, path)
        results['xgb'] = str(path)
    if 'lgbm' in models and LGBMClassifier is not None:
        clf = train_lgbm(X_all, y_all)
        path = out_dir / 'trained_lgbm.pkl'
        dump({'model': clf, 'features': feature_cols}, path)
        results['lgbm'] = str(path)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='BTCUSDC,ETHUSDC,LTCUSDC')
    ap.add_argument('--timeframe', default='1h')
    ap.add_argument('--format', choices=['csv', 'parquet'], default='parquet')
    ap.add_argument('--models', default='lr,xgb,lgbm')
    ap.add_argument('--horizon', type=int, default=1, help='Velas hacia delante para el target (ej. 3 en 5m~15m)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--log_results', default=None, help='Ruta CSV para guardar resultados de experimento')
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    models = [m.strip() for m in args.models.split(',') if m.strip()]

    print(f"[wf] building dataset symbols={symbols} tf={args.timeframe} fmt={args.format}")
    df_feat = build_dataset(symbols, args.timeframe, args.format, horizon=args.horizon)
    print(f"[wf] samples={len(df_feat)}")

    print(f"[wf] walk-forward {args.folds} folds, models={models}")
    scores = evaluate_walkforward(df_feat, models=models, k=args.folds)
    for m, sc in scores.items():
        print(f"  {m}: accuracy={sc:.3f}")
    if args.log_results:
        import csv, datetime
        Path(args.log_results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_results, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([
                datetime.datetime.utcnow().isoformat(),
                ';'.join(symbols),
                args.timeframe,
                args.horizon,
                'lr' in models and scores.get('lr', 0.0) or '',
                'xgb' in models and scores.get('xgb', 0.0) or '',
                'lgbm' in models and scores.get('lgbm', 0.0) or '',
            ])

    # Entrenar completo y guardar
    paths = train_full_and_save(df_feat, models=models)
    # Elegir mejor según scores
    if scores:
        best = max(scores.items(), key=lambda x: x[1])[0]
        best_path = paths.get(best)
        if best_path:
            out = Path('src/models/best_model.pkl')
            # Enlazamos copiando
            import shutil
            shutil.copyfile(best_path, out)
            print(f"[wf] best model: {best} -> {out}")


if __name__ == '__main__':
    main()
