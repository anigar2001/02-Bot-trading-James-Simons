"""Entrenamiento del 'Modelo Latino' sobre velas 1h y ventana diaria.

Detecta eventos de ruptura del rango de la mañana (00:00–13:00 UTC) y etiqueta éxito
si el objetivo (+k*ATR) se alcanza antes que el stop en un horizonte H horas.

Uso (desde contenedor fetcher):

  docker compose run --rm fetcher \
    python -m src.models.latin_training \
      --symbols BTCUSDC,ETHUSDC,LTCUSDC \
      --timeframe 1h \
      --format parquet \
      --horizon 4 \
      --atr_mult_target 1.2 \
      --atr_mult_stop 0.8

Salida: src/models/latin_model.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:
    LGBMClassifier = None  # type: ignore


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


def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        (df['high'] - df['low']).abs(),
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def filter_parts_by_date(parts: List[Path], start_date: str | None, end_date: str | None) -> List[Path]:
    """Filtra archivos mensuales (YYYY-MM.*) por rango [start_date, end_date] inclusive.
    Si el nombre no sigue el patrón, conserva el archivo.
    """
    if not parts or (not start_date and not end_date):
        return parts
    start_month = pd.to_datetime(start_date).to_period('M').to_timestamp() if start_date else None
    end_month = pd.to_datetime(end_date).to_period('M').to_timestamp() if end_date else None
    selected: List[Path] = []
    for p in parts:
        stem = p.stem  # p.ej., '2024-01'
        try:
            m = pd.to_datetime(stem, format='%Y-%m')
        except Exception:
            selected.append(p)
            continue
        ok = (start_month is None or m >= start_month) and (end_month is None or m <= end_month)
        if ok:
            selected.append(p)
    return selected


def build_events_local(
    df: pd.DataFrame,
    *,
    horizon: int = 4,
    atr_mult_target: float = 1.2,
    atr_mult_stop: float = 0.8,
    tz: str = 'UTC',
    morning_start: str = '00:00',
    morning_end: str = '13:00',
    session_start: str = '13:00',
    session_end: str | None = None,
):
    """
    Construye eventos usando ventanas horarias locales (maneja DST vía zoneinfo).
    - Rango mañana: [morning_start, morning_end) en tz local.
    - Operativa: [session_start, session_end] en tz local (si session_end se indica).
    - Éxito si take se alcanza antes que stop en `horizon` velas (acotado por fin de sesión cuando aplica).
    """
    # Asegurar índice UTC
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    loc_tz = ZoneInfo(tz)
    local_idx = df.index.tz_convert(loc_tz)
    atr = atr_series(df)
    rows = []

    # Agrupar por día en horario local
    local_dates = pd.Series(local_idx.date, index=df.index)
    for day, idx in local_dates.groupby(local_dates.values):
        df_day = df.loc[idx]
        lday = local_idx.loc[idx]
        if df_day.empty:
            continue

        ms = pd.to_datetime(morning_start).time()
        me = pd.to_datetime(morning_end).time()
        ss = pd.to_datetime(session_start).time()
        se = pd.to_datetime(session_end).time() if session_end else None

        # Rango de la mañana
        morning_mask = (lday.time >= ms) & (lday.time < me)
        morning = df_day.loc[morning_mask]
        if morning.empty:
            continue
        hi = float(morning['high'].max())
        lo = float(morning['low'].min())
        atr_d = float(atr.loc[df_day.index].iloc[-1]) if len(df_day) else np.nan
        if np.isnan(atr_d) or atr_d <= 0:
            continue
        buffer = 0.25 * atr_d

        # Tramo operativo local
        in_sess = lday.time >= ss
        if se is not None:
            in_sess &= lday.time <= se
        op = df_day.loc[in_sess]
        if op.empty:
            continue

        breakout_idx = None
        for ts, row in op.iterrows():
            if row['close'] > hi + buffer:
                breakout_idx = ts
                break
        if breakout_idx is None:
            continue

        entry = float(df.loc[breakout_idx, 'close'])
        stop = entry - atr_mult_stop * atr_d
        take = entry + atr_mult_target * atr_d

        # Horizonte efectivo (recortado por fin de sesión si aplica)
        h = max(1, int(horizon))
        if session_end:
            end_local_dt = pd.Timestamp.combine(pd.to_datetime(day).date(), se).replace(tzinfo=loc_tz)
            end_utc = end_local_dt.astimezone(ZoneInfo('UTC'))
            avail = df.loc[breakout_idx:end_utc]
            h = min(h, max(0, len(avail)))

        df_fut = df.loc[breakout_idx:]
        hit_take = (df_fut['high'].rolling(window=h, min_periods=1).max().iloc[h-1] >= take) if len(df_fut) >= h and h >= 1 else False
        hit_stop = (df_fut['low'].rolling(window=h, min_periods=1).min().iloc[h-1] <= stop) if len(df_fut) >= h and h >= 1 else False
        target = 1 if (hit_take and not hit_stop) else 0

        # Features al momento de ruptura
        ema50 = df['close'].ewm(span=50, adjust=False).mean().loc[breakout_idx]
        ema200 = df['close'].ewm(span=200, adjust=False).mean().loc[breakout_idx]
        ema_gap = float((ema50 - ema200) / (ema200 if ema200 != 0 else 1e-9))
        v = df['volume'].loc[:breakout_idx].iloc[-20:]
        vz = float((v.iloc[-1] - v.mean()) / (v.std() if v.std() != 0 else 1e-9))
        range_rel = (hi - lo) / (entry if entry != 0 else 1e-9)
        atrn = atr_d / entry

        rows.append({
            'ts': breakout_idx,
            'ema_gap': ema_gap,
            'v_z20': vz,
            'range_morning': range_rel,
            'atrn': atrn,
            'target': target,
        })

    return pd.DataFrame(rows)

def build_events(df: pd.DataFrame, horizon: int = 4, atr_mult_target: float = 1.2, atr_mult_stop: float = 0.8):
    # Asume df 1h UTC
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    atr = atr_series(df)
    rows = []
    # Agrupar por día UTC
    for day, df_day in df.groupby(df.index.date):
        # Rango de la mañana 00:00–13:00
        morning = df_day[df_day.index.time < pd.to_datetime('13:00').time()]
        if morning.empty:
            continue
        hi = float(morning['high'].max())
        lo = float(morning['low'].min())
        atr_d = float(atr.loc[df_day.index].iloc[-1]) if len(df_day) else np.nan
        if np.isnan(atr_d) or atr_d <= 0:
            continue
        buffer = 0.25 * atr_d
        # tramo operativo >= 13:00
        op = df_day[df_day.index.time >= pd.to_datetime('13:00').time()]
        if op.empty:
            continue
        # Detectar primera ruptura alcista válida (long-only por simplicidad)
        breakout_idx = None
        for ts, row in op.iterrows():
            if row['close'] > hi + buffer:
                breakout_idx = ts
                break
        if breakout_idx is None:
            continue
        entry = float(df.loc[breakout_idx, 'close'])
        stop = entry - atr_mult_stop * atr_d
        take = entry + atr_mult_target * atr_d
        # Horizonte de H horas
        df_fut = df.loc[breakout_idx:]
        # 1) ¿llega a take primero?
        hit_take = (df_fut['high'].rolling(window=horizon, min_periods=1).max().iloc[horizon-1] >= take) if len(df_fut) >= horizon else False
        # 2) ¿o toca stop primero?
        hit_stop = (df_fut['low'].rolling(window=horizon, min_periods=1).min().iloc[horizon-1] <= stop) if len(df_fut) >= horizon else False
        target = 1 if (hit_take and not hit_stop) else 0

        # Features al momento de ruptura
        # EMA gap simple (1h)
        ema50 = df['close'].ewm(span=50, adjust=False).mean().loc[breakout_idx]
        ema200 = df['close'].ewm(span=200, adjust=False).mean().loc[breakout_idx]
        ema_gap = float((ema50 - ema200) / (ema200 if ema200 != 0 else 1e-9))
        # Volumen relativo 20h
        v = df['volume'].loc[:breakout_idx].iloc[-20:]
        vz = float((v.iloc[-1] - v.mean()) / (v.std() if v.std() != 0 else 1e-9))
        # Rango mañana relativo
        range_rel = (hi - lo) / (entry if entry != 0 else 1e-9)
        # ATR normalizado
        atrn = atr_d / entry

        rows.append({
            'ts': breakout_idx,
            'ema_gap': ema_gap,
            'v_z20': vz,
            'range_morning': range_rel,
            'atrn': atrn,
            'target': target,
        })

    return pd.DataFrame(rows)


def best_threshold(y_true, y_proba, metric: str = 'accuracy'):
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.3, 0.7, 41):
        y_hat = (y_proba >= t).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_hat, zero_division=0)
        else:
            score = accuracy_score(y_true, y_hat)
        if score > best:
            best, best_t = score, t
    return best_t, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='BTCUSDC,ETHUSDC,LTCUSDC')
    ap.add_argument('--timeframe', default='1h')
    ap.add_argument('--format', choices=['csv', 'parquet'], default='parquet')
    ap.add_argument('--horizon', type=int, default=4)
    ap.add_argument('--atr_mult_target', type=float, default=1.2)
    ap.add_argument('--atr_mult_stop', type=float, default=0.8)
    ap.add_argument('--out', default='src/models/latin_model.pkl')
    ap.add_argument('--models', default='lr,xgb,lgbm', help='Modelos a probar: lr,xgb,lgbm')
    ap.add_argument('--metric', default='accuracy', choices=['accuracy','f1'])
    # Ventanas horarias locales y zona horaria
    ap.add_argument('--tz', default='UTC', help='Zona horaria, e.g., Europe/Madrid')
    ap.add_argument('--morning_start', default='00:00')
    ap.add_argument('--morning_end', default='13:00')
    ap.add_argument('--session_start', default='13:00')
    ap.add_argument('--session_end', default=None)
    # Rango de fechas (inclusive) para limitar el estudio
    ap.add_argument('--start_date', default=None, help='YYYY-MM-DD inclusive')
    ap.add_argument('--end_date', default=None, help='YYYY-MM-DD inclusive')
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    frames = []
    for sym in symbols:
        key = to_symbol_key(sym)
        parts = list_parts(key, args.timeframe, args.format)
        parts = filter_parts_by_date(parts, args.start_date, args.end_date)
        if not parts:
            print(f"[latin] no parts for {sym}")
            continue
        df = read_concat(parts, args.format)
        # Filtrar filas por rango exacto (UTC) si se indican
        if args.start_date:
            start_ts = pd.to_datetime(args.start_date, utc=True)
            df = df[df.index >= start_ts]
        if args.end_date:
            end_ts = pd.to_datetime(args.end_date, utc=True) + pd.Timedelta(days=1)
            df = df[df.index < end_ts]
        if df.empty:
            continue
        ev = build_events_local(
            df,
            horizon=args.horizon,
            atr_mult_target=args.atr_mult_target,
            atr_mult_stop=args.atr_mult_stop,
            tz=args.tz,
            morning_start=args.morning_start,
            morning_end=args.morning_end,
            session_start=args.session_start,
            session_end=args.session_end,
        )
        if not ev.empty:
            ev['symbol'] = sym
            frames.append(ev)

    if not frames:
        raise SystemExit("[latin] sin eventos para entrenar")
    data = pd.concat(frames, axis=0).sort_values('ts')
    feature_cols = ['ema_gap', 'v_z20', 'range_morning', 'atrn']
    X = data[feature_cols].values
    y = data['target'].values
    n = len(data)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    models_req = [m.strip() for m in args.models.split(',') if m.strip()]
    results = {}

    # Pesos por clase para desequilibrio
    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    def eval_model(probas):
        t, score = best_threshold(y_test, probas, metric=args.metric)
        y_hat = (probas >= t).astype(int)
        print(classification_report(y_test, y_hat, digits=3, zero_division=0))
        return t, score

    best_path = None
    best_score = -1.0
    best_name = None

    if 'lr' in models_req:
        base = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight='balanced')
        lr = CalibratedClassifierCV(base, method='sigmoid', cv=3)
        lr.fit(X_train, y_train)
        prob = lr.predict_proba(X_test)[:, 1]
        t, sc = eval_model(prob)
        path = Path(args.out).with_name('latin_trained_lr.pkl')
        dump({'model': lr, 'features': feature_cols, 'meta': {
            'type': 'latin_breakout', 'timeframe': args.timeframe, 'horizon': args.horizon,
            'atr_take': args.atr_mult_target, 'atr_stop': args.atr_mult_stop, 'threshold': t
        }}, path)
        results['lr'] = (str(path), sc)
        if sc > best_score:
            best_score, best_path, best_name = sc, path, 'lr'

    if 'xgb' in models_req and XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic', n_jobs=-1, reg_lambda=1.0, eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
        )
        xgb.fit(X_train, y_train)
        prob = xgb.predict_proba(X_test)[:, 1]
        t, sc = eval_model(prob)
        path = Path(args.out).with_name('latin_trained_xgb.pkl')
        dump({'model': xgb, 'features': feature_cols, 'meta': {
            'type': 'latin_breakout', 'timeframe': args.timeframe, 'horizon': args.horizon,
            'atr_take': args.atr_mult_target, 'atr_stop': args.atr_mult_stop, 'threshold': t
        }}, path)
        results['xgb'] = (str(path), sc)
        if sc > best_score:
            best_score, best_path, best_name = sc, path, 'xgb'

    if 'lgbm' in models_req and LGBMClassifier is not None:
        lgbm = LGBMClassifier(
            n_estimators=600, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            objective='binary', n_jobs=-1, class_weight='balanced'
        )
        lgbm.fit(X_train, y_train)
        prob = lgbm.predict_proba(X_test)[:, 1]
        t, sc = eval_model(prob)
        path = Path(args.out).with_name('latin_trained_lgbm.pkl')
        dump({'model': lgbm, 'features': feature_cols, 'meta': {
            'type': 'latin_breakout', 'timeframe': args.timeframe, 'horizon': args.horizon,
            'atr_take': args.atr_mult_target, 'atr_stop': args.atr_mult_stop, 'threshold': t
        }}, path)
        results['lgbm'] = (str(path), sc)
        if sc > best_score:
            best_score, best_path, best_name = sc, path, 'lgbm'

    # Guardar mejor
    if best_path is not None:
        out = Path(args.out)
        dump({'best': best_name, 'path': str(best_path)}, out.with_name('latin_best_info.pkl'))
        print(f"[latin] best {best_name} -> {best_path} (score={best_score:.3f})")
    else:
        print('[latin] no model trained')


if __name__ == '__main__':
    main()
