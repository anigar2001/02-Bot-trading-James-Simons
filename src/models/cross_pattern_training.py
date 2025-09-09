"""Entrenamiento de patrón cruzado BTC→LTC (base→target).

Idea: si por la mañana (ventana local) BTC sube y LTC no sube,
estimar la probabilidad de que por la tarde (ventana local) LTC suba.

Soporta:
- Zona horaria local con DST (zoneinfo)
- Rango de fechas (start/end)
- Particiones parquet/csv en src/data/historical/partitioned/{SYM}/{TF}
- Modelos: LR (calibrada), XGB, LGBM
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from joblib import dump
from zoneinfo import ZoneInfo

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)

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


def filter_parts_by_date(parts: List[Path], start_date: str | None, end_date: str | None) -> List[Path]:
    if not parts or (not start_date and not end_date):
        return parts
    start_month = pd.to_datetime(start_date).to_period('M').to_timestamp() if start_date else None
    end_month = pd.to_datetime(end_date).to_period('M').to_timestamp() if end_date else None
    selected: List[Path] = []
    for p in parts:
        stem = p.stem
        try:
            m = pd.to_datetime(stem, format='%Y-%m')
        except Exception:
            selected.append(p)
            continue
        if (start_month is None or m >= start_month) and (end_month is None or m <= end_month):
            selected.append(p)
    return selected


def window_slice_day(df_utc: pd.DataFrame, tz: ZoneInfo, day_idx: pd.Index, start: str, end: str) -> pd.DataFrame:
    if df_utc.empty or len(day_idx) == 0:
        return df_utc.iloc[0:0]
    local_idx = df_utc.index.tz_convert(tz)
    pos = df_utc.index.get_indexer(day_idx)
    lday = local_idx.take(pos)
    t_start = pd.to_datetime(start).time()
    t_end = pd.to_datetime(end).time()
    mask = (lday.time >= t_start) & (lday.time < t_end)
    return df_utc.loc[day_idx][mask]


def build_groups_by_local_day(df_utc: pd.DataFrame, tz: ZoneInfo) -> Dict[object, pd.Index]:
    local = df_utc.index.tz_convert(tz)
    s = pd.Series(local.date, index=df_utc.index)
    return {day: pd.Index(idx) for day, idx in s.groupby(s.values).groups.items()}


def build_day_rows_cross(
    df_base: pd.DataFrame,
    df_target: pd.DataFrame,
    *,
    tz_name: str,
    morning: Tuple[str, str],
    afternoon: Tuple[str, str],
) -> pd.DataFrame:
    # Asegurar UTC
    if df_base.index.tzinfo is None:
        df_base.index = df_base.index.tz_localize('UTC')
    else:
        df_base.index = df_base.index.tz_convert('UTC')
    if df_target.index.tzinfo is None:
        df_target.index = df_target.index.tz_localize('UTC')
    else:
        df_target.index = df_target.index.tz_convert('UTC')

    tz = ZoneInfo(tz_name)
    g_base = build_groups_by_local_day(df_base, tz)
    g_tgt = build_groups_by_local_day(df_target, tz)
    days = sorted(set(g_base.keys()) & set(g_tgt.keys()))

    rows = []
    for day in days:
        idx_b = g_base.get(day)
        idx_t = g_tgt.get(day)
        if idx_b is None or idx_t is None:
            continue
        # Ventanas
        w_b_m = window_slice_day(df_base, tz, idx_b, morning[0], morning[1])
        w_t_m = window_slice_day(df_target, tz, idx_t, morning[0], morning[1])
        w_t_a = window_slice_day(df_target, tz, idx_t, afternoon[0], afternoon[1])
        if w_b_m.empty or w_t_m.empty or w_t_a.empty:
            continue

        b_m_ret = float(w_b_m['close'].iloc[-1] / w_b_m['close'].iloc[0] - 1.0)
        t_m_ret = float(w_t_m['close'].iloc[-1] / w_t_m['close'].iloc[0] - 1.0)
        t_a_ret = float(w_t_a['close'].iloc[-1] / w_t_a['close'].iloc[0] - 1.0)

        # Condición de entrada: BTC sube, LTC no sube
        if not (b_m_ret > 0 and t_m_ret <= 0):
            continue

        # Features de target
        # ATRn (día) target
        tr = pd.concat([
            (df_target.loc[idx_t]['high'] - df_target.loc[idx_t]['low']).abs(),
            (df_target.loc[idx_t]['high'] - df_target.loc[idx_t]['close'].shift()).abs(),
            (df_target.loc[idx_t]['low'] - df_target.loc[idx_t]['close'].shift()).abs(),
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        atrn = float((atr14.iloc[-1] / df_target.loc[idx_t]['close'].iloc[-1]) if pd.notna(atr14.iloc[-1]) else np.nan)
        if not np.isfinite(atrn):
            continue
        rng_hi = float(w_t_m['high'].max())
        rng_lo = float(w_t_m['low'].min())
        range_morning = (rng_hi - rng_lo) / float(df_target.loc[idx_t]['close'].iloc[-1])
        vol20 = df_target.loc[idx_t]['close'].pct_change().rolling(20).std().iloc[-1]
        vol20 = float(vol20) if pd.notna(vol20) else 0.0

        rows.append({
            'date': pd.Timestamp(day),
            'base_morning_ret': b_m_ret,
            'target_morning_ret': t_m_ret,
            'ret_diff': b_m_ret - t_m_ret,
            'atrn': atrn,
            'range_morning': range_morning,
            'vol20': vol20,
            'target': 1 if t_a_ret > 0 else 0,
        })

    return pd.DataFrame(rows)


def best_threshold(y_true, y_proba, mode: str) -> tuple[float, float]:
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.2, 0.8, 61):
        y_hat = (y_proba >= t).astype(int)
        if mode == 'f1_pos':
            score = f1_score(y_true, y_hat, zero_division=0)
        elif mode == 'f1_macro':
            score = f1_score(y_true, y_hat, average='macro', zero_division=0)
        elif mode == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_hat)
        elif mode == 'youden':
            tp = ((y_true == 1) & (y_hat == 1)).sum()
            fn = ((y_true == 1) & (y_hat == 0)).sum()
            tn = ((y_true == 0) & (y_hat == 0)).sum()
            fp = ((y_true == 0) & (y_hat == 1)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            score = tpr - fpr
        else:
            score = accuracy_score(y_true, y_hat)
        if score > best:
            best, best_t = score, t
    return best_t, best


def print_summary(ds: pd.DataFrame, *, tz: str, morning: Tuple[str, str], afternoon: Tuple[str, str], start_date: str | None, end_date: str | None, timeframe: str, base: str, target: str):
    total = int(len(ds))
    pos = int((ds['target'] == 1).sum())
    pct = (pos / total * 100.0) if total else 0.0
    days = int(ds['date'].nunique()) if 'date' in ds.columns else None
    rng = f"{start_date or 'min'} → {end_date or 'max'}"
    print("[cross-pattern] Resumen de ocurrencias:")
    print(f"- Base={base} Target={target} | tz={tz}")
    print(f"- Ventanas: mañana {morning[0]}–{morning[1]}, tarde {afternoon[0]}–{afternoon[1]}")
    print(f"- Rango de estudio: {rng} | Timeframe: {timeframe}")
    if days is not None:
        print(f"- Días locales con patrón (BTC mañana↑ & LTC no↑): {days}")
    print(f"- Ocurrencias: {total}")
    print(f"- Salidas positivas (tarde LTC↑): {pos} de {total} ({pct:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='BTCUSDC', help='Símbolo base (ej. BTCUSDC)')
    ap.add_argument('--target', default='LTCUSDC', help='Símbolo objetivo (ej. LTCUSDC)')
    ap.add_argument('--timeframe', default='1h')
    ap.add_argument('--format', choices=['csv','parquet'], default='parquet')
    ap.add_argument('--tz', default='Europe/Madrid')
    ap.add_argument('--morning_start', default='07:30')
    ap.add_argument('--morning_end', default='10:00')
    ap.add_argument('--afternoon_start', default='15:00')
    ap.add_argument('--afternoon_end', default='17:30')
    ap.add_argument('--start_date', default=None)
    ap.add_argument('--end_date', default=None)
    ap.add_argument('--models', default='lr,xgb,lgbm')
    ap.add_argument('--metric', default='f1', choices=['accuracy','f1'])
    ap.add_argument('--threshold_mode', default='balanced_accuracy', choices=['f1_pos','f1_macro','accuracy','balanced_accuracy','youden'])
    ap.add_argument('--split_by_day', action='store_true')
    ap.add_argument('--out_prefix', default='src/models/cross_pattern')
    ap.add_argument('--log_results', default=None, help='Ruta CSV para registrar resumen por ejecución')
    args = ap.parse_args()

    base_key, tgt_key = to_symbol_key(args.base), to_symbol_key(args.target)
    parts_b = filter_parts_by_date(list_parts(base_key, args.timeframe, args.format), args.start_date, args.end_date)
    parts_t = filter_parts_by_date(list_parts(tgt_key, args.timeframe, args.format), args.start_date, args.end_date)
    if not parts_b or not parts_t:
        raise SystemExit('[cross-pattern] faltan particiones para base o target')
    df_b = read_concat(parts_b, args.format)
    df_t = read_concat(parts_t, args.format)
    if args.start_date:
        start_ts = pd.to_datetime(args.start_date, utc=True)
        df_b = df_b[df_b.index >= start_ts]
        df_t = df_t[df_t.index >= start_ts]
    if args.end_date:
        end_ts = pd.to_datetime(args.end_date, utc=True) + pd.Timedelta(days=1)
        df_b = df_b[df_b.index < end_ts]
        df_t = df_t[df_t.index < end_ts]
    if df_b.empty or df_t.empty:
        raise SystemExit('[cross-pattern] datos vacíos tras filtrar')

    ds = build_day_rows_cross(
        df_b,
        df_t,
        tz_name=args.tz,
        morning=(args.morning_start, args.morning_end),
        afternoon=(args.afternoon_start, args.afternoon_end),
    )
    if ds.empty:
        raise SystemExit('[cross-pattern] sin filas (ningún día cumple el patrón)')

    feature_cols = ['base_morning_ret', 'target_morning_ret', 'ret_diff', 'atrn', 'range_morning', 'vol20']
    X_all = ds[feature_cols].copy()
    y_all = ds['target'].values

    print_summary(
        ds,
        tz=args.tz,
        morning=(args.morning_start, args.morning_end),
        afternoon=(args.afternoon_start, args.afternoon_end),
        start_date=args.start_date, end_date=args.end_date,
        timeframe=args.timeframe, base=args.base, target=args.target,
    )

    if args.split_by_day and 'date' in ds.columns:
        tz = ZoneInfo(args.tz)
        ds_local = ds['date'].dt.tz_localize('UTC').dt.tz_convert(tz) if ds['date'].dt.tz is None else ds['date'].dt.tz_convert(tz)
        unique_days = np.array(sorted(set(ds_local.dt.date)))
        dsplit = int(len(unique_days) * 0.8)
        train_days = set(unique_days[:dsplit])
        train_mask = ds_local.dt.date.apply(lambda d: d in train_days)
        X_train, X_test = X_all[train_mask], X_all[~train_mask]
        y_train, y_test = y_all[train_mask.values], y_all[~train_mask.values]
    else:
        n = len(ds)
        split = int(n * 0.8)
        X_train, X_test = X_all.iloc[:split], X_all.iloc[split:]
        y_train, y_test = y_all[:split], y_all[split:]

    results: Dict[str, Dict[str, object]] = {}

    if 'lr' in args.models.split(','):
        base = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight='balanced')
        lr = CalibratedClassifierCV(base, method='sigmoid', cv=3)
        lr.fit(X_train, y_train)
        prob = lr.predict_proba(X_test)[:, 1]
        t, sc = best_threshold(y_test, prob, args.threshold_mode)
        y_hat = (prob >= t).astype(int)
        print('[cross-pattern] LR report:')
        print(classification_report(y_test, y_hat, digits=3, zero_division=0))
        out = Path(args.out_prefix + '_lr.pkl')
        dump({'model': lr, 'features': feature_cols, 'meta': {
            'type': 'cross_pattern', 'base': args.base, 'target': args.target,
            'tz': args.tz,
            'windows': {'morning': [args.morning_start, args.morning_end], 'afternoon': [args.afternoon_start, args.afternoon_end]},
            'threshold': t,
            'date_range': [args.start_date, args.end_date],
            'timeframe': args.timeframe,
        }}, out)
        results['lr'] = {'path': str(out), 'score': float(sc), 'threshold': float(t)}

    if 'xgb' in args.models.split(',') and XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic', n_jobs=-1, reg_lambda=1.0, eval_metric='logloss',
        )
        xgb.fit(X_train, y_train)
        prob = xgb.predict_proba(X_test)[:, 1]
        t, sc = best_threshold(y_test, prob, args.threshold_mode)
        y_hat = (prob >= t).astype(int)
        print('[cross-pattern] XGB report:')
        print(classification_report(y_test, y_hat, digits=3, zero_division=0))
        out = Path(args.out_prefix + '_xgb.pkl')
        dump({'model': xgb, 'features': feature_cols, 'meta': {
            'type': 'cross_pattern', 'base': args.base, 'target': args.target,
            'tz': args.tz,
            'windows': {'morning': [args.morning_start, args.morning_end], 'afternoon': [args.afternoon_start, args.afternoon_end]},
            'threshold': t,
            'date_range': [args.start_date, args.end_date],
            'timeframe': args.timeframe,
        }}, out)
        results['xgb'] = {'path': str(out), 'score': float(sc), 'threshold': float(t)}

    if 'lgbm' in args.models.split(',') and LGBMClassifier is not None:
        lgbm = LGBMClassifier(
            n_estimators=800,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary',
            n_jobs=-1,
            class_weight='balanced',
            num_leaves=15,
            max_depth=3,
            min_data_in_leaf=5,
            force_col_wise=True,
            verbosity=-1,
        )
        lgbm.fit(X_train, y_train)
        prob = lgbm.predict_proba(X_test)[:, 1]
        t, sc = best_threshold(y_test, prob, args.threshold_mode)
        y_hat = (prob >= t).astype(int)
        print('[cross-pattern] LGBM report:')
        print(classification_report(y_test, y_hat, digits=3, zero_division=0))
        out = Path(args.out_prefix + '_lgbm.pkl')
        dump({'model': lgbm, 'features': feature_cols, 'meta': {
            'type': 'cross_pattern', 'base': args.base, 'target': args.target,
            'tz': args.tz,
            'windows': {'morning': [args.morning_start, args.morning_end], 'afternoon': [args.afternoon_start, args.afternoon_end]},
            'threshold': t,
            'date_range': [args.start_date, args.end_date],
            'timeframe': args.timeframe,
        }}, out)
        results['lgbm'] = {'path': str(out), 'score': float(sc), 'threshold': float(t)}

    if results:
        best_name, best_info = max(results.items(), key=lambda kv: kv[1]['score'])
        print(f"[cross-pattern] best={best_name} score={best_info['score']:.3f} -> {best_info['path']}")
        # Log CSV
        if args.log_results:
            from datetime import datetime
            Path(args.log_results).parent.mkdir(parents=True, exist_ok=True)
            header = [
                'timestamp','base','target','timeframe','tz','start_date','end_date',
                'total','positives','pos_pct','best_model','best_score',
                'lr_score','xgb_score','lgbm_score','lr_thr','xgb_thr','lgbm_thr'
            ]
            total = int(len(ds)); positives = int((ds['target']==1).sum()); pos_pct = (positives/total*100.0) if total else 0.0
            row = {
                'timestamp': datetime.utcnow().isoformat(),
                'base': args.base,
                'target': args.target,
                'timeframe': args.timeframe,
                'tz': args.tz,
                'start_date': args.start_date or '',
                'end_date': args.end_date or '',
                'total': total,
                'positives': positives,
                'pos_pct': f"{pos_pct:.2f}",
                'best_model': best_name,
                'best_score': f"{best_info['score']:.4f}",
                'lr_score': results.get('lr',{}).get('score',''),
                'xgb_score': results.get('xgb',{}).get('score',''),
                'lgbm_score': results.get('lgbm',{}).get('score',''),
                'lr_thr': results.get('lr',{}).get('threshold',''),
                'xgb_thr': results.get('xgb',{}).get('threshold',''),
                'lgbm_thr': results.get('lgbm',{}).get('threshold',''),
            }
            import csv
            write_header = not Path(args.log_results).exists()
            with open(args.log_results, 'a', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=header)
                if write_header:
                    w.writeheader()
                w.writerow(row)
    else:
        print('[cross-pattern] no model trained')


if __name__ == '__main__':
    main()
