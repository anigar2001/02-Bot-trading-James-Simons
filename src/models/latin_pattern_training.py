"""Entrenamiento de un modelo basado en patrón intradía tipo 'Latin'.

Objetivo: Para los días en que:
- Entre 07:30 y 10:00 (hora local) el precio SUBE, y
- Entre 10:00 y 15:00 el precio BAJA,

estimar la probabilidad de que entre 15:00 y 17:30 el precio SUBA
(p. ej., cierre de 17:30 > cierre de 15:00).

Permite configurar la zona horaria (por defecto Europe/Madrid),
ajustar las ventanas y filtrar por rango de fechas.
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
from sklearn.metrics import classification_report, accuracy_score, f1_score, balanced_accuracy_score

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
        stem = p.stem  # 'YYYY-MM'
        try:
            m = pd.to_datetime(stem, format='%Y-%m')
        except Exception:
            selected.append(p)
            continue
        if (start_month is None or m >= start_month) and (end_month is None or m <= end_month):
            selected.append(p)
    return selected


def window_slice_day(df_utc: pd.DataFrame, tz: ZoneInfo, day_idx: pd.Index, start: str, end: str) -> pd.DataFrame:
    """Devuelve el sub-DataFrame dentro del día local indicado por day_idx
    y ventana [start, end) en hora local (incluye límites por candle)."""
    if df_utc.empty or len(day_idx) == 0:
        return df_utc.iloc[0:0]
    local_idx = df_utc.index.tz_convert(tz)
    # Mapear índices UTC a posiciones para extraer las horas locales correspondientes
    pos = df_utc.index.get_indexer(day_idx)
    lday = local_idx.take(pos)
    t_start = pd.to_datetime(start).time()
    t_end = pd.to_datetime(end).time()
    mask = (lday.time >= t_start) & (lday.time < t_end)
    return df_utc.loc[day_idx][mask]


def build_day_rows(df_utc: pd.DataFrame, *, tz_name: str,
                   morning_up: Tuple[str, str] = ("07:30", "10:00"),
                   midday_down: Tuple[str, str] = ("10:00", "15:00"),
                   afternoon_target: Tuple[str, str] = ("15:00", "17:30")) -> pd.DataFrame:
    """Construye filas por día cumpliendo el patrón y con etiqueta de tarde.

    Label: 1 si close(afternoon_end) > close(afternoon_start), 0 en caso contrario.
    Filtro: morning_return > 0 y midday_return < 0.
    Features: morning_return, midday_return, atrn (día), range_morning, vol20.
    """
    if df_utc.index.tzinfo is None:
        df_utc.index = df_utc.index.tz_localize('UTC')
    else:
        df_utc.index = df_utc.index.tz_convert('UTC')
    tz = ZoneInfo(tz_name)
    local_idx = df_utc.index.tz_convert(tz)

    # Agrupar por día local
    local_dates = pd.Series(local_idx.date, index=df_utc.index)
    rows = []
    groups = local_dates.groupby(local_dates.values).groups
    for day, idx_vals in groups.items():
        idx = pd.Index(idx_vals)
        df_day = df_utc.loc[idx]
        if df_day.empty:
            continue

        w_m = window_slice_day(df_utc, tz, idx, *morning_up)
        w_md = window_slice_day(df_utc, tz, idx, *midday_down)
        w_a = window_slice_day(df_utc, tz, idx, *afternoon_target)

        if w_m.empty or w_md.empty or w_a.empty:
            continue

        # Returns por ventana (usando primeros/últimos cierres dentro de cada tramo)
        m_ret = float(w_m['close'].iloc[-1] / w_m['close'].iloc[0] - 1.0)
        md_ret = float(w_md['close'].iloc[-1] / w_md['close'].iloc[0] - 1.0)
        a_ret = float(w_a['close'].iloc[-1] / w_a['close'].iloc[0] - 1.0)

        # Filtro patrón
        if not (m_ret > 0 and md_ret < 0):
            continue

        # ATR normalizado aprox (TR media 14 / close actual)
        tr = pd.concat([
            (df_day['high'] - df_day['low']).abs(),
            (df_day['high'] - df_day['close'].shift()).abs(),
            (df_day['low'] - df_day['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        atrn = float((atr14.iloc[-1] / df_day['close'].iloc[-1]) if pd.notna(atr14.iloc[-1]) else np.nan)
        if not np.isfinite(atrn):
            continue

        # Rango de la mañana relativo (usando 07:30-10:00)
        rng_hi = float(w_m['high'].max())
        rng_lo = float(w_m['low'].min())
        range_morning = (rng_hi - rng_lo) / float(df_day['close'].iloc[-1])

        # Volatilidad simple rolling 20h del día hasta ese momento
        vol20 = df_day['close'].pct_change().rolling(20).std().iloc[-1]
        vol20 = float(vol20) if pd.notna(vol20) else 0.0

        rows.append({
            'date': pd.Timestamp(day),
            'morning_ret': m_ret,
            'midday_ret': md_ret,
            'atrn': atrn,
            'range_morning': range_morning,
            'vol20': vol20,
            'target': 1 if a_ret > 0 else 0,
        })

    return pd.DataFrame(rows)


def build_dataset(symbols: List[str], timeframe: str, fmt: str,
                  tz: str,
                  morning_up: Tuple[str, str],
                  midday_down: Tuple[str, str],
                  afternoon_target: Tuple[str, str],
                  start_date: str | None, end_date: str | None) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        key = to_symbol_key(sym)
        parts = list_parts(key, timeframe, fmt)
        parts = filter_parts_by_date(parts, start_date, end_date)
        if not parts:
            print(f"[latin-pattern] no parts for {sym}")
            continue
        df = read_concat(parts, fmt)
        if df.empty:
            continue
        # recorte exacto por fecha si se especifica
        if start_date:
            start_ts = pd.to_datetime(start_date, utc=True)
            df = df[df.index >= start_ts]
        if end_date:
            end_ts = pd.to_datetime(end_date, utc=True) + pd.Timedelta(days=1)
            df = df[df.index < end_ts]
        if df.empty:
            continue
        rows = build_day_rows(df, tz_name=tz,
                              morning_up=morning_up,
                              midday_down=midday_down,
                              afternoon_target=afternoon_target)
        if not rows.empty:
            rows['symbol'] = sym
            frames.append(rows)
    if not frames:
        raise SystemExit("[latin-pattern] sin filas tras construir dataset")
    return pd.concat(frames, axis=0).sort_values('date')


def best_threshold(y_true, y_proba, mode: str) -> tuple[float, float]:
    """Selecciona umbral según `mode`.
    mode ∈ {'f1_pos','f1_macro','accuracy','balanced_accuracy','youden'}
    Devuelve (threshold, score).
    """
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
            # Youden J = TPR - FPR
            tp = ((y_true == 1) & (y_hat == 1)).sum()
            fn = ((y_true == 1) & (y_hat == 0)).sum()
            tn = ((y_true == 0) & (y_hat == 0)).sum()
            fp = ((y_true == 0) & (y_hat == 1)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            score = tpr - fpr
        else:  # accuracy
            score = accuracy_score(y_true, y_hat)
        if score > best:
            best, best_t = score, t
    return best_t, best


def print_summary(ds: pd.DataFrame, tz: str, windows: Dict[str, Tuple[str, str]], start_date: str | None, end_date: str | None, timeframe: str):
    total = int(len(ds))
    pos = int((ds['target'] == 1).sum())
    neg = total - pos
    pct = (pos / total * 100.0) if total else 0.0
    days = int(ds['date'].nunique()) if 'date' in ds.columns else None
    symbols = sorted(ds['symbol'].unique()) if 'symbol' in ds.columns else []
    per_sym = ds.groupby('symbol')['target'].agg(['count','sum']).reset_index() if 'symbol' in ds.columns else pd.DataFrame()
    rng = f"{start_date or 'min'} → {end_date or 'max'}"
    # Párrafo resumen
    print("[latin-pattern] Resumen de ocurrencias:")
    print(f"- Ventanas (local {tz}): mañana {windows['morning'][0]}–{windows['morning'][1]}, media {windows['midday'][0]}–{windows['midday'][1]}, tarde {windows['afternoon'][0]}–{windows['afternoon'][1]}")
    print(f"- Rango de estudio: {rng} | Timeframe: {timeframe}")
    if days is not None:
        print(f"- Días locales con patrón (entrada cumplida): {days}")
    print(f"- Ocurrencias (entrada = mañana↑ y media↓): {total}")
    print(f"- Salidas positivas (tarde↑): {pos} de {total} ({pct:.1f}%)")
    if not per_sym.empty and len(symbols) > 1:
        # Listado conciso por símbolo
        for _, row in per_sym.iterrows():
            c = int(row['count']); s = int(row['sum']); pp = (s/c*100.0) if c else 0.0
            print(f"  · {row['symbol']}: entradas={c}, salidas+={s} ({pp:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', default='ETHUSDC', help='Uno o varios, separados por coma')
    ap.add_argument('--timeframe', default='1h')
    ap.add_argument('--format', choices=['csv', 'parquet'], default='parquet')
    ap.add_argument('--tz', default='Europe/Madrid')
    ap.add_argument('--morning_start', default='07:30')
    ap.add_argument('--morning_end', default='10:00')
    ap.add_argument('--midday_start', default='10:00')
    ap.add_argument('--midday_end', default='15:00')
    ap.add_argument('--afternoon_start', default='15:00')
    ap.add_argument('--afternoon_end', default='17:30')
    ap.add_argument('--start_date', default=None, help='YYYY-MM-DD inclusive')
    ap.add_argument('--end_date', default=None, help='YYYY-MM-DD inclusive')
    ap.add_argument('--models', default='lr,xgb,lgbm')
    ap.add_argument('--metric', default='accuracy', choices=['accuracy','f1'], help='Métrica de reporte del informe')
    ap.add_argument('--threshold_mode', default='f1_pos', choices=['f1_pos','f1_macro','accuracy','balanced_accuracy','youden'], help='Criterio para elegir el umbral')
    ap.add_argument('--split_by_day', action='store_true', help='Divide train/test por días locales (evita fuga intra-día)')
    ap.add_argument('--out_prefix', default='src/models/latin_pattern')
    ap.add_argument('--log_results', default=None, help='Ruta CSV donde registrar resumen por ejecución')
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    models = [m.strip() for m in args.models.split(',') if m.strip()]

    ds = build_dataset(
        symbols,
        timeframe=args.timeframe,
        fmt=args.format,
        tz=args.tz,
        morning_up=(args.morning_start, args.morning_end),
        midday_down=(args.midday_start, args.midday_end),
        afternoon_target=(args.afternoon_start, args.afternoon_end),
        start_date=args.start_date,
        end_date=args.end_date,
    )
    feature_cols = ['morning_ret', 'midday_ret', 'atrn', 'range_morning', 'vol20']
    X_all = ds[feature_cols].copy()
    y_all = ds['target'].values

    # Resumen antes de entrenar
    print_summary(
        ds,
        tz=args.tz,
        windows={'morning': (args.morning_start, args.morning_end), 'midday': (args.midday_start, args.midday_end), 'afternoon': (args.afternoon_start, args.afternoon_end)},
        start_date=args.start_date, end_date=args.end_date, timeframe=args.timeframe,
    )

    if args.split_by_day:
        # Split por días locales
        tz = ZoneInfo(args.tz)
        ds_local_dates = ds['date'].dt.tz_localize('UTC').dt.tz_convert(tz) if ds['date'].dt.tz is None else ds['date'].dt.tz_convert(tz)
        unique_days = np.array(sorted(set(ds_local_dates.dt.date)))
        dsplit = int(len(unique_days) * 0.8)
        train_days = set(unique_days[:dsplit])
        train_mask = ds_local_dates.dt.date.apply(lambda d: d in train_days)
        X_train_df, X_test_df = X_all[train_mask], X_all[~train_mask]
        y_train, y_test = y_all[train_mask.values], y_all[~train_mask.values]
    else:
        n = len(ds)
        split = int(n * 0.8)
        X_train_df, X_test_df = X_all.iloc[:split], X_all.iloc[split:]
        y_train, y_test = y_all[:split], y_all[split:]

    results: Dict[str, Dict[str, object]] = {}

    if 'lr' in models:
        base = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight='balanced')
        lr = CalibratedClassifierCV(base, method='sigmoid', cv=3)
        lr.fit(X_train_df, y_train)
        prob = lr.predict_proba(X_test_df)[:, 1]
        t, sc = best_threshold(y_test, prob, args.threshold_mode)
        y_hat = (prob >= t).astype(int)
        print('[latin-pattern] LR report:')
        print(classification_report(y_test, y_hat, digits=3, zero_division=0))
        out = Path(args.out_prefix + '_lr.pkl')
        dump({'model': lr, 'features': feature_cols, 'meta': {
            'type': 'latin_pattern', 'tz': args.tz,
            'windows': {
                'morning': [args.morning_start, args.morning_end],
                'midday': [args.midday_start, args.midday_end],
                'afternoon': [args.afternoon_start, args.afternoon_end],
            },
            'threshold': t,
            'date_range': [args.start_date, args.end_date],
            'timeframe': args.timeframe,
        }}, out)
        results['lr'] = {'path': str(out), 'score': float(sc), 'threshold': float(t)}

    if 'xgb' in models and XGBClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic', n_jobs=-1, reg_lambda=1.0, eval_metric='logloss',
        )
        xgb.fit(X_train_df, y_train)
        prob = xgb.predict_proba(X_test_df)[:, 1]
        t, sc = best_threshold(y_test, prob, args.threshold_mode)
        y_hat = (prob >= t).astype(int)
        print('[latin-pattern] XGB report:')
        print(classification_report(y_test, y_hat, digits=3, zero_division=0))
        out = Path(args.out_prefix + '_xgb.pkl')
        dump({'model': xgb, 'features': feature_cols, 'meta': {
            'type': 'latin_pattern', 'tz': args.tz,
            'windows': {
                'morning': [args.morning_start, args.morning_end],
                'midday': [args.midday_start, args.midday_end],
                'afternoon': [args.afternoon_start, args.afternoon_end],
            },
            'threshold': t,
            'date_range': [args.start_date, args.end_date],
            'timeframe': args.timeframe,
        }}, out)
        results['xgb'] = {'path': str(out), 'score': float(sc), 'threshold': float(t)}

    if 'lgbm' in models and LGBMClassifier is not None:
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
        lgbm.fit(X_train_df, y_train)
        prob = lgbm.predict_proba(X_test_df)[:, 1]
        t, sc = best_threshold(y_test, prob, args.threshold_mode)
        y_hat = (prob >= t).astype(int)
        print('[latin-pattern] LGBM report:')
        print(classification_report(y_test, y_hat, digits=3, zero_division=0))
        out = Path(args.out_prefix + '_lgbm.pkl')
        dump({'model': lgbm, 'features': feature_cols, 'meta': {
            'type': 'latin_pattern', 'tz': args.tz,
            'windows': {
                'morning': [args.morning_start, args.morning_end],
                'midday': [args.midday_start, args.midday_end],
                'afternoon': [args.afternoon_start, args.afternoon_end],
            },
            'threshold': t,
            'date_range': [args.start_date, args.end_date],
            'timeframe': args.timeframe,
        }}, out)
        results['lgbm'] = {'path': str(out), 'score': float(sc), 'threshold': float(t)}

    if results:
        best_name, best_info = max(results.items(), key=lambda kv: kv[1]['score'])
        print(f"[latin-pattern] best={best_name} score={best_info['score']:.3f} -> {best_info['path']}")
        # Log CSV
        if args.log_results:
            from datetime import datetime
            Path(args.log_results).parent.mkdir(parents=True, exist_ok=True)
            header = [
                'timestamp','symbols','timeframe','tz','start_date','end_date',
                'total','positives','pos_pct','best_model','best_score',
                'lr_score','xgb_score','lgbm_score',
                'lr_thr','xgb_thr','lgbm_thr'
            ]
            total = int(len(ds)); positives = int((ds['target']==1).sum()); pos_pct = (positives/total*100.0) if total else 0.0
            row = {
                'timestamp': datetime.utcnow().isoformat(),
                'symbols': ','.join(symbols),
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
        print('[latin-pattern] no model trained')


if __name__ == '__main__':
    main()
