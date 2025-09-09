import os
from pathlib import Path
from typing import List

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import subprocess
import datetime
import csv
import json

from src.utils.api import BinanceClient


def read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding='utf-8')
    except Exception:
        return ""


def load_trades(path: str, limit: int = 100) -> List[dict]:
    p = Path(path)
    if not p.exists():
        return []
    lines = p.read_text(encoding='utf-8').strip().splitlines()[-limit:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return list(reversed(out))


def create_app():
    app = Flask(__name__, template_folder='templates', static_folder='static')
    # Configurar secret key para sesiones/flash
    app.secret_key = os.getenv('SECRET_KEY') or os.getenv('FLASK_SECRET_KEY') or 'dev-secret-change-me'
    
    def normalize_trades(trades):
        rows = []
        for t in trades:
            ts = t.get('ts', '')
            strat = t.get('strategy', '')
            action = t.get('action', t.get('signal', {}).get('action'))
            reason = (t.get('signal') or {}).get('reason', '')
            pnl = t.get('pnl')
            # single order
            if isinstance(t.get('order'), dict):
                od = t['order']
                rows.append({
                    'ts': ts,
                    'strategy': strat,
                    'symbols': od.get('symbol') or (t.get('signal') or {}).get('symbol'),
                    'sides': od.get('side') or action,
                    'amounts': od.get('amount') or od.get('origQty') or od.get('executedQty'),
                    'action': action,
                    'reason': reason,
                    'pnl': pnl,
                })
                continue
            # multi-leg
            if isinstance(t.get('orders'), list):
                syms = []
                sides = []
                amts = []
                for od in t['orders']:
                    syms.append(od.get('symbol'))
                    sides.append(od.get('side'))
                    amts.append(od.get('amount') or od.get('origQty') or od.get('executedQty'))
                rows.append({
                    'ts': ts,
                    'strategy': strat,
                    'symbols': ", ".join([str(x) for x in syms if x]),
                    'sides': ", ".join([str(x) for x in sides if x]),
                    'amounts': ", ".join([str(x) for x in amts if x is not None]),
                    'action': action,
                    'reason': reason,
                    'pnl': pnl,
                })
                continue
            rows.append({'ts': ts, 'strategy': strat, 'symbols': '', 'sides': '', 'amounts': '', 'action': action, 'reason': reason, 'pnl': pnl})
        return rows

    @app.route('/')
    def index():
        api_key = os.getenv('BINANCE_API_KEY', '')
        api_secret = os.getenv('BINANCE_API_SECRET', '')
        api_base = os.getenv('API_BASE', 'https://testnet.binance.vision')
        client = BinanceClient(api_key=api_key, api_secret=api_secret, api_base=api_base, enable_rate_limit=True, testnet=True, dry_run=True)
        try:
            client.load_markets()
        except Exception:
            pass
        try:
            balance = client.get_balance()
        except Exception:
            balance = {'free': {}, 'used': {}, 'total': {}}

        # Filtrar balance a sólo BTC, USDT y LTC
        assets_of_interest = ['BTC', 'USDC', 'LTC']
        totals = balance.get('total') or {}
        frees = balance.get('free') or {}
        balance_filtered = {}
        for a in assets_of_interest:
            val_total = totals.get(a)
            val_free = frees.get(a)
            if val_total or val_free:
                try:
                    balance_filtered[a] = {
                        'free': float(val_free) if val_free is not None else 0.0,
                        'total': float(val_total) if val_total is not None else 0.0,
                    }
                except Exception:
                    balance_filtered[a] = {
                        'free': val_free,
                        'total': val_total,
                    }
        trades = load_trades('src/data/logs/trades.jsonl', limit=50)
        trade_rows = normalize_trades(trades)
        pos_json = read_text('src/data/logs/positions.json')
        alloc_json = read_text('src/data/logs/allocator.json')
        # Valorar en USDT (USDT=1, BTC/LTC vía últimos precios)
        total_usdc = 0.0
        try:
            prices = {
                'USDC': 1.0,
                'BTC': float(client.fetch_last_price('BTC/USDC')),
                'LTC': float(client.fetch_last_price('LTC/USDC')),
            }
        except Exception:
            prices = {'USDC': 1.0, 'BTC': 0.0, 'LTC': 0.0}
        for a, vals in balance_filtered.items():
            try:
                px = prices.get(a, 0.0)
                val_usdc = float(vals.get('total', 0.0)) * float(px)
                balance_filtered[a]['value_usdc'] = val_usdc
                total_usdc += val_usdc
            except Exception:
                pass

        # Parse positions
        positions = []
        try:
            pos_dict = json.loads(pos_json) if pos_json else {}
            for sym, p in (pos_dict or {}).items():
                positions.append({
                    'symbol': sym,
                    'strategy': p.get('strategy', ''),
                    'side': p.get('side', ''),
                    'qty': p.get('qty', 0),
                    'entry_price': p.get('entry_price', 0),
                    'stop_loss': p.get('stop_loss'),
                    'take_profit': p.get('take_profit'),
                })
        except Exception:
            positions = []

        # Parse allocator snapshot
        allocator = {}
        try:
            allocator = json.loads(alloc_json) if alloc_json else {}
            if isinstance(allocator.get('regime'), str):
                allocator['regime'] = json.loads(allocator['regime'])
            if isinstance(allocator.get('weights'), str):
                allocator['weights'] = json.loads(allocator['weights'])
        except Exception:
            allocator = {}

        return render_template('index.html', balance=balance_filtered, balance_total_usdc=total_usdc, trades=trade_rows, positions=positions, allocator=allocator)

    @app.route('/train', methods=['GET', 'POST'])
    def train():
        logs_dir = Path('src/data/training_logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        runs_csv = logs_dir / 'runs.csv'

        trainers = [
            {
                'key': 'latin_training',
                'module': 'src.models.latin_training',
                'desc': 'Latin (ruptura mañana con ATR)'
            },
            {
                'key': 'latin_pattern_training',
                'module': 'src.models.latin_pattern_training',
                'desc': 'Patrón Latin (mañana↑, media↓ → tarde↑)'
            },
            {
                'key': 'cross_pattern_training',
                'module': 'src.models.cross_pattern_training',
                'desc': 'Patrón cruzado BTC→LTC (mañana BTC↑ & LTC no↑ → tarde LTC↑)'
            },
            {
                'key': 'walkforward_training',
                'module': 'src.models.walkforward_training',
                'desc': 'Walk-forward multi-model'
            },
            {
                'key': 'pairs_training',
                'module': 'src.models.pairs_training',
                'desc': 'Convergencia de pares (BTC/LTC)'
            },
        ]

        # Leer runs previos
        past_runs = []
        if runs_csv.exists():
            try:
                with runs_csv.open('r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Compute local timestamp for display
                        try:
                            tz_name = os.getenv('DASHBOARD_TZ', 'Europe/Madrid')
                            ts = row.get('timestamp', '')
                            if ts and len(ts) >= 16 and ts.endswith('Z'):
                                from zoneinfo import ZoneInfo
                                import datetime as _dt
                                y=int(ts[0:4]); mo=int(ts[4:6]); d=int(ts[6:8]); hh=int(ts[9:11]); mm=int(ts[11:13]); ss=int(ts[13:15])
                                dt = _dt.datetime(y,mo,d,hh,mm,ss,tzinfo=ZoneInfo('UTC')).astimezone(ZoneInfo(tz_name))
                                row['timestamp_local'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception:
                            row['timestamp_local'] = row.get('timestamp')
                        # Enriquecer con resumen si hay stdout
                        try:
                            stdout_rel = row.get('stdout') or ''
                            logs_dir = Path('src/data/training_logs')
                            stdout_path = logs_dir / Path(stdout_rel).name if stdout_rel else None
                            summary = {}
                            if stdout_path and stdout_path.exists():
                                txt = stdout_path.read_text(encoding='utf-8', errors='ignore')
                                # latin-pattern
                                if '[latin-pattern]' in txt:
                                    import re
                                    m = re.search(r"Salidas positivas.*:\s*(\d+)\s+de\s+(\d+)\s+\(([^\)]+)\)", txt)
                                    if m:
                                        summary['pos'] = m.group(1); summary['total'] = m.group(2); summary['pos_pct'] = m.group(3)
                                    m2 = re.search(r"\[latin-pattern\] best=([a-zA-Z0-9_]+) score=([0-9\.]+)", txt)
                                    if m2:
                                        summary['best_model'] = m2.group(1); summary['best_score'] = m2.group(2)
                                # cross-pattern
                                if '[cross-pattern]' in txt:
                                    import re
                                    m = re.search(r"Salidas positivas.*:\s*(\d+)\s+de\s+(\d+)\s+\(([^\)]+)\)", txt)
                                    if m:
                                        summary['pos'] = m.group(1); summary['total'] = m.group(2); summary['pos_pct'] = m.group(3)
                                    m2 = re.search(r"\[cross-pattern\] best=([a-zA-Z0-9_]+) score=([0-9\.]+)", txt)
                                    if m2:
                                        summary['best_model'] = m2.group(1); summary['best_score'] = m2.group(2)
                            row.update(summary)
                        except Exception:
                            pass
                        # Enriquecer desde CSV de resultados (fallback o detalle)
                        try:
                            logs_dir = Path('src/data/training_logs')
                            res_hint = row.get('results_csv')
                            trainer_key = row.get('trainer')
                            if not res_hint and trainer_key in ('latin_pattern_training','cross_pattern_training','walkforward_training'):
                                fname = 'latin_pattern_results.csv' if trainer_key=='latin_pattern_training' else ('cross_pattern_results.csv' if trainer_key=='cross_pattern_training' else 'walkforward_results.csv')
                                res_hint = str(logs_dir / fname)
                                row['results_csv'] = res_hint
                            if res_hint:
                                res_path = Path(res_hint)
                                if not res_path.is_absolute():
                                    res_path = logs_dir / Path(res_hint).name
                                if res_path.exists():
                                    with res_path.open('r', encoding='utf-8', newline='') as rf:
                                        rreader = csv.DictReader(rf)
                                        last = None
                                        for last in rreader:
                                            pass
                                        if last:
                                            for k in ('best_model','best_score','pos_pct'):
                                                if last.get(k): row[k] = last.get(k)
                                            if last.get('positives') and last.get('total'):
                                                row['pos'] = last['positives']; row['total'] = last['total']
                                            for m in ('lr','xgb','lgbm'):
                                                sc = last.get(f'{m}_score'); thr = last.get(f'{m}_thr')
                                                if sc not in (None, ''): row[f'{m}_score'] = sc
                                                if thr not in (None, ''): row[f'{m}_thr'] = thr
                                            if not row.get('best_model') and any(last.get(f'{m}_score') for m in ('lr','xgb','lgbm')):
                                                try:
                                                    vals = {m: float(last.get(f'{m}_score') or 0) for m in ('lr','xgb','lgbm')}
                                                    best_m = max(vals, key=vals.get)
                                                    row['best_model'] = best_m; row['best_score'] = f"{vals[best_m]:.4f}"
                                                except Exception:
                                                    pass
                        except Exception:
                            pass
                        # Enriquecer desde CSV de resultados si está disponible
                        try:
                            results_csv = row.get('results_csv')
                            if results_csv:
                                res_path = Path(results_csv)
                                if not res_path.is_absolute():
                                    res_path = Path('src/data/training_logs') / res_path.name
                                if res_path.exists():
                                    with res_path.open('r', encoding='utf-8', newline='') as rf:
                                        rreader = csv.DictReader(rf)
                                        last = None
                                        for last in rreader:
                                            pass
                                        if last:
                                            for k in ('best_model','best_score','pos_pct'):
                                                if last.get(k):
                                                    row[k] = last.get(k)
                                            if last.get('positives') and last.get('total'):
                                                row['pos'] = last['positives']
                                                row['total'] = last['total']
                                            for m in ('lr','xgb','lgbm'):
                                                sc = last.get(f'{m}_score')
                                                thr = last.get(f'{m}_thr')
                                                if sc is not None and sc != '':
                                                    row[f'{m}_score'] = sc
                                                if thr is not None and thr != '':
                                                    row[f'{m}_thr'] = thr
                                            if not row.get('best_model') and any(last.get(f'{m}_score') for m in ('lr','xgb','lgbm')):
                                                best_name, best_val = None, -1.0
                                                for m in ('lr','xgb','lgbm'):
                                                    try:
                                                        val = float(last.get(f'{m}_score') or 0)
                                                        if val > best_val:
                                                            best_val = val; best_name = m
                                                    except Exception:
                                                        pass
                                                if best_name:
                                                    row['best_model'] = best_name
                                                    row['best_score'] = f"{best_val:.4f}"
                        except Exception:
                            pass
                        past_runs.append(row)
                past_runs = list(reversed(past_runs))[:50]
            except Exception:
                past_runs = []

        if request.method == 'POST':
            trainer_key = request.form.get('trainer')
            extra_args = request.form.get('extra_args', '').strip()
            # Campos comunes
            timeframe = request.form.get('timeframe', '1h')
            fmt = request.form.get('format', 'parquet')
            tz = request.form.get('tz', '')
            symbols = request.form.get('symbols', '')
            base = request.form.get('base', '')
            target = request.form.get('target', '')
            start_date = request.form.get('start_date', '')
            end_date = request.form.get('end_date', '')

            trainer = next((t for t in trainers if t['key'] == trainer_key), None)
            if trainer is None:
                flash('Entrenador no válido', 'danger')
                return redirect(url_for('train'))

            # Validación de obligatorios según entrenador
            if trainer_key in ('latin_training','latin_pattern_training','walkforward_training'):
                if not symbols:
                    flash('El campo Símbolos es obligatorio para este entrenador.', 'danger')
                    return redirect(url_for('train'))
            if trainer_key == 'cross_pattern_training':
                if not base or not target:
                    flash('Los campos Base y Target son obligatorios para cross_pattern.', 'danger')
                    return redirect(url_for('train'))
            if trainer_key == 'pairs_training':
                if not request.form.get('pairs_symbols'):
                    flash('El campo Symbols (A,B) es obligatorio para pairs_training.', 'danger')
                    return redirect(url_for('train'))

            args = []
            # Componer argumentos básicos por tipo
            if trainer_key in ('latin_training', 'latin_pattern_training'):
                if symbols:
                    args += ['--symbols', symbols]
                if tz:
                    args += ['--tz', tz]
            if trainer_key == 'cross_pattern_training':
                if base:
                    args += ['--base', base]
                if target:
                    args += ['--target', target]
                if tz:
                    args += ['--tz', tz]

            # comunes
            if timeframe:
                args += ['--timeframe', timeframe]
            if fmt:
                args += ['--format', fmt]
            if start_date:
                args += ['--start_date', start_date]
            if end_date:
                args += ['--end_date', end_date]

            # Mapear parámetros específicos según entrenador
            if trainer_key == 'latin_training':
                if request.form.get('latin_horizon'):
                    args += ['--horizon', request.form.get('latin_horizon')]
                if request.form.get('latin_atr_target'):
                    args += ['--atr_mult_target', request.form.get('latin_atr_target')]
                if request.form.get('latin_atr_stop'):
                    args += ['--atr_mult_stop', request.form.get('latin_atr_stop')]
                if request.form.get('latin_morning_start'):
                    args += ['--morning_start', request.form.get('latin_morning_start')]
                if request.form.get('latin_morning_end'):
                    args += ['--morning_end', request.form.get('latin_morning_end')]
                if request.form.get('latin_session_start'):
                    args += ['--session_start', request.form.get('latin_session_start')]
                if request.form.get('latin_session_end'):
                    args += ['--session_end', request.form.get('latin_session_end')]
                if request.form.get('latin_models'):
                    args += ['--models', request.form.get('latin_models')]
                if request.form.get('latin_metric'):
                    args += ['--metric', request.form.get('latin_metric')]

            if trainer_key == 'latin_pattern_training':
                if request.form.get('lp_morning_start'):
                    args += ['--morning_start', request.form.get('lp_morning_start')]
                if request.form.get('lp_morning_end'):
                    args += ['--morning_end', request.form.get('lp_morning_end')]
                if request.form.get('lp_midday_start'):
                    args += ['--midday_start', request.form.get('lp_midday_start')]
                if request.form.get('lp_midday_end'):
                    args += ['--midday_end', request.form.get('lp_midday_end')]
                if request.form.get('lp_afternoon_start'):
                    args += ['--afternoon_start', request.form.get('lp_afternoon_start')]
                if request.form.get('lp_afternoon_end'):
                    args += ['--afternoon_end', request.form.get('lp_afternoon_end')]
                if request.form.get('lp_models'):
                    args += ['--models', request.form.get('lp_models')]
                if request.form.get('lp_metric'):
                    args += ['--metric', request.form.get('lp_metric')]
                if request.form.get('lp_threshold_mode'):
                    args += ['--threshold_mode', request.form.get('lp_threshold_mode')]
                if request.form.get('lp_split_by_day') == 'on':
                    args += ['--split_by_day']

            if trainer_key == 'cross_pattern_training':
                if request.form.get('cp_morning_start'):
                    args += ['--morning_start', request.form.get('cp_morning_start')]
                if request.form.get('cp_morning_end'):
                    args += ['--morning_end', request.form.get('cp_morning_end')]
                if request.form.get('cp_afternoon_start'):
                    args += ['--afternoon_start', request.form.get('cp_afternoon_start')]
                if request.form.get('cp_afternoon_end'):
                    args += ['--afternoon_end', request.form.get('cp_afternoon_end')]
                if request.form.get('cp_models'):
                    args += ['--models', request.form.get('cp_models')]
                if request.form.get('cp_metric'):
                    args += ['--metric', request.form.get('cp_metric')]
                if request.form.get('cp_threshold_mode'):
                    args += ['--threshold_mode', request.form.get('cp_threshold_mode')]
                if request.form.get('cp_split_by_day') == 'on':
                    args += ['--split_by_day']

            if trainer_key == 'walkforward_training':
                if request.form.get('wf_models'):
                    args += ['--models', request.form.get('wf_models')]
                if request.form.get('wf_folds'):
                    args += ['--folds', request.form.get('wf_folds')]
                if request.form.get('wf_horizon'):
                    args += ['--horizon', request.form.get('wf_horizon')]
                if request.form.get('wf_log_results'):
                    args += ['--log_results', request.form.get('wf_log_results')]

            if trainer_key == 'pairs_training':
                if request.form.get('pairs_symbols'):
                    args += ['--symbols', request.form.get('pairs_symbols')]
                if request.form.get('pairs_window'):
                    args += ['--window', request.form.get('pairs_window')]
                if request.form.get('pairs_horizon'):
                    args += ['--horizon', request.form.get('pairs_horizon')]

            if extra_args:
                # split simple por espacios respetando pares key value
                args += extra_args.split()

            # Ejecutar
            ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            mod = trainer['module']
            cmd = ['python', '-m', mod] + args
            log_path = logs_dir / f"{ts}_{trainer_key}.log"
            err_path = logs_dir / f"{ts}_{trainer_key}.err"
            try:
                # Auto set de CSV de resultados si el entrenador lo soporta
                auto_results = None
                if trainer_key in ('latin_pattern_training','cross_pattern_training','walkforward_training'):
                    # Por entrenador, define un CSV por tipo
                    out_name = 'latin_pattern_results.csv' if trainer_key=='latin_pattern_training' else ('cross_pattern_results.csv' if trainer_key=='cross_pattern_training' else 'walkforward_results.csv')
                    auto_results = logs_dir / out_name
                    # añadir --log_results también a walkforward
                    args += ['--log_results', str(auto_results)]

                # Asegurar CSV de resultados también para pairs_training y reconstruir cmd si se añadió --log_results
                if trainer_key == 'pairs_training':
                    try:
                        if ' --log_results ' not in (' ' + ' '.join(args) + ' '):
                            auto_results = logs_dir / 'pairs_results.csv'
                            args += ['--log_results', str(auto_results)]
                            cmd = ['python', '-m', mod] + args
                    except Exception:
                        pass
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd='.', timeout=None)
                log_path.write_text(proc.stdout or '', encoding='utf-8')
                err_path.write_text(proc.stderr or '', encoding='utf-8')
                # Parse training results from stdout and optional CSV
                best_model = ''
                best_score = ''
                pos = ''
                total = ''
                pos_pct = ''
                lr_score = xgb_score = lgbm_score = ''
                lr_thr = xgb_thr = lgbm_thr = ''
                try:
                    out_txt = proc.stdout or ''
                    import re, csv as _csv
                    if trainer_key == 'walkforward_training':
                        for m in ('lr','xgb','lgbm'):
                            macc = re.search(rf"\b{m}\s*:\s*accuracy=([0-9\.]+)", out_txt)
                            if macc:
                                val = macc.group(1)
                                if m=='lr': lr_score = val
                                if m=='xgb': xgb_score = val
                                if m=='lgbm': lgbm_score = val
                        try:
                            candidates = [(float(lr_score or 0), 'lr'), (float(xgb_score or 0), 'xgb'), (float(lgbm_score or 0), 'lgbm')]
                            best_val, best_name = max(candidates)
                            if best_val > 0:
                                best_model = best_name; best_score = f"{best_val:.4f}"
                        except Exception:
                            pass
                    elif trainer_key == 'pairs_training':
                        # Ejemplo: "[pairs] model=lr score=0.5939 metric=accuracy f1=0.6293"
                        mp = re.search(r"\[pairs\]\s*model=([a-zA-Z0-9_]+)\s*score=([0-9\.]+).*?f1=([0-9\.]+)", out_txt)
                        if mp:
                            best_model = f"pairs_{mp.group(1)}"; best_score = mp.group(2)
                            lr_score = mp.group(2)  # reutiliza columna visible
                        # Parsear 'support' del classification_report para Positivos/Total
                        # Líneas típicas:
                        # "           0      0.574     0.530     0.551      6313"
                        # "           1      0.609     0.651     0.629      7117"
                        m0 = re.search(r"^\s*0\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+(\d+)", out_txt, re.M)
                        m1 = re.search(r"^\s*1\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+(\d+)", out_txt, re.M)
                        try:
                            pos = m1.group(4) if m1 else ''
                            total = str(int((m0 and m0.group(4) or '0')) + int((m1 and m1.group(4) or '0')))
                            if pos and total and int(total) > 0:
                                pos_pct = f"{(int(pos)/int(total)*100):.2f}"
                        except Exception:
                            pass
                    else:
                        mpos = re.search(r"Salidas positivas.*:\s*(\d+)\s+de\s+(\d+)\s+\(([^\)]+)\)", out_txt)
                        if mpos:
                            pos, total, pos_pct = mpos.group(1), mpos.group(2), mpos.group(3)
                        mbest = re.search(r"\[(latin-pattern|cross-pattern)\] best=([a-zA-Z0-9_]+) score=([0-9\.]+)", out_txt)
                        if mbest:
                            best_model, best_score = mbest.group(2), mbest.group(3)
                    if auto_results and auto_results.exists():
                        with auto_results.open('r', encoding='utf-8', newline='') as rf:
                            rreader = _csv.DictReader(rf)
                            last = None
                            for last in rreader:
                                pass
                            if last:
                                lr_score = last.get('lr_score', lr_score) or lr_score
                                xgb_score = last.get('xgb_score', xgb_score) or xgb_score
                                lgbm_score = last.get('lgbm_score', lgbm_score) or lgbm_score
                                lr_thr = last.get('lr_thr', lr_thr) or lr_thr
                                xgb_thr = last.get('xgb_thr', xgb_thr) or xgb_thr
                                lgbm_thr = last.get('lgbm_thr', lgbm_thr) or lgbm_thr
                                best_model = last.get('best_model', best_model) or best_model
                                best_score = last.get('best_score', best_score) or best_score
                                pos = last.get('positives', pos) or pos
                                total = last.get('total', total) or total
                                pos_pct = last.get('pos_pct', pos_pct) or pos_pct
                except Exception:
                    pass
                # Build compact params string from cmd/fields
                def _flags_from_cmd(parts):
                    flags = {}
                    i = 0
                    while i < len(parts):
                        p = parts[i]
                        if isinstance(p, str) and p.startswith('--'):
                            key = p.lstrip('-')
                            nxt = parts[i+1] if i+1 < len(parts) else ''
                            if isinstance(nxt, str) and (nxt.startswith('--') or nxt == ''):
                                flags[key] = True
                                i += 1
                            else:
                                flags[key] = str(nxt)
                                i += 2
                        else:
                            i += 1
                    return flags
                flags = _flags_from_cmd(cmd)
                segs = []
                if symbols:
                    segs.append(f"symbols={symbols}")
                if base:
                    segs.append(f"base={base}")
                if target:
                    segs.append(f"target={target}")
                if timeframe:
                    segs.append(f"tf={timeframe}")
                if fmt:
                    segs.append(f"fmt={fmt}")
                if tz:
                    segs.append(f"tz={tz}")
                if start_date or end_date:
                    segs.append(f"dates={start_date or ''}..{end_date or ''}")
                if flags.get('morning_start') or flags.get('morning_end'):
                    segs.append(f"morning={flags.get('morning_start','')}–{flags.get('morning_end','')}")
                if flags.get('midday_start') or flags.get('midday_end'):
                    segs.append(f"midday={flags.get('midday_start','')}–{flags.get('midday_end','')}")
                if flags.get('afternoon_start') or flags.get('afternoon_end'):
                    segs.append(f"afternoon={flags.get('afternoon_start','')}–{flags.get('afternoon_end','')}")
                if flags.get('models'):
                    segs.append(f"models={flags.get('models')}")
                if flags.get('metric'):
                    segs.append(f"metric={flags.get('metric')}")
                if flags.get('threshold_mode'):
                    segs.append(f"mode={flags.get('threshold_mode')}")
                if 'split_by_day' in flags:
                    segs.append("split_by_day=1")
                if flags.get('horizon'):
                    segs.append(f"horizon={flags.get('horizon')}")
                if flags.get('atr_mult_target'):
                    segs.append(f"atr_take={flags.get('atr_mult_target')}")
                if flags.get('atr_mult_stop'):
                    segs.append(f"atr_stop={flags.get('atr_mult_stop')}")
                if flags.get('window'):
                    segs.append(f"window={flags.get('window')}")
                params_str = ' '.join(segs)
                # Append CSV
                header = ['timestamp','trainer','module','returncode','cmd','stdout','stderr','timeframe','format','tz','symbols','base','target','start_date','end_date','results_csv','params','best_model','best_score','pos','total','pos_pct','lr_score','xgb_score','lgbm_score','lr_thr','xgb_thr','lgbm_thr']
                write_header = not runs_csv.exists()
                # Si existe pero con cabecera antigua, archivar y forzar nueva cabecera
                if runs_csv.exists():
                    try:
                        first = runs_csv.open('r', encoding='utf-8').readline()
                        if 'results_csv' not in first or 'best_score' not in first:
                            legacy = runs_csv.parent / f"runs_legacy_{ts}.csv"
                            runs_csv.rename(legacy)
                            write_header = True
                    except Exception:
                        pass
                with runs_csv.open('a', encoding='utf-8', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=header)
                    if write_header:
                        w.writeheader()
                    w.writerow({
                        'timestamp': ts,
                        'trainer': trainer_key,
                        'module': mod,
                        'returncode': proc.returncode,
                        'cmd': ' '.join(cmd),
                        'stdout': f"logs/{log_path.name}",
                        'stderr': f"logs/{err_path.name}",
                          'timeframe': timeframe,
                          'format': fmt,
                          'tz': tz,
                          'symbols': symbols,
                          'base': base,
                          'target': target,
                          'start_date': start_date,
                          'end_date': end_date,
                          'results_csv': str(auto_results) if auto_results else '',
                          'params': params_str,
                          'best_model': best_model,
                          'best_score': best_score,
                          'pos': pos,
                          'total': total,
                          'pos_pct': pos_pct,
                          'lr_score': lr_score,
                          'xgb_score': xgb_score,
                          'lgbm_score': lgbm_score,
                          'lr_thr': lr_thr,
                          'xgb_thr': xgb_thr,
                          'lgbm_thr': lgbm_thr,
                      })
                flash(f"Entrenamiento '{trainer_key}' ejecutado (rc={proc.returncode}).", 'info')
                return redirect(url_for('train'))
            except Exception as e:
                err = f"Fallo al ejecutar entrenamiento: {e}"
                err_path.write_text(str(e), encoding='utf-8')
                flash(err, 'danger')
                return redirect(url_for('train'))

        return render_template('train.html', trainers=trainers, runs=past_runs)

    @app.route('/logs/<path:filename>')
    def logs(filename: str):
        logs_dir = Path('src/data/training_logs')
        return send_from_directory(logs_dir, filename, as_attachment=False)

    @app.route('/grid', methods=['GET','POST'])
    def grid():
        logs_dir = Path('src/data/training_logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        runs_csv = logs_dir / 'grid_runs.csv'

        past = []
        if runs_csv.exists():
            try:
                with runs_csv.open('r', encoding='utf-8', newline='') as f:
                    r = csv.DictReader(f)
                    for row in r:
                        # fecha local
                        try:
                            tz_name = os.getenv('DASHBOARD_TZ', 'Europe/Madrid')
                            ts = row.get('timestamp','')
                            if ts and ts.endswith('Z') and len(ts)>=16:
                                from zoneinfo import ZoneInfo
                                import datetime as _dt
                                y=int(ts[0:4]); mo=int(ts[4:6]); d=int(ts[6:8]); hh=int(ts[9:11]); mm=int(ts[11:13]); ss=int(ts[13:15])
                                dt = _dt.datetime(y,mo,d,hh,mm,ss,tzinfo=ZoneInfo('UTC')).astimezone(ZoneInfo(tz_name))
                                row['timestamp_local'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception:
                            pass
                        past.append(row)
                past = list(reversed(past))[:50]
            except Exception:
                past = []

        # Valores prellenados si inspeccionamos un pkl
        pre_spec = {}

        if request.method == 'POST':
            action = request.form.get('action', 'run')
            strategy = request.form.get('strategy', 'load_dispatch_swing')
            model_paths_json = request.form.get('model_paths_json', '').strip()
            model_pkl_single = request.form.get('model_pkl_single', '').strip()
            symbol = request.form.get('symbol', '').strip()
            timeframe = request.form.get('timeframe', '').strip()
            horizon = request.form.get('horizon', '').strip()
            start_date = request.form.get('start_date', '').strip()
            end_date = request.form.get('end_date', '').strip()
            thr_up = request.form.get('thr_up', '').strip()
            thr_down = request.form.get('thr_down', '').strip()

            # Inspeccionar pkl para prellenar
            if action == 'inspect' and model_pkl_single:
                try:
                    from joblib import load as _load
                    obj = _load(model_pkl_single)
                    spec = obj.get('spec') if isinstance(obj, dict) else None
                    if spec:
                        pre_spec = {
                            'symbol': getattr(spec, 'symbol', ''),
                            'timeframe': getattr(spec, 'timeframe', ''),
                            'horizon': getattr(spec, 'horizon', ''),
                        }
                        flash(f"Modelo cargado: {pre_spec}", 'info')
                except Exception as e:
                    flash(f"No se pudo leer el modelo: {e}", 'danger')
                return render_template('grid.html', runs=past, pre=pre_spec)

            # Ejecutar backtest rápido mapeando fechas a días
            try:
                from datetime import datetime as _dt
                days = 60
                if start_date and end_date:
                    sd = _dt.fromisoformat(start_date)
                    ed = _dt.fromisoformat(end_date)
                    days = max(1, (ed - sd).days)
            except Exception:
                days = 60

            ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            log_path = logs_dir / f"{ts}_grid.log"
            err_path = logs_dir / f"{ts}_grid.err"
            out_txt = ''
            rc = 0
            try:
                if strategy == 'load_dispatch_swing':
                    # Usar backtest con symbol/timeframe/horizon
                    if model_pkl_single and not model_paths_json:
                        # Si hay un pkl, intenta inferir la spec
                        try:
                            from joblib import load as _load
                            obj = _load(model_pkl_single)
                            spec = obj.get('spec') if isinstance(obj, dict) else None
                            if spec:
                                symbol = symbol or getattr(spec, 'symbol', symbol)
                                timeframe = timeframe or getattr(spec, 'timeframe', timeframe)
                                horizon = horizon or getattr(spec, 'horizon', horizon)
                        except Exception:
                            pass
                    thr_up_val = thr_up if thr_up else '0.0005'
                    thr_down_val = thr_down if thr_down else '-0.0005'
                    code = (
                        "from src.backtests.backtest_strategies import backtest_load_dispatch_swing; "
                        f"backtest_load_dispatch_swing('{symbol or 'BTC/USDC'}','{timeframe or '1h'}','{horizon or '1d'}',{days}, initial_capital=1000.0, thr_up={thr_up_val}, thr_down={thr_down_val})"
                    )
                else:
                    code = f"from src.backtests.backtest_strategies import backtest_peak_shaving_scalping; backtest_peak_shaving_scalping('{symbol or 'BTC/USDC'}','{timeframe or '1m'}',{days})"

                proc = subprocess.run(['python','-c', code], capture_output=True, text=True, cwd='.', timeout=None)
                out_txt = (proc.stdout or '') + '\n' + (proc.stderr or '')
                rc = proc.returncode
                log_path.write_text(out_txt, encoding='utf-8')
                err_path.write_text(proc.stderr or '', encoding='utf-8')
                header = ['timestamp','strategy','symbol','timeframe','horizon','start_date','end_date','days','stdout','stderr','returncode']
                write_header = not runs_csv.exists()
                with runs_csv.open('a', encoding='utf-8', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=header)
                    if write_header:
                        w.writeheader()
                    w.writerow({
                        'timestamp': ts,
                        'strategy': strategy,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'horizon': horizon,
                        'start_date': start_date,
                        'end_date': end_date,
                        'days': days,
                        'stdout': f"logs/{log_path.name}",
                        'stderr': f"logs/{err_path.name}",
                        'returncode': rc,
                    })
                flash(f"Grid run '{strategy}' ejecutado (rc={rc}).", 'info')
                return redirect(url_for('grid'))
            except Exception as e:
                err = f"Fallo grid run: {e}"
                err_path.write_text(str(e), encoding='utf-8')
                flash(err, 'danger')
                return redirect(url_for('grid'))

        return render_template('grid.html', runs=past, pre=pre_spec)

    @app.route('/ai', methods=['GET','POST'])
    def ai_check():
        logs_dir = Path('src/data/training_logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        runs_csv = logs_dir / 'ai_runs.csv'

        past = []
        if runs_csv.exists():
            try:
                with runs_csv.open('r', encoding='utf-8', newline='') as f:
                    r = csv.DictReader(f)
                    for row in r:
                        # fecha local
                        try:
                            tz_name = os.getenv('DASHBOARD_TZ', 'Europe/Madrid')
                            ts = row.get('timestamp','')
                            if ts and ts.endswith('Z') and len(ts)>=16:
                                from zoneinfo import ZoneInfo
                                import datetime as _dt
                                y=int(ts[0:4]); mo=int(ts[4:6]); d=int(ts[6:8]); hh=int(ts[9:11]); mm=int(ts[11:13]); ss=int(ts[13:15])
                                dt = _dt.datetime(y,mo,d,hh,mm,ss,tzinfo=ZoneInfo('UTC')).astimezone(ZoneInfo(tz_name))
                                row['timestamp_local'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception:
                            pass
                        past.append(row)
                past = list(reversed(past))[:50]
            except Exception:
                past = []

        if request.method == 'POST':
            ai_mode = request.form.get('ai_mode','single')
            symbols = request.form.get('symbols','BTCUSDC,ETHUSDC,LTCUSDC')
            timeframe = request.form.get('timeframe','1h')
            fmt = request.form.get('format','parquet')
            start_date = request.form.get('start_date','')
            end_date = request.form.get('end_date','')
            initial_capital = request.form.get('initial_capital','300')
            buy_thresh = request.form.get('buy_thresh','0.6')
            sell_thresh = request.form.get('sell_thresh','0.4')
            model_path = request.form.get('model_path','')
            pairs_symbols = request.form.get('pairs_symbols','')
            z_entry = request.form.get('z_entry','2.0')
            z_exit = request.form.get('z_exit','0.5')
            prob_thresh = request.form.get('prob_thresh','0.6')
            pairs_model_path = request.form.get('pairs_model_path','')
            pairs_window = request.form.get('pairs_window','50')
            pairs_max_hold = request.form.get('pairs_max_hold','240')
            spot_mode = (request.form.get('spot_mode','false') or 'false').strip().lower()

            if ai_mode == 'pairs':
                # Construir A,B de forma robusta: usa pairs_symbols si viene; si no, toma los 2 primeros de symbols; fallback por defecto
                syms_src = pairs_symbols if pairs_symbols else symbols
                parts = [s.strip().upper() for s in (syms_src or '').split(',') if s.strip()]
                pair_ab = (parts[0] + ',' + parts[1]) if len(parts) >= 2 else 'BTCUSDC,LTCUSDC'
                cmd = ['python','-m','src.backtests.pairs_portfolio_check',
                       '--symbols', pair_ab,
                       '--timeframe', timeframe,
                       '--format', fmt,
                       '--initial_capital', str(initial_capital),
                       '--z_entry', str(z_entry),
                       '--z_exit', str(z_exit),
                       '--prob_thresh', str(prob_thresh),
                       '--window', str(pairs_window),
                       '--max_hold', str(pairs_max_hold)]
                if start_date:
                    cmd += ['--start_date', start_date]
                if end_date:
                    cmd += ['--end_date', end_date]
                if pairs_model_path:
                    cmd += ['--model_path', pairs_model_path]
                elif model_path:
                    cmd += ['--model_path', model_path]
                if spot_mode == 'true':
                    cmd += ['--spot_mode']
            else:
                cmd = ['python','-m','src.backtests.ai_portfolio_check',
                       '--symbols', symbols,
                       '--timeframe', timeframe,
                       '--format', fmt,
                       '--initial_capital', str(initial_capital),
                       '--buy_thresh', str(buy_thresh),
                       '--sell_thresh', str(sell_thresh)]
                if start_date:
                    cmd += ['--start_date', start_date]
                if end_date:
                    cmd += ['--end_date', end_date]
                if model_path:
                    cmd += ['--model_path', model_path]

            ts = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
            log_path = logs_dir / f"{ts}_ai.log"
            err_path = logs_dir / f"{ts}_ai.err"
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, cwd='.', timeout=None)
                out = proc.stdout or ''
                log_path.write_text(out, encoding='utf-8')
                err_path.write_text(proc.stderr or '', encoding='utf-8')
                # Detectar salidas
                csv_path = ''
                png_path = ''
                ops_long = ''
                ops_short = ''
                fees_usdc = ''
                import re
                m1 = re.search(r"saved CSV:\s*(.+)", out)
                if m1:
                    csv_path = m1.group(1).strip()
                m2 = re.search(r"saved PNG:\s*(.+)", out)
                if m2:
                    png_path = m2.group(1).strip()
                m3 = re.search(r"ops long=(\d+)\s+short=(\d+)\s+fees_usdc=([0-9\.]+)", out)
                if m3:
                    ops_long, ops_short, fees_usdc = m3.group(1), m3.group(2), m3.group(3)

                header = ['timestamp','mode','symbols','timeframe','format','start_date','end_date','initial_capital','buy_thresh','sell_thresh','z_entry','z_exit','prob_thresh','spot_mode','ops_long','ops_short','fees_usdc','model_path','csv','png','rc','stdout','stderr']
                write_header = not runs_csv.exists()
                with runs_csv.open('a', encoding='utf-8', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=header)
                    if write_header:
                        w.writeheader()
                    w.writerow({
                        'timestamp': ts,
                        'mode': ai_mode,
                        'symbols': symbols,
                        'timeframe': timeframe,
                        'format': fmt,
                        'start_date': start_date,
                        'end_date': end_date,
                        'initial_capital': initial_capital,
                        'buy_thresh': buy_thresh,
                        'sell_thresh': sell_thresh,
                        'z_entry': z_entry if ai_mode=='pairs' else '',
                        'z_exit': z_exit if ai_mode=='pairs' else '',
                        'prob_thresh': prob_thresh if ai_mode=='pairs' else '',
                        'spot_mode': spot_mode if ai_mode=='pairs' else '',
                        'ops_long': ops_long if ai_mode=='pairs' else '',
                        'ops_short': ops_short if ai_mode=='pairs' else '',
                        'fees_usdc': fees_usdc if ai_mode=='pairs' else '',
                        'model_path': pairs_model_path if ai_mode=='pairs' else model_path,
                        'csv': csv_path,
                        'png': png_path,
                        'rc': proc.returncode,
                        'stdout': f"logs/{log_path.name}",
                        'stderr': f"logs/{err_path.name}",
                    })
                flash(f"AI check ejecutado (rc={proc.returncode}).", 'info')
                return redirect(url_for('ai_check'))
            except Exception as e:
                err = f"Fallo AI check: {e}"
                err_path.write_text(str(e), encoding='utf-8')
                flash(err, 'danger')
                return redirect(url_for('ai_check'))

        return render_template('ai.html', runs=past)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
