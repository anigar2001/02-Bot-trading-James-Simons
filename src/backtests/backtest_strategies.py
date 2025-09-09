from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy


@dataclass
class BTResult:
    symbol: str
    trades: int
    winrate: float
    pnl: float
    ret_pct: float


def simulate_series(df: pd.DataFrame, strategy, initial_capital: float = 300.0, size_quote: float = 15.0) -> BTResult:
    equity = initial_capital
    pos_side: Optional[str] = None
    qty: float = 0.0
    entry: float = 0.0
    stop: Optional[float] = None
    take: Optional[float] = None
    trades = 0
    wins = 0

    for i in range(250, len(df)):
        window = df.iloc[: i + 1]
        md = {strategy.symbol: window}
        sig = strategy.check_signal(md)
        close = float(window["close"].iloc[-1])
        high = float(window["high"].iloc[-1])
        low = float(window["low"].iloc[-1])

        # Gestión de posición existente
        if pos_side:
            # Chequear stop/take
            exit_price = None
            if pos_side == "BUY":
                if stop and low <= stop:
                    exit_price = stop
                elif take and high >= take:
                    exit_price = take
            else:  # SELL
                if stop and high >= stop:
                    exit_price = stop
                elif take and low <= take:
                    exit_price = take
            # Opposite signal cierra a close
            if not exit_price and ((pos_side == "BUY" and sig.action == "SELL") or (pos_side == "SELL" and sig.action == "BUY")):
                exit_price = close

            if exit_price:
                pnl = (exit_price - entry) * qty if pos_side == "BUY" else (entry - exit_price) * qty
                equity += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                pos_side = None
                qty = 0.0
                entry = 0.0
                stop = None
                take = None
                continue

        # Sin posición: evaluar entrada
        if not pos_side and sig.action in ("BUY", "SELL"):
            pos_side = sig.action
            entry = close
            qty = size_quote / close
            stop = sig.stop_loss
            take = sig.take_profit

    ret = (equity - initial_capital) / initial_capital * 100
    winrate = (wins / trades * 100) if trades > 0 else 0.0
    return BTResult(symbol=strategy.symbol, trades=trades, winrate=winrate, pnl=equity - initial_capital, ret_pct=ret)


def run_quick_backtest(client, symbols: List[str], timeframe: str = "1h"):
    print("== Backtest rápido ==")
    for s in symbols:
        df = client.fetch_ohlcv_df(s, timeframe=timeframe, limit=1000)
        if len(df) < 300:
            print(f"{s}: insuficiente historial para backtest")
            continue
        mr = MeanReversionStrategy(symbol=s)
        mo = MomentumStrategy(symbol=s)
        r1 = simulate_series(df.copy(), mr)
        r2 = simulate_series(df.copy(), mo)
        print(f"{s} | MR -> trades:{r1.trades} winrate:{r1.winrate:.1f}% pnl:{r1.pnl:.2f} ret:{r1.ret_pct:.2f}%")
        print(f"{s} | MO -> trades:{r2.trades} winrate:{r2.winrate:.1f}% pnl:{r2.pnl:.2f} ret:{r2.ret_pct:.2f}%")


def _load_partition(symbol_key: str, tf: str) -> pd.DataFrame:
    from pathlib import Path
    base = Path('src/data/historical/partitioned')/symbol_key/tf
    files = sorted(base.glob('*.parquet'))
    if not files:
        raise SystemExit(f"[bkt] no data for {symbol_key} {tf}")
    frames = []
    for p in files:
        df = pd.read_parquet(p)
        idx = pd.to_datetime(df['timestamp'], utc=True)
        df = df[['open','high','low','close','volume']].copy()
        df.index = idx
        frames.append(df)
    out = pd.concat(frames, axis=0).sort_index()
    return out[~out.index.duplicated(keep='last')]


def backtest_load_dispatch_swing(symbol: str, tf_main: str = '1h', horizon: str = '1d', days: int = 60,
                                 initial_capital: float = 1000.0, thr_up: float = 0.0005, thr_down: float = -0.0005):
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import matplotlib.pyplot as plt
    from src.models.load_quantile_forecaster import LoadQuantileForecaster
    from src.utils.indicators import add_basic_indicators
    from src.strategies.load_dispatch_swing import LoadDispatchSwingStrategy, LoadDispatchConfig
    sym_key = symbol.replace('/', '')
    df = _load_partition(sym_key, tf_main).iloc[-days*24:]
    df_feat = add_basic_indicators(df).dropna()
    model_path = f"src/models/quantile_models/{sym_key}_{tf_main}_{horizon}_load_qf.pkl"
    try:
        strat = LoadDispatchSwingStrategy(LoadDispatchConfig(q_model_paths={horizon: model_path}, thr_up=thr_up, thr_down=thr_down))
    except Exception:
        print(f"[bkt] modelo no encontrado: {model_path}")
        return
    cash = float(initial_capital); qty = 0.0; val = []; cash_series = []; qty_series = []
    rsi_list = []; adx_list = []; bbw_list = []; q10_list = []; q50_list = []; q90_list = []; unc_list = []; action_list = []; size_list = []
    n_buy = 0; n_sell = 0
    for i in range(200, len(df_feat)):
        feats = {horizon: df_feat.iloc[:i+1]}
        sig = strat.infer(feats)
        px = float(df_feat['close'].iloc[i])
        if sig['action'] == 'BUY' and cash > 0:
            size = min(cash, cash * min(1.0, sig['size']))
            buy_q = size / px
            qty += buy_q; cash -= size; n_buy += 1
        elif sig['action'] == 'SELL' and qty > 0:
            cash += qty * px; qty = 0.0
            n_sell += 1
        val.append(cash + qty * px)
        cash_series.append(cash)
        qty_series.append(qty)
        # Métricas para diagnóstico
        q10, q50, q90 = sig.get('quantiles', (None, None, None))
        q10_list.append(q10); q50_list.append(q50); q90_list.append(q90)
        unc_list.append(sig.get('uncertainty'))
        rsi_list.append(sig.get('rsi'))
        adx_list.append(sig.get('adx'))
        bbw_list.append(sig.get('bb_width'))
        action_list.append(sig.get('action'))
        size_list.append(sig.get('size'))
    if val:
        arr = np.array(val)
        ret = arr[-1]/arr[0]-1
        dd = float(np.max(np.maximum.accumulate(arr) - arr) / np.maximum.accumulate(arr)[-1])
        print(f"[bkt] LoadDispatch {symbol} PnL={ret:.3f} MaxDD={dd:.3f} buys={n_buy} sells={n_sell}")
        # Guardar CSV/PNG como en AI Portfolio Check
        out_dir = Path('src/data/training_logs'); out_dir.mkdir(parents=True, exist_ok=True)
        idx = df_feat.index[200:200+len(val)]
        df_out = pd.DataFrame({
            'value_usdc': val,
            'cash': cash_series,
            f'qty_{sym_key}': qty_series,
            'action': action_list,
            'size': size_list,
            'q10': q10_list,
            'q50': q50_list,
            'q90': q90_list,
            'uncertainty': unc_list,
            'rsi': rsi_list,
            'adx': adx_list,
            'bb_width': bbw_list,
        }, index=idx)
        tag = f"ld_{sym_key}_{tf_main}_{horizon}_{days}d"
        csv_path = out_dir / f"equity_{tag}.csv"
        df_out.to_csv(csv_path)
        plt.figure(figsize=(10,4))
        df_out['value_usdc'].plot()
        plt.title(f"LoadDispatch Equity {symbol} tf={tf_main} hor={horizon} days={days}")
        plt.ylabel('USDC'); plt.grid(True, alpha=0.3); plt.tight_layout()
        png_path = out_dir / f"equity_{tag}.png"
        plt.savefig(png_path)
        print(f"[bkt] saved CSV: {csv_path}")
        print(f"[bkt] saved PNG: {png_path}")


def backtest_peak_shaving_scalping(symbol: str, tf: str = '1m', days: int = 14):
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import matplotlib.pyplot as plt
    from src.strategies.peak_shaving_scalping import PeakShavingScalpingStrategy, PeakShavingConfig
    sym_key = symbol.replace('/', '')
    df = _load_partition(sym_key, tf).iloc[-days*24*60:]
    strat = PeakShavingScalpingStrategy(PeakShavingConfig())
    cash = 1000.0; qty = 0.0; val = []; cash_series = []; qty_series = []
    ent_list = []; adx_list = []; exp_list = []; action_list = []
    for i in range(200, len(df)):
        px = float(df['close'].iloc[i])
        sig = strat.infer(df.iloc[:i+1])
        if sig['action'] == 'BUY' and cash > 0:
            size = cash * 0.5
            buy_q = size / px
            qty += buy_q; cash -= size
        elif sig['action'] == 'SELL' and qty > 0:
            cash += qty * px; qty = 0.0
        val.append(cash + qty * px)
        cash_series.append(cash)
        qty_series.append(qty)
        ent_list.append(sig.get('entropy'))
        adx_list.append(sig.get('adx'))
        exp_list.append(sig.get('exp_ret_proxy'))
        action_list.append(sig.get('action'))
    if val:
        arr = np.array(val)
        ret = arr[-1]/arr[0]-1
        dd = float(np.max(np.maximum.accumulate(arr) - arr) / np.maximum.accumulate(arr)[-1])
        print(f"[bkt] PeakShaving {symbol} PnL={ret:.3f} MaxDD={dd:.3f}")
        out_dir = Path('src/data/training_logs'); out_dir.mkdir(parents=True, exist_ok=True)
        idx = df.index[200:200+len(val)]
        df_out = pd.DataFrame({'value_usdc': val, 'cash': cash_series, f'qty_{sym_key}': qty_series, 'action': action_list, 'entropy': ent_list, 'adx': adx_list, 'exp_ret': exp_list}, index=idx)
        tag = f"ps_{sym_key}_{tf}_{days}d"
        csv_path = out_dir / f"equity_{tag}.csv"; df_out.to_csv(csv_path)
        plt.figure(figsize=(10,4)); df_out['value_usdc'].plot()
        plt.title(f"PeakShaving Equity {symbol} tf={tf} days={days}")
        plt.ylabel('USDC'); plt.grid(True, alpha=0.3); plt.tight_layout()
        png_path = out_dir / f"equity_{tag}.png"; plt.savefig(png_path)
        print(f"[bkt] saved CSV: {csv_path}")
        print(f"[bkt] saved PNG: {png_path}")
