import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Carga de variables de entorno
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

from src.utils.api import BinanceClient
from src.utils.helpers import setup_logging, json_log, ensure_dirs
from src.utils.risk import PositionManager, Position
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.pairs_arbitrage import PairsArbitrageStrategy
from src.strategies.latin_breakout import LatinBreakoutStrategy
from src.models.predictor import MLSignal
from src.meta import Allocator, PerformanceTracker, detect_regime, compute_pairs_z
from src.data.persist import append_1m, append_resampled_from_1m


def parse_args():
    parser = argparse.ArgumentParser(description="Bot de trading cuantitativo en Binance Testnet")
    parser.add_argument("--mode", choices=["live", "backtest"], default="live", help="Modo de ejecución")
    parser.add_argument("--symbols", default="BTC/USDC,ETH/USDC,LTC/USDC", help="Símbolos separados por coma")
    parser.add_argument("--timeframe", default="1m", help="Timeframe para OHLCV (e.g., 1m,5m,1h)")
    parser.add_argument("--strategy", choices=["mean", "momentum", "pairs", "latin", "load_dispatch_swing", "peak_shaving_scalping", "all"], default="all")
    parser.add_argument("--interval", type=int, default=30, help="Segundos entre iteraciones en modo live")
    parser.add_argument("--dry", action="store_true", help="No enviar órdenes reales (solo simular)")
    return parser.parse_args()


def load_env():
    if load_dotenv:
        load_dotenv()
    # Variables necesarias
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    api_base = os.getenv("API_BASE", "https://testnet.binance.vision")
    return api_key, api_secret, api_base


def build_strategies(name: str, symbols: List[str], ml=None, ml_buy_thresh: float = 0.55, ml_sell_thresh: float = 0.45):
    strategies = []
    if name in ("mean", "all"):
        for s in symbols:
            strategies.append(MeanReversionStrategy(symbol=s, ml=ml, ml_buy_thresh=ml_buy_thresh, ml_sell_thresh=ml_sell_thresh))
    if name in ("momentum", "all"):
        for s in symbols:
            strategies.append(MomentumStrategy(symbol=s, ml=ml, ml_buy_thresh=ml_buy_thresh, ml_sell_thresh=ml_sell_thresh))
    if name in ("pairs", "all"):
        # Pairs BTC/LTC usando USDC de referencia
        if "BTC/USDC" in symbols and "LTC/USDC" in symbols:
            strategies.append(PairsArbitrageStrategy(symbol_a="BTC/USDC", symbol_b="LTC/USDC"))
    if name in ("latin", "all"):
        for s in symbols:
            strategies.append(LatinBreakoutStrategy(symbol=s))
    return strategies


def main():
    args = parse_args()
    ensure_dirs(["src/data/logs", "src/data/historical", "src/models", "src/dashboard/static", "src/dashboard/templates"])
    logger = setup_logging(log_dir="src/data/logs")

    api_key, api_secret, api_base = load_env()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    logger.info(f"Modo: {args.mode} | Estrategia: {args.strategy} | Símbolos: {symbols} | TF: {args.timeframe}")

    # Si faltan claves, forzamos dry_run para evitar errores en órdenes privadas
    missing_keys = (not api_key) or (not api_secret)
    effective_dry = args.dry or missing_keys
    if missing_keys:
        logger.warning("BINANCE_API_KEY/SECRET no configuradas. Se fuerza --dry (simulación de órdenes).")
    client = BinanceClient(api_key=api_key, api_secret=api_secret, api_base=api_base, enable_rate_limit=True, testnet=True, dry_run=effective_dry)
    client.load_markets()

    pos_manager = PositionManager(initial_capital=float(os.getenv("INITIAL_CAPITAL", 300.0)), risk_per_trade=float(os.getenv("RISK_PER_TRADE", 0.02)), max_positions=int(os.getenv("MAX_OPEN_POSITIONS", 3)))

    if args.mode == "backtest":
        # Backtest simple (por rapidez); el módulo backtests ofrece más detalle
        from src.backtests.backtest_strategies import run_quick_backtest
        run_quick_backtest(client, symbols, args.timeframe)
        return

    use_ml = os.getenv("USE_ML", "0") == "1"
    ml_buy_thresh = float(os.getenv("ML_BUY_THRESH", 0.55))
    ml_sell_thresh = float(os.getenv("ML_SELL_THRESH", 0.45))
    ml = MLSignal() if use_ml else None

    strategies = [] if args.strategy in ("load_dispatch_swing", "peak_shaving_scalping") else build_strategies(args.strategy, symbols, ml=ml, ml_buy_thresh=ml_buy_thresh, ml_sell_thresh=ml_sell_thresh)

    # Meta-asignador por régimen (Fase 1 heurística) y tracker de performance
    perf_tracker = PerformanceTracker()
    allocator = Allocator(perf_tracker)
    logger.info(f"Estrategias cargadas: {[type(s).__name__ for s in strategies]}")

    # Bucle principal de trading (polling)
    while True:
        loop_start = time.time()
        try:
            # Estrategias 'grid-load' personalizadas (se ejecutan antes del flujo estándar)
            if args.strategy == "load_dispatch_swing":
                import json as _json
                from src.strategies.load_dispatch_swing import LoadDispatchSwingStrategy, LoadDispatchConfig
                from src.utils.indicators import add_basic_indicators
                paths_json = os.getenv('LOAD_QF_MODEL_PATHS', '{}')
                try:
                    model_paths = _json.loads(paths_json) if paths_json else {}
                except Exception:
                    model_paths = {}
                strat = LoadDispatchSwingStrategy(LoadDispatchConfig(q_model_paths=model_paths, thr_up=float(os.getenv('LOAD_QF_THR_UP', '0.0005')), thr_down=float(os.getenv('LOAD_QF_THR_DOWN', '-0.0005')), max_pos=float(os.getenv('LOAD_QF_MAX_POS', '1.0'))))
                for s in symbols:
                    feats_by_tf: Dict[str, pd.DataFrame] = {}
                    for tf in model_paths.keys():
                        try:
                            tf_eff = tf if tf != '1d' else '1h'
                            df_tf = client.fetch_ohlcv_df(s, timeframe=tf_eff, limit=800)
                            feats_by_tf[tf] = add_basic_indicators(df_tf).dropna()
                        except Exception:
                            continue
                    if feats_by_tf:
                        res = strat.infer(feats_by_tf)
                        quote_bal = client.get_quote_balance("USDC")
                        base_size = pos_manager.size_in_quote(quote_balance=quote_bal)
                        size_quote = base_size * max(0.05, min(1.0, float(res.get('size', 0.0))))
                        qty = client.quote_to_amount(s, size_quote)
                        if qty is not None and res['action'] in ("BUY", "SELL"):
                            if not client.dry_run:
                                order_res = client.create_market_order(s, side=res['action'].lower(), amount=qty)
                            else:
                                order_res = {"status": "simulated", "symbol": s, "side": res['action'].lower(), "amount": qty}
                            json_log("src/data/logs/trades.jsonl", {"ts": datetime.utcnow().isoformat(), "strategy": "LoadDispatchSwing", "signal": res, "order": order_res})
                time.sleep(max(0, args.interval - (time.time() - loop_start)))
                continue

            if args.strategy == "peak_shaving_scalping":
                from src.strategies.peak_shaving_scalping import PeakShavingScalpingStrategy, PeakShavingConfig
                strat = PeakShavingScalpingStrategy(PeakShavingConfig(ent_max=float(os.getenv('PEAK_ENT_MAX', '1.8')), adx_min=int(os.getenv('PEAK_ADX_MIN', '18'))))
                for s in symbols:
                    try:
                        df_1m = client.fetch_ohlcv_df(s, timeframe='1m', limit=400)
                    except Exception:
                        continue
                    res = strat.infer(df_1m)
                    quote_bal = client.get_quote_balance("USDC")
                    base_size = pos_manager.size_in_quote(quote_balance=quote_bal)
                    size_quote = base_size * 0.3
                    qty = client.quote_to_amount(s, size_quote)
                    if qty is not None and res['action'] in ("BUY", "SELL"):
                        if not client.dry_run:
                            order_res = client.create_market_order(s, side=res['action'].lower(), amount=qty)
                        else:
                            order_res = {"status": "simulated", "symbol": s, "side": res['action'].lower(), "amount": qty}
                        json_log("src/data/logs/trades.jsonl", {"ts": datetime.utcnow().isoformat(), "strategy": "PeakShavingScalping", "signal": res, "order": order_res})
                time.sleep(max(0, args.interval - (time.time() - loop_start)))
                continue
            # Preparar datos por símbolo (incluye tf superior para S/R)
            symbol_data: Dict[str, pd.DataFrame] = {}
            high_tf_data: Dict[str, pd.DataFrame] = {}
            for s in symbols:
                symbol_data[s] = client.fetch_ohlcv_df(s, timeframe=args.timeframe, limit=300)
                # Temporalidad superior (p.ej., si base es 1m -> 15m; 5m -> 1h; por defecto 15m)
                base_tf = args.timeframe
                tf_map = {"1m": "15m", "5m": "1h", "15m": "1h", "1h": "4h", "4h": "1d"}
                high_tf = tf_map.get(base_tf, "15m")
                high_tf_data[s] = client.fetch_ohlcv_df(s, timeframe=high_tf, limit=400)

            # Persistencia incremental de 1m y resample a TF superiores (opcional)
            try:
                persist_enabled = os.getenv("PERSIST_1M", "1") == "1"
                persist_targets = os.getenv("PERSIST_RESAMPLE_TARGETS", "5m,15m,1h,4h")
                target_list = [t.strip() for t in persist_targets.split(',') if t.strip()]
                if persist_enabled and args.timeframe == "1m":
                    for s in symbols:
                        df1 = symbol_data[s][["open","high","low","close","volume"]]
                        # Asegurar índice datetime UTC
                        if not isinstance(df1.index, pd.DatetimeIndex):
                            df1.index = pd.to_datetime(df1.index, utc=True)
                        appended = append_1m(s, df1)
                        if appended:
                            append_resampled_from_1m(s, df1, targets=target_list)
            except Exception:
                pass

            # Pairs z-score (si aplica)
            pairs_z = None
            if "BTC/USDC" in symbol_data and "LTC/USDC" in symbol_data:
                pairs_z = compute_pairs_z(symbol_data["BTC/USDC"], symbol_data["LTC/USDC"], window=50)

            # Calcular régimen por símbolo y estimar pesos globales (promedio simple por símbolo)
            regimes = []
            for s in symbols:
                last_price = float(symbol_data[s]['close'].iloc[-1])
                regimes.append(detect_regime(symbol_data[s], high_tf_data[s], last_price, pairs_z, sr_window=allocator.sr_window, sr_proximity_pct=allocator.sr_prox_pct))
            # Promediar indicadores de régimen
            if regimes:
                avg_trend = sum(r.trend_strength for r in regimes) / len(regimes)
                avg_vol = sum(r.vol for r in regimes) / len(regimes)
                any_near_sr = any(r.near_sr for r in regimes)
                regime_avg = type(regimes[0])(trend_strength=avg_trend, vol=avg_vol, near_sr=any_near_sr, pairs_z=pairs_z)
            else:
                regime_avg = None

            weights = allocator.weights(regime_avg) if regime_avg else None

            for strat in strategies:
                # Determinamos símbolos requeridos según tipo de estrategia
                req_symbols = strat.required_symbols()
                data: Dict[str, pd.DataFrame] = {sym: symbol_data[sym] for sym in req_symbols if sym in symbol_data}

                signal = strat.check_signal(data)

                # Gestionar primero señales multi‑leg (pares), incluso si action es HOLD
                if getattr(signal, "multi_leg", False):  # Para arbitraje de pares
                    # Nota: muchas estrategias multi‑leg señalan HOLD pero con legs adjuntas
                    if pos_manager.can_open_new():
                        # Para cada pata calculamos tamaño equivalente
                        # Simplificación: asignar la mitad del riesgo a cada pata
                        quote_bal = client.get_quote_balance("USDC")
                        base_size = pos_manager.size_in_quote(quote_balance=quote_bal, legs=2)
                        w_pairs = (weights.pairs if weights else 0.1)
                        size_quote = base_size * max(0.05, w_pairs)  # asegura mínimo
                        orders = []
                        for leg in signal.legs:
                            qty = client.quote_to_amount(leg.symbol, size_quote)
                            if qty is None:
                                logger.warning(f"No se pudo calcular cantidad para {leg.symbol}")
                                continue
                            if not client.dry_run:
                                res = client.create_market_order(leg.symbol, side=leg.side, amount=qty)
                            else:
                                res = {"status": "simulated", "symbol": leg.symbol, "side": leg.side, "amount": qty}
                            orders.append(res)
                        if orders:
                            pos_manager.register_pair_position(signal, orders)
                            json_log("src/data/logs/trades.jsonl", {"ts": datetime.utcnow().isoformat(), "strategy": type(strat).__name__, "signal": signal.to_dict(), "orders": orders})
                    continue

                # Si no es multi‑leg y la acción es HOLD, saltamos
                if signal.action == "HOLD":
                    continue

                # Señal single-leg
                symbol = signal.symbol
                quote_bal = client.get_quote_balance("USDC")
                base_size = pos_manager.size_in_quote(quote_balance=quote_bal)
                # Ponderar por pesos
                w_map = {"MeanReversionStrategy": (weights.mean if weights else 0.5),
                         "MomentumStrategy": (weights.momentum if weights else 0.5)}
                w = w_map.get(type(strat).__name__, 0.33)
                size_quote = base_size * max(0.05, w)
                qty = client.quote_to_amount(symbol, size_quote)
                if qty is None:
                    logger.warning(f"No se pudo calcular cantidad para {symbol}")
                    continue

                order_res = None
                if not client.dry_run:
                    order_res = client.create_market_order(symbol, side="buy" if signal.action == "BUY" else "sell", amount=qty)
                else:
                    order_res = {"status": "simulated", "symbol": symbol, "side": signal.action.lower(), "amount": qty}

                position = Position(symbol=symbol, side=signal.action, qty=qty, entry_price=client.fetch_last_price(symbol), stop_loss=signal.stop_loss, take_profit=signal.take_profit, strategy=type(strat).__name__)
                pos_manager.register_position(position)

                json_log("src/data/logs/trades.jsonl", {"ts": datetime.utcnow().isoformat(), "strategy": type(strat).__name__, "signal": signal.to_dict(), "order": order_res})

            # Revisar stops/takes de posiciones abiertas
            for p in list(pos_manager.open_positions.values()):
                try:
                    last = client.fetch_last_price(p.symbol)
                except Exception as _e:
                    # Evita romper el loop por timeouts puntuales; reintenta en la siguiente iteración
                    continue
                exit_side = None
                if p.stop_loss and ((p.is_long and last <= p.stop_loss) or (not p.is_long and last >= p.stop_loss)):
                    exit_side = "sell" if p.is_long else "buy"
                    reason = "STOP_LOSS"
                elif p.take_profit and ((p.is_long and last >= p.take_profit) or (not p.is_long and last <= p.take_profit)):
                    exit_side = "sell" if p.is_long else "buy"
                    reason = "TAKE_PROFIT"
                if exit_side:
                    if not client.dry_run:
                        res = client.create_market_order(p.symbol, side=exit_side, amount=p.qty)
                    else:
                        res = {"status": "simulated", "symbol": p.symbol, "side": exit_side, "amount": p.qty}
                    # PnL realizado
                    pnl = (last - p.entry_price) * p.qty if p.is_long else (p.entry_price - last) * p.qty
                    if p.strategy:
                        perf_tracker.update(p.strategy, pnl)
                    pos_manager.close_position(p.symbol)
                    json_log("src/data/logs/trades.jsonl", {"ts": datetime.utcnow().isoformat(), "strategy": "RiskManager", "action": reason, "order": res, "pnl": pnl})

        except Exception as e:
            logger.exception(f"Error en loop principal: {e}")

        # Snapshot de posiciones abiertas para dashboard (JSON plano)
        try:
            import json as _json
            snapshot = {k: v.__dict__ for k, v in pos_manager.open_positions.items()}
            with open("src/data/logs/positions.json", "w", encoding="utf-8") as f:
                f.write(_json.dumps(snapshot))
        except Exception:
            pass

        # Guardar snapshot de asignador/régimen para dashboard (JSON plano)
        try:
            import json as _json
            snap = {
                "ts": datetime.utcnow().isoformat(),
                "regime": (regime_avg.__dict__ if regime_avg else {}),
                "weights": (weights.to_dict() if weights else {}),
            }
            with open("src/data/logs/allocator.json", "w", encoding="utf-8") as f:
                f.write(_json.dumps(snap))
        except Exception:
            pass

        # Ritmo de iteración
        elapsed = time.time() - loop_start
        sleep_s = max(1, args.interval - int(elapsed))
        time.sleep(sleep_s)


if __name__ == "__main__":
    # Permite ejecutar también: python -m src.main
    main()
