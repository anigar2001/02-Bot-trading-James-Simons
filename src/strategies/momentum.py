from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .base import BaseStrategy, TradeSignal
from src.utils.indicators import ema, adx


@dataclass
class MomentumConfig:
    fast: int = 50
    slow: int = 200
    adx_min: int = 20
    stop_pct: float = 0.02
    trail: bool = False  # trailing stop opcional (no implementado aquí)


class MomentumStrategy(BaseStrategy):
    """Estrategia de momentum por cruces de medias y filtro ADX."""

    def __init__(self, symbol: str, config: MomentumConfig = MomentumConfig(), ml=None, ml_buy_thresh: float = 0.55, ml_sell_thresh: float = 0.45):
        super().__init__(ml=ml, ml_buy_thresh=ml_buy_thresh, ml_sell_thresh=ml_sell_thresh)
        self.symbol = symbol
        self.cfg = config

    def required_symbols(self):
        return [self.symbol]

    def check_signal(self, market_data: Dict[str, pd.DataFrame]) -> TradeSignal:
        df = market_data[self.symbol].copy()
        if len(df) < max(self.cfg.fast, self.cfg.slow) + 5:
            return TradeSignal(action="HOLD", symbol=self.symbol, reason="insuficiente_hist")

        df["ema_fast"] = ema(df["close"], self.cfg.fast)
        df["ema_slow"] = ema(df["close"], self.cfg.slow)
        df["adx"] = adx(df["high"], df["low"], df["close"], 14)
        last = df.iloc[-1]

        price = float(last["close"]) if pd.notna(last["close"]) else None
        if price is None:
            return TradeSignal(action="HOLD", symbol=self.symbol, reason="nan")

        bull = last["ema_fast"] > last["ema_slow"]
        bear = last["ema_fast"] < last["ema_slow"]
        strong = last["adx"] >= self.cfg.adx_min

        if bull and strong and self._ml_filter(df, 'BUY'):
            stop = price * (1 - self.cfg.stop_pct)
            take = price * (1 + self.cfg.stop_pct * 1.5)
            return TradeSignal(action="BUY", symbol=self.symbol, reason="bull_trend", stop_loss=stop, take_profit=take)
        if bear and strong and self._ml_filter(df, 'SELL'):
            stop = price * (1 + self.cfg.stop_pct)
            take = price * (1 - self.cfg.stop_pct * 1.5)
            return TradeSignal(action="SELL", symbol=self.symbol, reason="bear_trend", stop_loss=stop, take_profit=take)

        return TradeSignal(action="HOLD", symbol=self.symbol, reason="no_trend")
