from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .base import BaseStrategy, TradeSignal
from src.utils.indicators import ema, rsi


@dataclass
class MRConfig:
    period: int = 20
    threshold: float = 0.02  # 2%
    use_rsi: bool = True
    rsi_buy: int = 30
    rsi_sell: int = 70
    stop_mult: float = 1.5
    take_mult: float = 0.8


class MeanReversionStrategy(BaseStrategy):
    """Reversión a la media con EMA y filtro RSI opcional."""

    def __init__(self, symbol: str, config: MRConfig = MRConfig(), ml=None, ml_buy_thresh: float = 0.55, ml_sell_thresh: float = 0.45):
        super().__init__(ml=ml, ml_buy_thresh=ml_buy_thresh, ml_sell_thresh=ml_sell_thresh)
        self.symbol = symbol
        self.cfg = config

    def required_symbols(self):
        return [self.symbol]

    def check_signal(self, market_data: Dict[str, pd.DataFrame]) -> TradeSignal:
        df = market_data[self.symbol].copy()
        if len(df) < self.cfg.period + 5:
            return TradeSignal(action="HOLD", symbol=self.symbol, reason="insuficiente_hist")

        df["ema"] = ema(df["close"], self.cfg.period)
        df["rsi"] = rsi(df["close"], 14)
        last = df.iloc[-1]
        ema_v = float(last["ema"]) if pd.notna(last["ema"]) else None
        price = float(last["close"]) if pd.notna(last["close"]) else None
        if not ema_v or not price:
            return TradeSignal(action="HOLD", symbol=self.symbol, reason="nan")

        upper = ema_v * (1 + self.cfg.threshold)
        lower = ema_v * (1 - self.cfg.threshold)

        # Condiciones
        if price > upper:
            if ((not self.cfg.use_rsi) or (last["rsi"] >= self.cfg.rsi_sell)) and self._ml_filter(df, 'SELL'):
                stop = price * (1 + self.cfg.threshold * self.cfg.stop_mult)
                take = ema_v  # objetivo conservador: vuelta a la media
                return TradeSignal(action="SELL", symbol=self.symbol, reason="price_above_upper", stop_loss=stop, take_profit=take)

        if price < lower:
            if ((not self.cfg.use_rsi) or (last["rsi"] <= self.cfg.rsi_buy)) and self._ml_filter(df, 'BUY'):
                stop = price * (1 - self.cfg.threshold * self.cfg.stop_mult)
                take = ema_v
                return TradeSignal(action="BUY", symbol=self.symbol, reason="price_below_lower", stop_loss=stop, take_profit=take)

        return TradeSignal(action="HOLD", symbol=self.symbol, reason="no_edge")
