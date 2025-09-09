from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TradeLeg:
    symbol: str
    side: str  # 'buy' o 'sell'


@dataclass
class TradeSignal:
    action: str  # 'BUY' | 'SELL' | 'EXIT' | 'HOLD'
    symbol: Optional[str] = None
    reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    multi_leg: bool = False
    legs: Optional[List[TradeLeg]] = None

    def to_dict(self):
        return {
            "action": self.action,
            "symbol": self.symbol,
            "reason": self.reason,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "multi_leg": self.multi_leg,
            "legs": [l.__dict__ for l in (self.legs or [])],
        }


class BaseStrategy:
    def __init__(self, ml=None, ml_buy_thresh: float = 0.55, ml_sell_thresh: float = 0.45):
        # ml: objeto con método predict_proba_up(df)->float | None
        self.ml = ml
        self.ml_buy_thresh = ml_buy_thresh
        self.ml_sell_thresh = ml_sell_thresh

    def name(self) -> str:
        return type(self).__name__

    def required_symbols(self):
        raise NotImplementedError

    def check_signal(self, market_data):
        raise NotImplementedError

    def _ml_filter(self, df, action: str) -> bool:
        """Devuelve True si la señal pasa el filtro ML (o si no hay ML)."""
        if self.ml is None:
            return True
        try:
            proba = self.ml.predict_proba_up(df)
            if proba is None:
                return True
            if action == 'BUY':
                return proba >= self.ml_buy_thresh
            if action == 'SELL':
                return proba <= self.ml_sell_thresh
            return True
        except Exception:
            return True
