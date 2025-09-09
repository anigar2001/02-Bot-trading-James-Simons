from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .base import BaseStrategy, TradeSignal, TradeLeg


@dataclass
class PairsConfig:
    window: int = 50
    entry_z: float = 2.0
    exit_z: float = 0.5


class PairsArbitrageStrategy(BaseStrategy):
    """Arbitraje estadístico entre BTC/USDT y LTC/USDT mediante ratio y z-score.

    ratio = close_B / close_A (por defecto B=LTC, A=BTC)
    - Si z > entry_z: vender B y comprar A
    - Si z < -entry_z: comprar B y vender A
    - Salida cuando |z| < exit_z (no implementa cierre automático aquí)
    """

    def __init__(self, symbol_a: str = "BTC/USDC", symbol_b: str = "LTC/USDC", config: PairsConfig = PairsConfig()):
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.cfg = config

    def required_symbols(self):
        return [self.symbol_a, self.symbol_b]

    def check_signal(self, market_data: Dict[str, pd.DataFrame]) -> TradeSignal:
        a = market_data[self.symbol_a].copy()
        b = market_data[self.symbol_b].copy()
        # Alineamos por timestamp (inner join)
        df = pd.concat([
            a["close"].rename("a_close"),
            b["close"].rename("b_close")
        ], axis=1).dropna()
        if len(df) < self.cfg.window + 5:
            return TradeSignal(action="HOLD", reason="insuficiente_hist")

        df["ratio"] = df["b_close"] / df["a_close"]
        mu = df["ratio"].rolling(self.cfg.window).mean()
        sigma = df["ratio"].rolling(self.cfg.window).std()
        z = (df["ratio"] - mu) / sigma
        z_last = float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else 0.0

        if z_last > self.cfg.entry_z:
            # B caro vs A: vender B (LTC), comprar A (BTC)
            return TradeSignal(action="HOLD", multi_leg=True, legs=[
                TradeLeg(symbol=self.symbol_b, side="sell"),
                TradeLeg(symbol=self.symbol_a, side="buy"),
            ], reason=f"z>{self.cfg.entry_z}")

        if z_last < -self.cfg.entry_z:
            # B barato vs A: comprar B, vender A
            return TradeSignal(action="HOLD", multi_leg=True, legs=[
                TradeLeg(symbol=self.symbol_b, side="buy"),
                TradeLeg(symbol=self.symbol_a, side="sell"),
            ], reason=f"z<-{self.cfg.entry_z}")

        return TradeSignal(action="HOLD", reason="no_spread")
