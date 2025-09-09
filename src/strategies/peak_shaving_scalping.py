from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
from src.utils.grid_entropy import shannon_entropy
from src.utils.indicators import compute_adx


@dataclass
class PeakShavingConfig:
    ent_window: int = 120   # 120 * 1m = 2h
    ent_bins: int = 10
    ent_max: float = 1.8    # si entropía > ent_max => no trade
    adx_min: int = 18       # fuerza mínima de tendencia
    z_entry: float = 0.5    # umbral sobre retorno esperado (proxy simple)


class PeakShavingScalpingStrategy:
    """Scalping 1m con filtro de 'ruido de red' (entropía) + confirmación ADX."""

    def __init__(self, cfg: PeakShavingConfig):
        self.cfg = cfg

    def infer(self, df_1m: pd.DataFrame) -> Dict[str, Any]:
        rets = np.log(df_1m["close"]).diff().dropna().to_numpy()[-self.cfg.ent_window:]
        ent = shannon_entropy(rets, bins=self.cfg.ent_bins)
        adx = float(compute_adx(df_1m).iloc[-1])

        no_trade = (ent > self.cfg.ent_max) or (adx < self.cfg.adx_min)
        exp_ret = float(pd.Series(rets).ewm(span=20, adjust=False).mean().iloc[-1])  # proxy simple
        action = "HOLD"
        if not no_trade:
            if exp_ret > self.cfg.z_entry:
                action = "BUY"
            elif exp_ret < -self.cfg.z_entry:
                action = "SELL"

        return {"action": action, "entropy": float(ent), "adx": adx, "exp_ret_proxy": exp_ret}

