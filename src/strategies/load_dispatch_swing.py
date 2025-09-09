from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
from src.models.load_quantile_forecaster import LoadQuantileForecaster
from src.utils.load_reconcile import reconcile_load_quantiles
from src.utils.indicators import compute_rsi, compute_bbands, compute_adx
from src.utils.risk import dispatch_size_from_quantiles


@dataclass
class LoadDispatchConfig:
    q_model_paths: Dict[str, str]  # {'1d': '...pkl', '4h':'...pkl', '1h':'...pkl'}
    thr_up: float = 0.0005
    thr_down: float = -0.0005
    max_pos: float = 1.0
    min_unc: float = 1e-5


class LoadDispatchSwingStrategy:
    """Señal BUY/HOLD/SELL a partir de quantiles reconciliados y filtros técnicos (RSI/BB/ADX)."""

    def __init__(self, cfg: LoadDispatchConfig):
        self.cfg = cfg
        self.models = {tf: LoadQuantileForecaster.load(p) for tf, p in cfg.q_model_paths.items()}

    def infer(self, feats_by_tf: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        mtf_q: Dict[str, tuple] = {}
        for tf, model in self.models.items():
            X = feats_by_tf[tf].iloc[[-1]][model.spec.features]
            mtf_q[tf] = tuple(model.predict_quantiles(X)[0])

        q10, q50, q90 = reconcile_load_quantiles(mtf_q)
        unc = max(q90 - q10, self.cfg.min_unc)

        main_tf = '1d' if '1d' in feats_by_tf else '4h' if '4h' in feats_by_tf else next(iter(feats_by_tf))
        df = feats_by_tf[main_tf]
        rsi = float(compute_rsi(df).iloc[-1])
        bb = compute_bbands(df).iloc[-1]
        bb_width = float(bb['bb_width'])
        adx = float(compute_adx(df).iloc[-1])

        buy = (q50 > self.cfg.thr_up) and (unc < bb_width) and (rsi < 70) and (adx >= 20)
        sell = (q50 < self.cfg.thr_down) and (unc < bb_width) and (rsi > 30) and (adx >= 20)

        action, size = "HOLD", 0.0
        if buy:
            action = "BUY"
            size = dispatch_size_from_quantiles(q10, q50, q90, self.cfg.max_pos)
        elif sell:
            action = "SELL"
            size = -dispatch_size_from_quantiles(q10, -abs(q50), q90, self.cfg.max_pos)

        return {
            "action": action, "size": float(size),
            "quantiles": (float(q10), float(q50), float(q90)),
            "uncertainty": float(unc),
            "rsi": rsi, "adx": adx, "bb_width": float(bb_width),
            "mtf": mtf_q
        }

