from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.indicators import adx, ema
import os


@dataclass
class Regime:
    trend_strength: float  # ADX en timeframe base
    vol: float             # Desviación estándar de retornos (base)
    near_sr: bool          # Cerca de soporte/resistencia (tf superior)
    pairs_z: Optional[float] = None
    sr_state: str = "none"  # 'support' | 'resistance' | 'none'


class PerformanceTracker:
    """Acumula PnL reciente por estrategia y lo guarda en disco para persistencia."""

    def __init__(self, path: str = "src/data/logs/perf.json", window: int = 50):
        self.path = Path(path)
        self.window = window
        self.scores: Dict[str, List[float]] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.scores = json.loads(self.path.read_text(encoding='utf-8'))
            except Exception:
                self.scores = {}

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.write_text(json.dumps(self.scores), encoding='utf-8')
        except Exception:
            pass

    def update(self, strategy: str, pnl: float):
        arr = self.scores.get(strategy, [])
        arr.append(float(pnl))
        if len(arr) > self.window:
            arr = arr[-self.window:]
        self.scores[strategy] = arr
        self._save()

    def score(self, strategy: str) -> float:
        arr = self.scores.get(strategy, [])
        if not arr:
            return 0.0
        # Score sencillo: media truncada
        vals = np.array(arr)
        # Limitar outliers
        clip = np.clip(vals, np.percentile(vals, 5), np.percentile(vals, 95))
        return float(np.mean(clip))


def compute_sr_state(df_high_tf: pd.DataFrame, last_price: float, win: int = 50, proximity_pct: float = 0.005) -> str:
    """Detecta si el precio está cerca de un soporte o resistencia simple de ventana.

    Soporte: mínimo de las últimas `win` velas; Resistencia: máximo. Devuelve 'support',
    'resistance' o 'none' si no está cerca.
    """
    if len(df_high_tf) < win + 5:
        return "none"
    sub = df_high_tf.iloc[-win:]
    sup = float(sub['low'].min())
    res = float(sub['high'].max())
    near_sup = abs((last_price - sup) / last_price) <= proximity_pct
    near_res = abs((res - last_price) / last_price) <= proximity_pct
    if near_sup:
        return "support"
    if near_res:
        return "resistance"
    return "none"


def compute_pairs_z(df_a: pd.DataFrame, df_b: pd.DataFrame, window: int = 50) -> Optional[float]:
    try:
        df = pd.concat([
            df_a['close'].rename('a'),
            df_b['close'].rename('b')
        ], axis=1).dropna()
        if len(df) < window + 5:
            return None
        ratio = df['b'] / df['a']
        mu = ratio.rolling(window).mean()
        sd = ratio.rolling(window).std()
        z = (ratio - mu) / sd
        return float(z.iloc[-1]) if pd.notna(z.iloc[-1]) else None
    except Exception:
        return None


def detect_regime(df_base: pd.DataFrame, df_high_tf: pd.DataFrame, last_price: float, pairs_z: Optional[float], sr_window: int = 50, sr_proximity_pct: float = 0.005) -> Regime:
    # Fuerza de tendencia
    adx_v = float(adx(df_base['high'], df_base['low'], df_base['close'], 14).iloc[-1]) if len(df_base) > 30 else 0.0
    # Volatilidad reciente
    vol = float(df_base['close'].pct_change().rolling(30).std().iloc[-1]) if len(df_base) > 35 else 0.0
    # Proximidad a S/R en temporalidad superior
    sr_state = compute_sr_state(df_high_tf, last_price, win=sr_window, proximity_pct=sr_proximity_pct)
    near_sr = (sr_state != "none")
    return Regime(trend_strength=adx_v, vol=vol, near_sr=near_sr, pairs_z=pairs_z, sr_state=sr_state)


@dataclass
class Weights:
    mean: float
    momentum: float
    pairs: float

    def to_dict(self):
        s = self.mean + self.momentum + self.pairs
        return {"mean": self.mean, "momentum": self.momentum, "pairs": self.pairs, "sum": s}


class Allocator:
    """Asigna pesos a estrategias según régimen y rendimiento reciente (heurístico).

    - trending fuerte y lejos de S/R -> más Momentum
    - rango o cerca de S/R -> más Mean Reversion
    - si |pairs_z|>2 -> sube peso de Pares
    - ajusta por rendimiento reciente en +/-20%
    """

    def __init__(self, perf_tracker: PerformanceTracker):
        self.perf = perf_tracker
        # Parámetros desde entorno
        self.trend_adx_threshold = float(os.getenv('ALLOC_ADX_TREND', 25))
        self.sr_window = int(os.getenv('ALLOC_SR_WINDOW', 50))
        self.sr_prox_pct = float(os.getenv('ALLOC_SR_PROX_PCT', 0.005))
        self.pairs_z_entry = float(os.getenv('ALLOC_PAIRS_Z_ENTRY', 2.0))

    def base_weights(self, regime: Regime) -> Weights:
        trending = regime.trend_strength >= self.trend_adx_threshold
        if trending and not regime.near_sr:
            w = Weights(mean=0.3, momentum=0.6, pairs=0.1)
        else:
            w = Weights(mean=0.6, momentum=0.3, pairs=0.1)
        if regime.pairs_z is not None and abs(regime.pairs_z) >= self.pairs_z_entry:
            # Aumentar pares y normalizar
            add = 0.3
            w.pairs += add
            # Quitar proporcionalmente de las otras
            total_other = w.mean + w.momentum
            if total_other > 0:
                factor = (total_other - add) / total_other
                w.mean *= factor
                w.momentum *= factor
        return w

    def adjust_by_performance(self, w: Weights) -> Weights:
        # Multiplicadores por rendimiento en [0.8, 1.2]
        def mult(s: str) -> float:
            sc = self.perf.score(s)
            # Escala suave (función sigmoide aproximada)
            # asume PnL medio ~0; positivo -> >1; negativo -> <1
            return float(np.clip(1.0 + 0.4 * np.tanh(sc / 10.0), 0.8, 1.2))

        m_mean = mult('MeanReversionStrategy')
        m_mom = mult('MomentumStrategy')
        m_pairs = mult('PairsArbitrageStrategy')
        w2 = Weights(mean=w.mean * m_mean, momentum=w.momentum * m_mom, pairs=w.pairs * m_pairs)
        # Normaliza a suma<=1
        s = w2.mean + w2.momentum + w2.pairs
        if s > 1e-9:
            w2.mean /= s
            w2.momentum /= s
            w2.pairs /= s
        return w2

    def weights(self, regime: Regime) -> Weights:
        base = self.base_weights(regime)
        return self.adjust_by_performance(base)
