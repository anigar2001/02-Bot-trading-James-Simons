from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump, load

Quantiles = Tuple[float, float, float]


@dataclass
class LoadModelSpec:
    symbol: str
    timeframe: str
    horizon: str        # '15m'|'1h'|'1d'
    features: List[str]
    target: str         # 'ret_t+h'


class LoadQuantileForecaster:
    """
    3 GradientBoostingRegressor (loss='quantile') para q10/q50/q90.
    Target: retorno log a horizonte (t->t+h).
    """

    def __init__(self, spec: LoadModelSpec):
        self.spec = spec
        self.models: Dict[str, GradientBoostingRegressor] = {}

    @staticmethod
    def _model(alpha: float) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            loss="quantile", alpha=alpha,
            n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.9, random_state=42
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LoadQuantileForecaster":
        feats = self.spec.features
        self.models['q10'] = self._model(0.10).fit(X[feats], y)
        self.models['q50'] = self._model(0.50).fit(X[feats], y)
        self.models['q90'] = self._model(0.90).fit(X[feats], y)
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> List[Quantiles]:
        feats = self.spec.features
        q10 = self.models['q10'].predict(X[feats])
        q50 = self.models['q50'].predict(X[feats])
        q90 = self.models['q90'].predict(X[feats])
        return [(float(min(a, c)), float(b), float(max(a, c))) for a, b, c in zip(q10, q50, q90)]

    def save(self, path: str) -> None:
        dump({"spec": self.spec, "models": self.models}, path)

    @staticmethod
    def load(path: str) -> "LoadQuantileForecaster":
        obj = load(path)
        qf = LoadQuantileForecaster(obj["spec"])
        qf.models = obj["models"]
        return qf

