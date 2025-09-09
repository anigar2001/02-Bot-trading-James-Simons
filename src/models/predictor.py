from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import load

from src.utils.indicators import ema, rsi, adx


class MLSignal:
    def __init__(self, path: str = None):
        import os
        default_path = os.getenv('MODEL_PATH', 'src/models/trained_model.pkl')
        self.path = Path(path or default_path)
        self.bundle = None
        if self.path.exists():
            self.bundle = load(self.path)

    def ready(self) -> bool:
        return self.bundle is not None

    def predict_proba_up(self, df: pd.DataFrame) -> Optional[float]:
        if not self.ready() or len(df) < 210:
            return None
        feature_cols = self.bundle['features']
        row = self._build_row(df)[feature_cols].values.reshape(1, -1)
        proba = self.bundle['model'].predict_proba(row)[0][1]
        return float(proba)

    def _build_row(self, df: pd.DataFrame) -> pd.Series:
        tmp = df.copy().iloc[-300:]
        tmp['ret1'] = tmp['close'].pct_change()
        tmp['ema20'] = ema(tmp['close'], 20)
        tmp['ema50'] = ema(tmp['close'], 50)
        tmp['ema_gap'] = (tmp['ema20'] - tmp['ema50']) / tmp['ema50']
        tmp['rsi14'] = rsi(tmp['close'], 14)
        tmp['adx14'] = adx(tmp['high'], tmp['low'], tmp['close'], 14)
        tmp['vol'] = tmp['close'].pct_change().rolling(20).std()
        # Nuevas features
        tmp['roc3'] = tmp['close'].pct_change(3)
        tmp['roc10'] = tmp['close'].pct_change(10)
        tmp['ema20_slope'] = tmp['ema20'].pct_change()
        ma20 = tmp['close'].rolling(20).mean()
        sd20 = tmp['close'].rolling(20).std()
        tmp['bbp20_2'] = (tmp['close'] - ma20) / (2 * sd20)
        tr = pd.concat([
            (tmp['high'] - tmp['low']).abs(),
            (tmp['high'] - tmp['close'].shift()).abs(),
            (tmp['low'] - tmp['close'].shift()).abs()
        ], axis=1).max(axis=1)
        tmp['atr14n'] = tr.rolling(14).mean() / tmp['close']
        vmean = tmp['volume'].rolling(20).mean()
        vstd = tmp['volume'].rolling(20).std()
        tmp['v_z20'] = (tmp['volume'] - vmean) / vstd
        tmp['vel3'] = tmp['close'].pct_change(3)
        tmp['vol_vel'] = tmp['v_z20'] * tmp['vel3']
        return tmp.dropna().iloc[-1]
