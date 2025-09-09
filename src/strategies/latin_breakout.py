from dataclasses import dataclass
from typing import Dict, Optional
from zoneinfo import ZoneInfo

import os
import pandas as pd
from joblib import load

from .base import BaseStrategy, TradeSignal
from src.utils.indicators import adx, ema


@dataclass
class LatinConfig:
    session_start_utc: str = "13:00"  # inicio de operativa (UTC)
    morning_end_utc: str = "13:00"    # fin de rango de la mañana (UTC)
    buffer_atr_mult: float = 0.25      # margen sobre el rango usando ATR
    atr_period: int = 14
    stop_atr_mult: float = 0.8
    take_atr_mult: float = 1.2
    use_model: bool = True
    proba_thresh: float = float(os.getenv("LATIN_PROBA_THRESH", 0.6))
    model_path: Optional[str] = os.getenv("LATIN_MODEL_PATH")


class LatinBreakoutStrategy(BaseStrategy):
    """Estrategia 'Latino' basada en ruptura del rango de la mañana con velas 1h (sesión diaria).

    - Calcula high/low desde 00:00 UTC hasta morning_end_utc (por defecto 13:00).
    - Tras la hora de inicio, si el cierre supera el high + buffer (en ATR), genera BUY.
    - Aplica stop/take en múltiplos de ATR.
    - Opcional: filtra por un modelo de probabilidad (joblib) si LATIN_MODEL_PATH está definido.
    """

    def __init__(self, symbol: str, config: LatinConfig = LatinConfig(), ml=None, ml_buy_thresh: float = 0.55, ml_sell_thresh: float = 0.45):
        super().__init__(ml=ml, ml_buy_thresh=ml_buy_thresh, ml_sell_thresh=ml_sell_thresh)
        self.symbol = symbol
        self.cfg = config
        self.model = None
        if self.cfg.use_model and self.cfg.model_path:
            try:
                self.model = load(self.cfg.model_path)
            except Exception:
                self.model = None

    def required_symbols(self):
        return [self.symbol]

    def _time_in_session(self, ts_utc: pd.Timestamp) -> bool:
        """Comprueba si ts_utc cae dentro de la sesión local si está configurada.
        Fallback: usa session_start_utc como antes.
        """
        tz = getattr(self.cfg, 'tz', 'UTC')
        start_local = getattr(self.cfg, 'session_start_local', None)
        end_local = getattr(self.cfg, 'session_end_local', None)
        if start_local:
            ts_local = ts_utc.tz_convert(ZoneInfo(tz))
            tstart = pd.to_datetime(start_local).time()
            ok = ts_local.time() >= tstart
            if end_local:
                tend = pd.to_datetime(end_local).time()
                ok = ok and (ts_local.time() <= tend)
            return ok
        tstart = pd.to_datetime(self.cfg.session_start_utc).time()
        return ts_utc.time() >= tstart

    def _morning_mask(self, df_day_utc: pd.DataFrame) -> pd.Series:
        tz = getattr(self.cfg, 'tz', 'UTC')
        ms = getattr(self.cfg, 'morning_start_local', None)
        me = getattr(self.cfg, 'morning_end_local', None)
        if ms and me:
            local_idx = df_day_utc.index.tz_convert(ZoneInfo(tz))
            tms = pd.to_datetime(ms).time()
            tme = pd.to_datetime(me).time()
            return (local_idx.time >= tms) & (local_idx.time < tme)
        morning_end = pd.to_datetime(self.cfg.morning_end_utc).time()
        return df_day_utc.index.time < morning_end

    def _atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        tr = pd.concat([
            (df['high'] - df['low']).abs(),
            (df['high'] - df['close'].shift()).abs(),
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _features_for_model(self, df: pd.DataFrame, morning_high: float, morning_low: float) -> Optional[pd.Series]:
        # Construye un pequeño vector de características alineado con el entrenamiento propuesto
        tmp = df.copy().iloc[-60:]
        tmp['ema50'] = ema(tmp['close'], 50)
        tmp['ema200'] = ema(tmp['close'], 200)
        tmp['ema_gap'] = (tmp['ema50'] - tmp['ema200']) / tmp['ema200']
        tmp['adx14'] = adx(tmp['high'], tmp['low'], tmp['close'], 14)
        atr = self._atr(tmp, self.cfg.atr_period)
        tmp['atrn'] = atr / tmp['close']
        # rango de la mañana relativo al precio actual
        last = tmp.iloc[-1]
        rng = (morning_high - morning_low) / max(1e-9, float(last['close']))
        tmp['range_morning'] = rng
        out = tmp.dropna().iloc[-1]
        return out

    def check_signal(self, market_data: Dict[str, pd.DataFrame]) -> TradeSignal:
        df = market_data[self.symbol].copy()
        if df.empty or len(df) < 60:
            return TradeSignal(action="HOLD", symbol=self.symbol, reason="insuf_hist")
        # Asegurar índice UTC
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')

        last_ts = df.index[-1]
        if not self._time_in_session(last_ts):
            return TradeSignal(action="HOLD", symbol=self.symbol, reason="pre_session")

        # Calcular high/low de la mañana del mismo día UTC
        tz_name = getattr(self.cfg, 'tz', None)
        if tz_name:
            local_dates = df.index.tz_convert(ZoneInfo(tz_name)).date
            day_mask = local_dates == last_ts.tz_convert(ZoneInfo(tz_name)).date()
        else:
            day_mask = df.index.date == last_ts.date()
        df_day = df.loc[day_mask]
        if df_day.empty:
            return TradeSignal(action="HOLD", symbol=self.symbol, reason="no_day_data")
        morning_mask = self._morning_mask(df_day)
        df_morning = df_day.loc[morning_mask]
        if df_morning.empty:
            return TradeSignal(action="HOLD", symbol=self.symbol, reason="no_morning")
        morning_high = float(df_morning['high'].max())
        morning_low = float(df_morning['low'].min())

        # ATR y buffer
        atr = float(self._atr(df_day, self.cfg.atr_period).iloc[-1])
        buffer = self.cfg.buffer_atr_mult * atr if atr > 0 else 0.0
        price = float(df_day['close'].iloc[-1])

        # Filtro de tendencia básico con 1h
        ema50 = float(ema(df_day['close'], 50).iloc[-1])
        ema200 = float(ema(df_day['close'], 200).iloc[-1])
        adx_v = float(adx(df_day['high'], df_day['low'], df_day['close'], 14).iloc[-1])
        trend_ok = (ema50 > ema200) and (adx_v >= 18)

        # Ruptura larga
        if price > (morning_high + buffer) and trend_ok:
            # Modelo opcional
            if self.model is not None:
                feats = self._features_for_model(df_day, morning_high, morning_low)
                if feats is not None:
                    model_feats = self.model.get('features', [])
                    proba = self.model['model'].predict_proba(feats[model_feats].values.reshape(1, -1))[0][1]
                    thr = self.model.get('meta', {}).get('threshold', self.cfg.proba_thresh)
                    if proba < thr:
                        return TradeSignal(action="HOLD", symbol=self.symbol, reason="low_proba")

            stop = price - self.cfg.stop_atr_mult * atr if atr > 0 else morning_high
            take = price + self.cfg.take_atr_mult * atr if atr > 0 else price * 1.01
            return TradeSignal(action="BUY", symbol=self.symbol, reason="latin_breakout", stop_loss=stop, take_profit=take)

        return TradeSignal(action="HOLD", symbol=self.symbol, reason="no_breakout")
