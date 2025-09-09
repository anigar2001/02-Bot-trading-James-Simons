from typing import Optional
import numpy as np
import pandas as pd

# Fallback inteligente: usa TA-Lib si está disponible; si no, pandas_ta; si no, pandas puro
try:
    import talib as ta  # type: ignore
    HAS_TALIB = True
except Exception:
    HAS_TALIB = False
    try:
        import pandas_ta as pta  # type: ignore
        HAS_PANDAS_TA = True
    except Exception:
        HAS_PANDAS_TA = False


def ema(series: pd.Series, period: int) -> pd.Series:
    if HAS_TALIB:
        return pd.Series(ta.EMA(series.values.astype(float), timeperiod=period), index=series.index)
    if HAS_PANDAS_TA:
        return series.ta.ema(length=period)
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    if HAS_TALIB:
        return pd.Series(ta.SMA(series.values.astype(float), timeperiod=period), index=series.index)
    if HAS_PANDAS_TA:
        return series.ta.sma(length=period)
    return series.rolling(window=period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if HAS_TALIB:
        return pd.Series(ta.RSI(series.values.astype(float), timeperiod=period), index=series.index)
    if HAS_PANDAS_TA:
        return series.ta.rsi(length=period)
    # Implementación básica sin TA-Lib
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    if HAS_TALIB:
        return pd.Series(ta.ADX(high.values, low.values, close.values, timeperiod=period), index=close.index)
    if HAS_PANDAS_TA:
        return pta.adx(high=high, low=low, close=close, length=period)["ADX_14"]
    # ADX muy simplificado si no hay librerías (no ideal para prod)
    tr = (pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)).rolling(period).sum()
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=high.index).rolling(period).sum() / tr
    minus_di = 100 * pd.Series(minus_dm, index=high.index).rolling(period).sum() / tr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
    return dx.rolling(period).mean().fillna(0)


# Helpers extendidos (API estable)
def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return rsi(df['close'], period)


def compute_bbands(df: pd.DataFrame, period: int = 20, ndev: float = 2.0) -> pd.DataFrame:
    close = df['close']
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = ma + ndev * sd
    lower = ma - ndev * sd
    width = (upper - lower) / (ma.replace(0, np.nan))
    out = pd.DataFrame({
        'bb_lower': lower,
        'bb_mid': ma,
        'bb_upper': upper,
        'bb_width': width
    }, index=df.index)
    return out


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return adx(df['high'], df['low'], df['close'], period)


def add_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Añade indicadores básicos para modelos quantile.
    Columnas: rsi_14, bb_width_20, adx_14, ret_1_ema_20, vol_ema_20, sma_20, sma_50, sma_ratio_20_50
    """
    out = df.copy()
    out['ret_1'] = np.log(out['close']).diff()
    out['rsi_14'] = compute_rsi(out, 14)
    bb = compute_bbands(out, 20, 2.0)
    out['bb_width_20'] = bb['bb_width']
    out['adx_14'] = compute_adx(out, 14)
    out['ret_1_ema_20'] = out['ret_1'].ewm(span=20, adjust=False).mean()
    out['vol_ema_20'] = out['ret_1'].rolling(20).std().ewm(span=10, adjust=False).mean()
    out['sma_20'] = sma(out['close'], 20)
    out['sma_50'] = sma(out['close'], 50)
    out['sma_ratio_20_50'] = out['sma_20'] / out['sma_50'] - 1.0
    return out
