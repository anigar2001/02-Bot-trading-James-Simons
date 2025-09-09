"""Entrenamiento de un RandomForest para clasificar dirección próxima vela.

Uso:
    python -m src.models.model_training --symbol BTC/USDT --timeframe 1h --limit 2000
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from joblib import dump

from src.utils.api import BinanceClient
from src.utils.indicators import ema, rsi, adx
import os
import math
from datetime import datetime, timedelta


def load_data(symbol: str, timeframe: str, limit: int, client: BinanceClient, offline: bool = False) -> pd.DataFrame:
    hist_path = Path(f"src/data/historical/{symbol.replace('/', '_')}_{timeframe}.csv")
    if hist_path.exists():
        df = pd.read_csv(hist_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        return df
    # Descargar si no hay CSV y no estamos offline
    if not offline:
        try:
            df = client.fetch_ohlcv_df(symbol, timeframe=timeframe, limit=limit)
            df.reset_index().rename(columns={'index': 'timestamp'}).to_csv(hist_path, index=False)
            return df
        except Exception:
            pass
    # Generar datos sintéticos si no hay red o falla descarga
    df = generate_synthetic_ohlcv(timeframe=timeframe, limit=limit)
    out = df.reset_index().rename(columns={'index': 'timestamp'})
    Path('src/data/historical').mkdir(parents=True, exist_ok=True)
    out.to_csv(hist_path, index=False)
    return df


def timeframe_to_minutes(tf: str) -> int:
    mapping = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
    return mapping.get(tf, 60)


def generate_synthetic_ohlcv(timeframe: str = "1h", limit: int = 2000, start_price: float = 20000.0) -> pd.DataFrame:
    np.random.seed(42)
    minutes = timeframe_to_minutes(timeframe)
    start = datetime.utcnow() - timedelta(minutes=minutes * limit)
    prices = [start_price]
    for _ in range(limit - 1):
        drift = 0.0001
        vol = 0.005
        ret = np.random.normal(drift, vol)
        prices.append(prices[-1] * (1 + ret))
    closes = np.array(prices)
    highs = closes * (1 + np.random.uniform(0, 0.003, size=limit))
    lows = closes * (1 - np.random.uniform(0, 0.003, size=limit))
    opens = closes * (1 + np.random.uniform(-0.002, 0.002, size=limit))
    vols = np.random.uniform(1, 10, size=limit)
    idx = pd.date_range(start=start, periods=limit, freq=f"{minutes}min")
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': vols,
    }, index=idx)
    return df


def build_features(df: pd.DataFrame, horizon: int = 1, include_base: bool = False) -> pd.DataFrame:
    """Construye features técnicos y etiqueta binaria.

    horizon: número de velas hacia delante para la etiqueta (>=1)
    include_base: si True, incluye columnas base como close/volume para análisis externo
    """
    out = df.copy()
    # Features base
    out['ret1'] = out['close'].pct_change()
    out['ema20'] = ema(out['close'], 20)
    out['ema50'] = ema(out['close'], 50)
    out['ema200'] = ema(out['close'], 200)
    out['rsi14'] = rsi(out['close'], 14)
    out['adx14'] = adx(out['high'], out['low'], out['close'], 14)
    out['ema_gap'] = (out['ema20'] - out['ema50']) / out['ema50']
    out['vol'] = out['close'].pct_change().rolling(20).std()
    # Nuevas features
    out['roc3'] = out['close'].pct_change(3)
    out['roc10'] = out['close'].pct_change(10)
    out['ema20_slope'] = out['ema20'].pct_change()
    # Bollinger %B (20, 2)
    ma20 = out['close'].rolling(20).mean()
    sd20 = out['close'].rolling(20).std()
    out['bbp20_2'] = (out['close'] - ma20) / (2 * sd20)
    # ATR14 normalizado
    tr = pd.concat([
        (out['high'] - out['low']).abs(),
        (out['high'] - out['close'].shift()).abs(),
        (out['low'] - out['close'].shift()).abs()
    ], axis=1).max(axis=1)
    out['atr14n'] = tr.rolling(14).mean() / out['close']
    # Volumen z-score (20)
    vmean = out['volume'].rolling(20).mean()
    vstd = out['volume'].rolling(20).std()
    out['v_z20'] = (out['volume'] - vmean) / vstd
    # Velocidad (3 velas) y mezcla con volumen
    out['vel3'] = out['close'].pct_change(3)
    out['vol_vel'] = out['v_z20'] * out['vel3']

    # Etiqueta con horizonte configurable
    h = max(1, int(horizon))
    out['target'] = (out['close'].shift(-h) > out['close']).astype(int)

    if not include_base:
        # eliminar columnas temporales para evitar fuga
        out = out.drop(columns=['ema20', 'ema50', 'ema200'])
    out = out.dropna()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default='BTC/USDT')
    parser.add_argument('--timeframe', default='1h')
    parser.add_argument('--limit', type=int, default=2000)
    parser.add_argument('--offline', action='store_true', help='No descarga. Genera datos sintéticos si no hay CSV')
    args = parser.parse_args()

    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    api_base = os.getenv('API_BASE', 'https://testnet.binance.vision')
    client = BinanceClient(api_key=api_key, api_secret=api_secret, api_base=api_base, enable_rate_limit=True, testnet=True, dry_run=True)
    client.load_markets()

    offline = args.offline or (os.getenv('OFFLINE_SYNTHETIC', '1') == '1')
    df = load_data(args.symbol, args.timeframe, args.limit, client, offline=offline)
    feats = build_features(df)
    feature_cols = ['ret1', 'ema_gap', 'rsi14', 'adx14', 'vol', 'roc3', 'roc10', 'ema20_slope', 'bbp20_2', 'atr14n', 'v_z20', 'vel3', 'vol_vel']
    X = feats[feature_cols].values
    y = feats['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=200, random_state=7)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3))

    Path('src/models').mkdir(parents=True, exist_ok=True)
    dump({'model': model, 'features': feature_cols}, 'src/models/trained_model.pkl')
    print('Modelo guardado en src/models/trained_model.pkl')


if __name__ == '__main__':
    main()
