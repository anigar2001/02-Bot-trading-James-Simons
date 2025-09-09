from __future__ import annotations
import argparse, pathlib
import pandas as pd, numpy as np
from typing import List
from src.models.load_quantile_forecaster import LoadQuantileForecaster, LoadModelSpec
from src.utils.indicators import add_basic_indicators


def load_partition(symbol: str, timeframe: str) -> pd.DataFrame:
    base = pathlib.Path("src/data/historical/partitioned") / symbol / timeframe
    files = sorted(base.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No data for {symbol} {timeframe}. Run backfill.")
    df_list = []
    for f in files:
        df = pd.read_parquet(f)
        idx = pd.to_datetime(df['timestamp'], utc=True)
        df = df[['open','high','low','close','volume']].copy()
        df.index = idx
        df_list.append(df)
    out = pd.concat(df_list, axis=0).sort_index()
    out = out[~out.index.duplicated(keep='last')]
    return out


def make_supervised(df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    df = df.copy()
    df["ret_1"] = np.log(df["close"]).diff()
    step = {"15m": 15, "1h": 60, "1d": 60 * 24}[horizon]
    df["ret_t+h"] = np.log(df["close"]).shift(-step) - np.log(df["close"])
    df = add_basic_indicators(df)
    df = df.dropna()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="BTCUSDC,ETHUSDC,LTCUSDC")
    ap.add_argument("--timeframe", required=True, help="1m|5m|15m|1h|4h")
    ap.add_argument("--horizon", required=True, help="15m|1h|1d")
    ap.add_argument("--out", default="src/models/quantile_models")
    args = ap.parse_args()

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    feats: List[str] = [
        "rsi_14","bb_width_20","adx_14","ret_1_ema_20","vol_ema_20","sma_20","sma_50","sma_ratio_20_50"
    ]
    for sym in [s.strip() for s in args.symbols.split(',') if s.strip()]:
        df = load_partition(sym, args.timeframe)
        sup = make_supervised(df, args.horizon)
        X = sup[feats]
        y = sup["ret_t+h"]
        spec = LoadModelSpec(symbol=sym, timeframe=args.timeframe, horizon=args.horizon, features=feats, target="ret_t+h")
        qf = LoadQuantileForecaster(spec).fit(X, y)
        out_path = f"{args.out}/{sym}_{args.timeframe}_{args.horizon}_load_qf.pkl"
        qf.save(out_path)
        print(f"[OK] saved {sym} {args.timeframe} {args.horizon} -> {out_path}")


if __name__ == "__main__":
    main()

