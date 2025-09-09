import os
import subprocess
import sys


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    timeframe = os.getenv("TRAIN_TF", "1h")
    limit = os.getenv("TRAIN_LIMIT", "5000")
    offline = os.getenv("OFFLINE_SYNTHETIC", "1") == "1"
    symbols = ["BTC/USDT", "ETH/USDT", "LTC/USDT"]
    for s in symbols:
        cmd = [sys.executable, "-m", "src.models.model_training", "--symbol", s, "--timeframe", timeframe, "--limit", str(limit)]
        if offline:
            cmd.append("--offline")
        run(cmd)


if __name__ == "__main__":
    main()

