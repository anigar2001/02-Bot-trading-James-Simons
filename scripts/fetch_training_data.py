import os
import subprocess
import sys


SYMBOLS = os.getenv("TRAIN_SYMBOLS", "BTCUSDT,ETHUSDT,LTCUSDT")
TIMEFRAMES = os.getenv("TRAIN_TFS", "1h").split(",")
LIMIT = os.getenv("TRAIN_LIMIT", "5000")
START = os.getenv("TRAIN_START")  # opcional, YYYY-MM-DD


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    syms = [s.strip() for s in SYMBOLS.split(',') if s.strip()]
    for tf in TIMEFRAMES:
        for s in syms:
            cmd = [sys.executable, "-m", "src.data.download_data", "--symbols", s, "--timeframe", tf, "--limit", str(LIMIT)]
            if START:
                cmd += ["--start", START]
            run(cmd)


if __name__ == "__main__":
    main()

