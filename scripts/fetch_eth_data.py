import subprocess
import sys


TF_LIMITS = {
    "1m": 20000,
    "5m": 20000,
    "15m": 10000,
    "1h": 5000,
    "4h": 3000,
}


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    symbol = "ETHUSDT"
    for tf, limit in TF_LIMITS.items():
        run([sys.executable, "-m", "src.data.download_data", "--symbols", symbol, "--timeframe", tf, "--limit", str(limit)])


if __name__ == "__main__":
    main()

