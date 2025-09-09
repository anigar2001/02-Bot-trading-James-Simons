import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable


def ensure_dirs(paths: Iterable[str]):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def setup_logging(log_dir: str = "src/data/logs"):
    """Configura logging con rotación a archivo y salida a consola."""
    ensure_dirs([log_dir])
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Evitar handlers duplicados si se llama varias veces
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler(os.path.join(log_dir, "bot.log"), maxBytes=2_000_000, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def json_log(path: str, payload: dict):
    """Append line JSON al archivo indicado."""
    ensure_dirs([str(Path(path).parent)])
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

