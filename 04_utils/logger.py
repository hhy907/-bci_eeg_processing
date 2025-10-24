import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parents[1] / "02_data"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "bci_pipeline.log"


def get_logger(name: str = "bci") -> logging.Logger:
	logger = logging.getLogger(name)
	if logger.handlers:
		return logger
	logger.setLevel(logging.DEBUG)

	fmt = logging.Formatter(
		fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S"
	)

	ch = logging.StreamHandler()
	ch.setLevel(logging.INFO)
	ch.setFormatter(fmt)
	logger.addHandler(ch)

	fh = RotatingFileHandler(_LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(fmt)
	logger.addHandler(fh)

	logger.propagate = False
	return logger
