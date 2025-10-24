"""特征提取模块：提取各频段功率与熵等特征。"""
from __future__ import annotations

import importlib.util
import pathlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch

_ROOT = pathlib.Path(__file__).resolve().parents[1]
feat_cfg = _ROOT / "01_config" / "feature_config.py"
dev_cfg = _ROOT / "01_config" / "device_config.py"


def _import_module(path: pathlib.Path, name: str):
	spec = importlib.util.spec_from_file_location(name, str(path))
	mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
	assert spec and spec.loader
	spec.loader.exec_module(mod)  # type: ignore[union-attr]
	return mod


feature_config = _import_module(feat_cfg, "feature_config")
device_config = _import_module(dev_cfg, "device_config")
logger_mod = _import_module(_ROOT / "04_utils" / "logger.py", "logger")
get_logger = getattr(logger_mod, "get_logger")
log = get_logger(__name__)


class FeatureExtractor:
	def __init__(self, bands: Dict[str, Tuple[float, float]] | None = None):
		self.bands = bands or feature_config.BANDS

	def bandpower(self, data: np.ndarray) -> Dict[str, np.ndarray]:
		"""计算各频段功率（每通道一维）。"""
		if data.size == 0:
			return {k: np.zeros((0,), dtype=np.float32) for k in self.bands}
		fs = float(device_config.SAMPLE_RATE)
		# Welch
		freqs, psd = welch(data, fs=fs, nperseg=min(256, max(64, data.shape[0] // 2)), axis=0)
		bp: Dict[str, np.ndarray] = {}
		for name, (lo, hi) in self.bands.items():
			idx = (freqs >= lo) & (freqs < hi)
			val = np.trapz(psd[idx, :], freqs[idx], axis=0)
			bp[name] = val.astype(np.float32)
		return bp

	def spectral_entropy(self, data: np.ndarray) -> np.ndarray:
		if data.size == 0:
			return np.zeros((0,), dtype=np.float32)
		fs = float(device_config.SAMPLE_RATE)
		freqs, psd = welch(data, fs=fs, nperseg=min(256, max(64, data.shape[0] // 2)), axis=0)
		psd_norm = psd / (psd.sum(axis=0, keepdims=True) + 1e-8)
		ent = -(psd_norm * np.log(psd_norm + 1e-12)).sum(axis=0)
		return ent.astype(np.float32)

	def extract(self, data: np.ndarray) -> pd.DataFrame:
		"""提取窗口的特征并返回 DataFrame（每行对应一个通道）。"""
		bp = self.bandpower(data)
		ent = self.spectral_entropy(data)
		rows = []
		for ch_idx, ch_name in enumerate(device_config.DEVICE_CHANNELS):
			row = {f"{band}_power": bp[band][ch_idx] for band in self.bands.keys()}
			row["entropy"] = ent[ch_idx]
			row["channel"] = ch_name
			rows.append(row)
		df = pd.DataFrame(rows)
		return df

	def save_features(self, df: pd.DataFrame, subject: str = "S001") -> pathlib.Path:
		out_dir = _ROOT / "02_data" / "features"
		out_dir.mkdir(parents=True, exist_ok=True)
		path = out_dir / f"{subject}_{int(pathlib.time.time())}_features.csv"
		# 修正：pathlib.time 不存在
		import time
		path = out_dir / f"{subject}_{int(time.time())}_features.csv"
		df.to_csv(path, index=False, encoding="utf-8")
		log.info(f"已保存特征至 {path}")
		return path
