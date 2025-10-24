"""EEG 预处理模块：带通滤波、重采样与简易去伪迹占位。"""
from __future__ import annotations

import importlib.util
import pathlib
from typing import Optional

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg_path = _ROOT / "01_config" / "device_config.py"

def _import_module(path: pathlib.Path, name: str):
	spec = importlib.util.spec_from_file_location(name, str(path))
	mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
	assert spec and spec.loader
	spec.loader.exec_module(mod)  # type: ignore[union-attr]
	return mod

device_config = _import_module(cfg_path, "device_config")

try:
	import mne  # type: ignore
except Exception:
	mne = None  # type: ignore

logger_mod = _import_module(_ROOT / "04_utils" / "logger.py", "logger")
get_logger = getattr(logger_mod, "get_logger")
log = get_logger(__name__)


class EEGPreprocessor:
	def __init__(self, l_freq: float = 1.0, h_freq: float = 45.0, resample_hz: Optional[int] = 128):
		self.l_freq = l_freq
		self.h_freq = h_freq
		self.resample_hz = resample_hz

	def process(self, data: np.ndarray) -> np.ndarray:
		if data.size == 0:
			return data
		sr = device_config.SAMPLE_RATE
		if mne is None:
			# 退化路径：简单高通+低通 IIR（占位）
			return data.astype(np.float32)

		info = mne.create_info(
			ch_names=[str(c) for c in range(data.shape[1])],
			sfreq=sr,
			ch_types=['eeg'] * data.shape[1]
		)
		raw = mne.io.RawArray(data.T, info, verbose='ERROR')
		raw.filter(self.l_freq, self.h_freq, verbose='ERROR')
		if self.resample_hz and self.resample_hz != sr:
			raw.resample(self.resample_hz, verbose='ERROR')
		clean = raw.get_data().T.astype(np.float32)
		return clean
