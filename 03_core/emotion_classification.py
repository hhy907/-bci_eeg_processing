"""情绪分类模块占位：可替换为 Emotiv/Braindecode。

当前实现：
- 简单规则或阈值，生成 {Focus/Relax/Anxiety} 标签
- 通过 LSL 转发到 EMOTION_STREAM_NAME
"""
from __future__ import annotations

import importlib.util
import pathlib
from typing import List

import numpy as np
from pylsl import StreamInfo, StreamOutlet
import time
import csv

_ROOT = pathlib.Path(__file__).resolve().parents[1]
lsl_cfg = _ROOT / "01_config" / "lsl_config.py"


def _import_module(path: pathlib.Path, name: str):
	spec = importlib.util.spec_from_file_location(name, str(path))
	mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
	assert spec and spec.loader
	spec.loader.exec_module(mod)  # type: ignore[union-attr]
	return mod


lsl_config = _import_module(lsl_cfg, "lsl_config")
logger_mod = _import_module(_ROOT / "04_utils" / "logger.py", "logger")
get_logger = getattr(logger_mod, "get_logger")
log = get_logger(__name__)


class EmotionClassifier:
	def __init__(self):
		self.outlet = StreamOutlet(StreamInfo(
			name=lsl_config.EMOTION_STREAM_NAME,
			type="Markers",
			channel_count=1,
			nominal_srate=0.0,
			channel_format='string',
			source_id='emotion_labels_001'
		))

	def predict(self, processed_data: np.ndarray, timestamps: np.ndarray) -> List[tuple[float, str]]:
		labels: List[tuple[float, str]] = []
		if processed_data.size == 0:
			return labels
		# 简单规则：计算每个时间窗口所有通道的 alpha/beta 比例占位（此处直接用方差代替）
		energy = processed_data.var(axis=1)
		for ts, e in zip(timestamps, energy):
			if e > np.percentile(energy, 66):
				lab = "Focus"
			elif e < np.percentile(energy, 33):
				lab = "Relax"
			else:
				lab = "Anxiety"
			self.outlet.push_sample([lab], float(ts))
			labels.append((float(ts), lab))
		log.debug(f"输出 {len(labels)} 条情绪标签到 LSL")
		return labels

	def save_labels(self, labels: List[tuple[float, str]], subject: str = "S001") -> pathlib.Path | None:
		if not labels:
			return None
		out_dir = _ROOT / "02_data" / "emotion_labels"
		out_dir.mkdir(parents=True, exist_ok=True)
		path = out_dir / f"{subject}_{int(time.time())}_labels.csv"
		with path.open("w", newline="", encoding="utf-8") as f:
			w = csv.writer(f)
			w.writerow(["timestamp", "label"])
			w.writerows(labels)
		log.info(f"已保存情绪标签至 {path}")
		return path
