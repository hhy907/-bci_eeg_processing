"""一键运行主程序：采集→预处理→特征→分类→存储。"""
from __future__ import annotations

import importlib.util
import pathlib
import time
import os

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent


def _import(path: pathlib.Path, name: str):
	spec = importlib.util.spec_from_file_location(name, str(path))
	mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
	assert spec and spec.loader
	spec.loader.exec_module(mod)  # type: ignore[union-attr]
	return mod


device_config = _import(ROOT / "config" / "device_config.py", "device_config")
logger_mod = _import(ROOT / "04_utils" / "logger.py", "logger")
get_logger = getattr(logger_mod, "get_logger")
log = get_logger("bci.main")

EEGAcquirer = _import(ROOT / "03_core" / "data_acquisition.py", "data_acquisition").EEGAcquirer
EEGPreprocessor = _import(ROOT / "03_core" / "preprocessing.py", "preprocessing").EEGPreprocessor
EmotionClassifier = _import(ROOT / "03_core" / "emotion_classification.py", "emotion_classification").EmotionClassifier


def run_loop():
	acquirer = EEGAcquirer(simulate=True)  # 没有设备时可先用模拟
	preprocessor = EEGPreprocessor()
	classifier = EmotionClassifier()

	acquirer.start()
	max_seconds = float(os.getenv("BCI_MAX_SECONDS", "0") or 0)
	start = time.time()
	try:
		while True:
			raw_data, ts = acquirer.get_lsl_data()
			if raw_data.size == 0:
				time.sleep(0.1)
				continue
			processed = preprocessor.process(raw_data)
			labels = classifier.predict(processed, ts)
			classifier.save_labels(labels)
			# 示例：也可以保存原始片段
			# acquirer.save_raw(raw_data, ts)
			time.sleep(1.0)
			if max_seconds > 0 and (time.time() - start) >= max_seconds:
				log.info(f"达到限定时长 {max_seconds}s，正常退出。")
				break
	except KeyboardInterrupt:
		log.info("收到中断，准备停止...")
	finally:
		acquirer.stop()


if __name__ == "__main__":
	run_loop()
