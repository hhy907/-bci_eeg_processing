"""Simple tests for EEGPreprocessor.

These are minimal smoke tests to ensure the preprocessing pipeline runs
with the available optional dependencies (mne / neurokit2 / scipy).
"""
import numpy as np
import importlib.util, pathlib

# load preprocessing module by path (package name starts with digits so normal import is invalid)
pp_path = pathlib.Path(__file__).resolve().parents[1] / "03_core" / "preprocessing.py"
spec = importlib.util.spec_from_file_location("preprocessing", str(pp_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore
EEGPreprocessor = getattr(mod, "EEGPreprocessor")


def test_preprocess_smoke():
	# create 2 seconds of synthetic 8-channel data at SAMPLE_RATE
	import importlib.util, pathlib
	cfg_path = pathlib.Path(__file__).resolve().parents[1] / "config" / "device_config.py"
	spec = importlib.util.spec_from_file_location("device_config", str(cfg_path))
	device = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(device)  # type: ignore
	SAMPLE_RATE = getattr(device, "SAMPLE_RATE")

	n_ch = 8
	sr = SAMPLE_RATE
	t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
	data = np.stack([np.sin(2 * np.pi * 10 * t) for _ in range(n_ch)], axis=1)

	proc = EEGPreprocessor()
	out = proc.process(data)
	assert out.shape == data.shape
	assert out.dtype == np.float32
# 测试预处理效果：绘制前后对比图（略）
