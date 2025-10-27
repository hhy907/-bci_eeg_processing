"""
EEG数据采集模块
- 通过 Emotiv Cortex SDK 订阅 EEG 数据
- 将样本转发到 LSL（名称见 config/lsl_config.py）
- 提供从 LSL 读取数据的辅助方法，便于 main.py 串联
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import numpy as np
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop
import threading

# 动态导入以支持数字开头的包路径
import importlib.util
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg_path = _ROOT / "config" / "device_config.py"
lsl_path = _ROOT / "config" / "lsl_config.py"

def _import_module(path: pathlib.Path, name: str):
	spec = importlib.util.spec_from_file_location(name, str(path))
	mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
	assert spec and spec.loader
	spec.loader.exec_module(mod)  # type: ignore[union-attr]
	return mod

device_config = _import_module(cfg_path, "device_config")
lsl_config = _import_module(lsl_path, "lsl_config")
logger_mod = _import_module(_ROOT / "04_utils" / "logger.py", "logger")
get_logger = getattr(logger_mod, "get_logger")

try:
	# Emotiv Cortex SDK（Python）
	from cortex import Cortex
except Exception:  # pragma: no cover - 未安装时主流程可用模拟模式
	Cortex = None  # type: ignore

log = get_logger(__name__)


class EEGAcquirer:
	"""EEG 采集器：连接 Emotiv，转发 LSL，并可从 LSL 读取。"""

	def __init__(self, client_id: str | None = None, client_secret: str | None = None,
				 simulate: bool = False):
		self.simulate = simulate or Cortex is None
		self.client_id = client_id or os.getenv("EMOTIV_CLIENT_ID", "")
		self.client_secret = client_secret or os.getenv("EMOTIV_CLIENT_SECRET", "")

		# 输出 LSL 流（EEG）
		self.info = StreamInfo(
			name=lsl_config.EEG_STREAM_NAME,
			type=lsl_config.EEG_STREAM_TYPE,
			channel_count=len(device_config.DEVICE_CHANNELS),
			nominal_srate=device_config.SAMPLE_RATE,
			channel_format='float32',
			source_id='emotiv_eeg_001'
		)
		# 附加通道标签元数据
		chns = self.info.desc().append_child("channels")
		for label in device_config.DEVICE_CHANNELS:
			ch = chns.append_child("channel")
			ch.append_child_value("label", label)
			ch.append_child_value("unit", "uV")
			ch.append_child_value("type", "EEG")

		self.outlet = StreamOutlet(self.info)
		self._inlet: StreamInlet | None = None

		self._cortex = None  # type: ignore[assignment]
		self._sim_thread: threading.Thread | None = None
		self._stop_flag = threading.Event()
		if not self.simulate:
			self._cortex = Cortex(client_id=self.client_id, client_secret=self.client_secret)

	# ------- Emotiv 回调与推送 LSL -------
	def on_eeg_data(self, data: List[float]):
		"""Emotiv 数据回调：data = [timestamp, ch1, ch2, ..., quality]
		仅推送通道数据至 LSL，使用原始时间戳。"""
		if not data:
			return
		timestamp = float(data[0])
		# 取前 N 通道
		n = len(device_config.DEVICE_CHANNELS)
		eeg_sample = list(map(float, data[1:1+n]))
		self.outlet.push_sample(eeg_sample, timestamp)
		if len(data) > 1 + n:
			sigq = data[-1]
			log.debug(f"推送EEG：ts={timestamp:.3f} ch={eeg_sample} 质量={sigq}")
		else:
			log.debug(f"推送EEG：ts={timestamp:.3f} ch={eeg_sample}")

	def start(self):
		"""启动设备采集，并建立一个本地 LSL inlet 以便读取。"""
		if self.simulate:
			log.warning("未检测到 Emotiv Cortex SDK，将使用模拟数据流用于调试。")
			# 启动模拟线程，按采样率推送随机数据
			def _runner():
				n = len(device_config.DEVICE_CHANNELS)
				period = 1.0 / float(device_config.SAMPLE_RATE)
				while not self._stop_flag.is_set():
					t = time.time()
					sample = [t] + [float(np.random.randn()*8.0) for _ in range(n)] + [100]
					self.on_eeg_data(sample)
					time.sleep(period)
			self._sim_thread = threading.Thread(target=_runner, daemon=True)
			self._sim_thread.start()
		else:
			assert self._cortex is not None
			self._cortex.do_prepare_steps()
			self._cortex.sub_request("eeg", self.on_eeg_data)
			log.info("Emotiv EEG 订阅已启动并转发至 LSL。")

		# 建立 inlet（读取自身发出的流或外部同名流）
		streams = resolve_byprop('name', lsl_config.EEG_STREAM_NAME, timeout=3)
		if not streams:
			# 等待片刻让 outlet 建立广播
			time.sleep(0.5)
			streams = resolve_byprop('name', lsl_config.EEG_STREAM_NAME, timeout=5)
		if streams:
			self._inlet = StreamInlet(streams[0])
		else:
			log.error("未找到匹配的 LSL EEG 流，get_lsl_data 将不可用。")

	def stop(self):
		self._stop_flag.set()
		if self._sim_thread and self._sim_thread.is_alive():
			self._sim_thread.join(timeout=1.0)
		if self._cortex:
			try:
				self._cortex.close()
			except Exception:
				pass
		log.info("EEG 采集已停止。")

	# ------- 从 LSL 读取数据（主循环用） -------
	def get_lsl_data(self, max_chunk_len: int = 256) -> Tuple[np.ndarray, np.ndarray]:
		"""从 LSL 读取一小段数据块。

		返回
		- data: shape=(samples, channels)
		- timestamps: shape=(samples,)
		"""
		if self._inlet is None:
			return np.empty((0, len(device_config.DEVICE_CHANNELS)), dtype=np.float32), np.empty((0,), dtype=np.float64)
		samples, timestamps = self._inlet.pull_chunk(timeout=0.0, max_samples=max_chunk_len)
		if not samples:
			return np.empty((0, len(device_config.DEVICE_CHANNELS)), dtype=np.float32), np.empty((0,), dtype=np.float64)
		data = np.asarray(samples, dtype=np.float32)
		ts = np.asarray(timestamps, dtype=np.float64)
		return data, ts

	# 可选：将原始数据保存为 .npy（用于 test_acquisition）
	def save_raw(self, data: np.ndarray, timestamps: np.ndarray, subject: str = "S001"):
		data_dir = _ROOT / "02_data" / "raw_eeg"
		data_dir.mkdir(parents=True, exist_ok=True)
		t = int(time.time())
		np.save(data_dir / f"{subject}_{t}_eeg.npy", data)
		np.save(data_dir / f"{subject}_{t}_ts.npy", timestamps)
		log.info(f"已保存原始EEG与时间戳至 {data_dir}")


# 简易模拟器（无设备时可用于开发联调）
if Cortex is None:
	class _SimRunner:
		def __init__(self, acq: EEGAcquirer):
			self.acq = acq

		def run_once(self):
			sr = device_config.SAMPLE_RATE
			n = len(device_config.DEVICE_CHANNELS)
			t = time.time()
			sample = [t] + [np.random.randn()*10 for _ in range(n)] + [100]  # 质量=100
			self.acq.on_eeg_data(sample)

