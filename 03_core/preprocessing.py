

"""EEG 预处理模块：带通滤波、重采样与 NeuroKit2 自动伪迹去除（最佳努力）以及 scipy 退化路径。"""
from __future__ import annotations

import importlib.util
import pathlib
from typing import Optional

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
cfg_path = _ROOT / "config" / "device_config.py"


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

# optional dependencies
try:
    import neurokit2 as nk  # type: ignore
except Exception:
    nk = None  # type: ignore

try:
    from scipy import signal as _spsignal  # type: ignore
except Exception:
    _spsignal = None  # type: ignore

logger_mod = _import_module(_ROOT / "04_utils" / "logger.py", "logger")
get_logger = getattr(logger_mod, "get_logger")
log = get_logger(__name__)


class EEGPreprocessor:
    def __init__(self, l_freq: float = 1.0, h_freq: float = 45.0, resample_hz: Optional[int] = 128):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.resample_hz = resample_hz

    def process(self, data: np.ndarray) -> np.ndarray:
        """Process raw EEG data.

        Args:
            data: numpy array shaped (n_samples, n_channels)

        Returns:
            clean: numpy array shaped (n_samples, n_channels), dtype float32
        """
        if data is None or data.size == 0:
            return data
        sr = device_config.SAMPLE_RATE

        # NOTE: NeuroKit2 cleaning will be applied after filtering (see MNE path).

        # If MNE is available use its robust filtering/resample utilities
        if mne is None:
            # Fallback: use scipy filtering if available, otherwise pass-through
            if _spsignal is None:
                log.warning("mne and scipy.signal both unavailable — returning raw data cast to float32")
                return data.astype(np.float32)
            # scipy fallback: apply zero-phase bandpass then optional resample
            nyq = 0.5 * sr
            low = max(self.l_freq / nyq, 1e-8)
            high = min(self.h_freq / nyq, 0.999999)
            if low >= high:
                log.warning("invalid filter band, returning raw data")
                clean = data.astype(np.float32)
            else:
                sos = _spsignal.butter(4, [low, high], btype='bandpass', output='sos')
                # data shape: (n_samples, n_channels)
                try:
                    clean = _spsignal.sosfiltfilt(sos, data, axis=0).astype(np.float32)
                except Exception:
                    # If sosfiltfilt fails (very short data), fall back to sosfilt
                    clean = _spsignal.sosfilt(sos, data, axis=0).astype(np.float32)
                if self.resample_hz and self.resample_hz != sr:
                    # resample each channel using resample_poly
                    p = int(self.resample_hz)
                    q = int(sr)
                    try:
                        clean = _spsignal.resample_poly(clean, p, q, axis=0).astype(np.float32)
                    except Exception:
                        log.warning("scipy resample failed; returning filtered (unresampled) data")
            return clean

        # MNE path
        info = mne.create_info(
            ch_names=[str(c) for c in range(data.shape[1])],
            sfreq=sr,
            ch_types=['eeg'] * data.shape[1]
        )
        raw = mne.io.RawArray(data.T, info, verbose='ERROR')
        # MNE filtering
        raw.filter(self.l_freq, self.h_freq, verbose='ERROR')
        # If NeuroKit2 is available, apply channel-wise cleaning once (after filtering).
        # Create a new RawArray from cleaned data instead of writing to raw._data.
        if nk is not None:
            try:
                arr = raw.get_data().T
                arr = self._apply_neurokit_clean(arr, int(raw.info['sfreq']))
                # build a new RawArray from cleaned data (preserve info)
                cleaned_raw = mne.io.RawArray(arr.T, info, verbose='ERROR')
                raw = cleaned_raw
            except Exception:
                log.debug("neurokit2 in-mne-path cleaning failed; continuing")
        if self.resample_hz and self.resample_hz != sr:
            raw.resample(self.resample_hz, verbose='ERROR')
        clean = raw.get_data().T.astype(np.float32)
        return clean

    def _apply_neurokit_clean(self, data: np.ndarray, sr: int) -> np.ndarray:
        """Best-effort NeuroKit2 cleaning for multichannel data.

        Data shape: (n_samples, n_channels).
        This tries a few common NeuroKit2 APIs and falls back gracefully.
        """
        if nk is None:
            return data
        clean = np.asarray(data, dtype=np.float32).copy()
        # Ensure 2D
        if clean.ndim == 1:
            clean = clean[:, None]
        n_ch = clean.shape[1]
        for ch in range(n_ch):
            sig = clean[:, ch]
            ok = False
            # Try high-level eeg_clean (if available)
            if hasattr(nk, 'eeg_clean'):
                try:
                    out = nk.eeg_clean(sig, sampling_rate=sr)
                    # eeg_clean may return array or dict
                    if isinstance(out, dict) and 'Signal' in out:
                        sig = np.asarray(out['Signal'], dtype=np.float32)
                    else:
                        sig = np.asarray(out, dtype=np.float32)
                    ok = True
                except Exception:
                    pass
            # Try generic nk.signal.clean
            if not ok and hasattr(nk, 'signal') and hasattr(nk.signal, 'clean'):
                try:
                    sig = np.asarray(nk.signal.clean(sig, sampling_rate=sr), dtype=np.float32)
                    ok = True
                except Exception:
                    pass
            # Try fallback: nk.signal.filter if available
            if not ok and hasattr(nk, 'signal') and hasattr(nk.signal, 'filter'):
                try:
                    sig = np.asarray(nk.signal.filter(sig, self.l_freq, self.h_freq, sampling_rate=sr), dtype=np.float32)
                    ok = True
                except Exception:
                    pass
            # If nothing worked, leave signal as-is
            clean[:, ch] = sig
        return clean
