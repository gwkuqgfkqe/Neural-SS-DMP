"""Data utilities and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import interpolate, signal


def car_reference(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=1, keepdims=True)
    return x - mean


def bandpass_filter(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    # Use zero-phase filtering for offline decoding to avoid phase distortion.
    return signal.sosfiltfilt(sos, x, axis=0)

def notch_filter(
    x: np.ndarray,
    fs: float,
    freq_hz: float = 60.0,
    q: float = 30.0,
    harmonics: bool = True,
) -> np.ndarray:
    if fs <= 0 or freq_hz <= 0:
        return x
    nyq = 0.5 * fs
    max_hz = nyq - 1.0e-6
    freqs = [freq_hz]
    if harmonics:
        k = 2
        while k * freq_hz < max_hz:
            freqs.append(k * freq_hz)
            k += 1
    y = x
    for f in freqs:
        w0 = f / nyq
        if w0 >= 1.0:
            continue
        b, a = signal.iirnotch(w0, q)
        y = signal.filtfilt(b, a, y, axis=0)
    return y



def downsample_mean(x: np.ndarray, factor: int) -> np.ndarray:
    n = (x.shape[0] // factor) * factor
    x = x[:n]
    new_shape = (n // factor, factor) + x.shape[1:]
    return x.reshape(new_shape).mean(axis=1)


def exp_smooth(x: np.ndarray, dt: float, tau_sec: float) -> np.ndarray:
    if tau_sec is None or tau_sec <= 0:
        return x
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False
    alpha = 1.0 - np.exp(-float(dt) / float(tau_sec))
    alpha = float(min(max(alpha, 0.0), 1.0))
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0].astype(np.float32)
    for t in range(1, x.shape[0]):
        y[t] = (1.0 - alpha) * y[t - 1] + alpha * x[t]
    if squeeze:
        return y[:, 0]
    return y


def resample_to_rate(x: np.ndarray, src_rate: float, dst_rate: float) -> np.ndarray:
    if np.isclose(src_rate, dst_rate):
        return x
    up = int(round(dst_rate))
    down = int(round(src_rate))
    return signal.resample_poly(x, up=up, down=down, axis=0)


def compute_psd_features(x: np.ndarray, fs: float, cfg: Optional[dict] = None) -> np.ndarray:
    if cfg is None:
        cfg = {}
    mode = cfg.get("mode", "stft")
    bands = cfg.get(
        "bands",
        [
            (1.0, 4.0),
            (4.0, 8.0),
            (8.0, 13.0),
            (13.0, 30.0),
            (30.0, 50.0),
            (50.0, 70.0),
            (70.0, 100.0),
            (100.0, 150.0),
        ],
    )
    log_power = cfg.get("log_power", True)
    standardize = cfg.get("standardize", True)

    if x.ndim == 1:
        x = x[:, None]
    n_samples = x.shape[0]

    if mode == "welch":
        nperseg = int(round(float(cfg.get("nperseg_sec", 0.25)) * fs))
        step = int(round(float(cfg.get("step_sec", 0.05)) * fs))
        nperseg = max(2, min(nperseg, n_samples))
        step = max(1, min(step, nperseg))
        starts = np.arange(0, n_samples - nperseg + 1, step, dtype=int)
        if starts.size == 0:
            starts = np.array([0], dtype=int)
        frames = []
        nyq = 0.5 * fs
        for start in starts:
            seg = x[start : start + nperseg]
            f, pxx = signal.welch(seg, fs=fs, nperseg=nperseg, axis=0)
            band_feats = []
            for lo, hi in bands:
                hi_use = min(float(hi), nyq)
                if hi_use <= float(lo):
                    band_power = np.zeros((x.shape[1],), dtype=np.float32)
                else:
                    mask = (f >= float(lo)) & (f <= hi_use)
                    if not np.any(mask):
                        band_power = np.zeros((x.shape[1],), dtype=np.float32)
                    else:
                        band_power = np.mean(pxx[mask], axis=0)
                band_feats.append(band_power.astype(np.float32))
            frames.append(np.concatenate(band_feats, axis=0))
        feats = np.stack(frames, axis=0)
    else:
        nperseg = int(round(float(cfg.get("nperseg_sec", 0.25)) * fs))
        step = int(round(float(cfg.get("step_sec", 0.05)) * fs))
        nperseg = max(2, min(nperseg, n_samples))
        step = max(1, min(step, nperseg))
        noverlap = max(nperseg - step, 0)
        f, _, zxx = signal.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=0, boundary=None)
        power = np.abs(zxx) ** 2
        band_series = []
        for lo, hi in bands:
            mask = (f >= lo) & (f <= hi)
            if not np.any(mask):
                band_power = np.zeros((power.shape[1], x.shape[1]), dtype=np.float32)
            else:
                band_power = np.mean(power[mask], axis=0)
            band_series.append(band_power)
        feats = np.concatenate(band_series, axis=-1)

    if log_power:
        feats = np.log1p(feats)
    if standardize and feats.shape[0] > 1:
        mean = feats.mean(axis=0, keepdims=True)
        std = feats.std(axis=0, keepdims=True) + 1e-6
        feats = (feats - mean) / std
    return feats.astype(np.float32)


def reconstruct_target(
    x: np.ndarray,
    dt: float,
    cfg: Optional[dict] = None,
) -> np.ndarray:
    if cfg is None:
        cfg = {}
    if not cfg.get("enabled", False):
        return x
    method = cfg.get("method", "savgol")
    if method == "savgol":
        window_sec = float(cfg.get("window_sec", 0.05))
        polyorder = int(cfg.get("polyorder", 3))
        win = max(int(round(window_sec / dt)), polyorder + 2)
        if win % 2 == 0:
            win += 1
        if win >= x.shape[0]:
            return x
        return signal.savgol_filter(x, window_length=win, polyorder=polyorder, axis=0, mode="interp")
    if method in ("interp", "linear", "spline", "cubic"):
        kind = "linear" if method in ("interp", "linear") else "cubic"
        x_in = x
        if x_in.ndim == 1:
            x_in = x_in[:, None]
        n, d = x_in.shape
        if n < 2:
            return x
        t = np.arange(n)
        out = np.empty_like(x_in)
        for i in range(d):
            y = x_in[:, i]
            if np.allclose(y, y[0]):
                out[:, i] = y
                continue
            change = np.where(np.abs(np.diff(y)) > 0)[0] + 1
            idx = np.concatenate(([0], change, [n - 1]))
            idx = np.unique(idx)
            kind_use = kind
            if kind_use == "cubic" and idx.size < 4:
                kind_use = "linear"
            f = interpolate.interp1d(
                idx,
                y[idx],
                kind=kind_use,
                fill_value="extrapolate",
                assume_sorted=True,
            )
            out[:, i] = f(t)
        if x.ndim == 1:
            return out[:, 0]
        return out
    if method == "lowpass":
        cutoff = float(cfg.get("lowpass_hz", 10.0))
        order = int(cfg.get("order", 4))
        sos = signal.butter(order, cutoff, btype="lowpass", fs=1.0 / dt, output="sos")
        return signal.sosfiltfilt(sos, x, axis=0)
    raise ValueError(f"Unknown target reconstruction method '{method}'")


def finite_diff(x: np.ndarray, dt: float) -> np.ndarray:
    dx = np.zeros_like(x)
    dx[1:] = (x[1:] - x[:-1]) / dt
    dx[0] = dx[1]
    return dx


def second_diff(x: np.ndarray, dt: float) -> np.ndarray:
    ddx = np.zeros_like(x)
    ddx[1:] = (x[1:] - x[:-1]) / dt
    ddx = finite_diff(ddx, dt)
    return ddx


@dataclass
class Normalizer:
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "Normalizer":
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0) + 1e-6
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            return x
        return (x - self.mean) / self.std
