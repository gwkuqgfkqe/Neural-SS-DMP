"""H1 (Human 7DoF) dataset loader."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List

import h5py
import numpy as np

from .base import Recording
from .utils import exp_smooth, finite_diff, reconstruct_target, resample_to_rate


def _load_spike_counts(f: h5py.File, edges: np.ndarray) -> np.ndarray:
    spike_times = f["units"]["spike_times"][()]
    spike_idx = f["units"]["spike_times_index"][()]
    n_units = spike_idx.shape[0]

    counts = np.zeros((edges.shape[0] - 1, n_units), dtype=np.float32)
    start = 0
    for i, end in enumerate(spike_idx):
        st = spike_times[start:end]
        hist, _ = np.histogram(st, bins=edges)
        counts[:, i] = hist
        start = end
    return counts


def _interp_to_times(y: np.ndarray, src_t: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.interp(dst_t, src_t, y).astype(np.float32)
    out = np.empty((dst_t.shape[0], y.shape[1]), dtype=np.float32)
    for d in range(y.shape[1]):
        out[:, d] = np.interp(dst_t, src_t, y[:, d]).astype(np.float32)
    return out


def load_h1_recordings(cfg: Dict[str, Any], preprocess_cfg: Dict[str, Any]) -> List[Recording]:
    data_dir = cfg["data_dir"]
    split = cfg.get("split", "held-in-minival")
    use_signal = cfg.get("use_signal", "spikes")
    target_field = cfg.get("target", "openloop_pos")
    neural_rate = cfg.get("neural_rate_hz")
    bin_size_sec = cfg.get("bin_size_sec")
    spike_preprocess = cfg.get("spike_preprocess", "log1p")
    falcon_preprocess = cfg.get("falcon_preprocess", {})
    exp_tau_sec = float(falcon_preprocess.get("exp_tau_sec", 0.05))

    pattern = os.path.join(data_dir, f"sub-HumanPitt-{split}", "*.nwb")
    files = sorted(glob.glob(pattern))
    recordings: List[Recording] = []

    for path in files:
        with h5py.File(path, "r") as f:
            kin = f["acquisition"]["OpenLoopKinematics"]["data"][()]
            vel = f["acquisition"]["OpenLoopKinematicsVelocity"]["data"][()]
            start = float(f["acquisition"]["OpenLoopKinematics"]["starting_time"][()])
            rate = f["acquisition"]["OpenLoopKinematics"]["starting_time"].attrs.get("rate")
            dt = float(rate) if rate is not None else 0.02
            ts = start + np.arange(kin.shape[0], dtype=np.float64) * dt
            eval_mask = f["acquisition"]["eval_mask"]["data"][()].astype(bool)

            if use_signal != "spikes":
                raise ValueError(f"Unsupported H1 signal '{use_signal}'")

            if bin_size_sec is not None:
                bin_dt = float(bin_size_sec)
                t0 = float(ts[0])
                t1 = float(ts[-1])
                n_bins = int(np.floor((t1 - t0) / bin_dt)) + 1
                if n_bins < 2:
                    continue
                edges = t0 + np.arange(n_bins + 1, dtype=np.float64) * bin_dt
                bin_t = edges[:-1]
                neural = _load_spike_counts(f, edges)
                kin = _interp_to_times(kin, ts, bin_t)
                vel = _interp_to_times(vel, ts, bin_t)
                eval_mask = (_interp_to_times(eval_mask.astype(np.float32), ts, bin_t) > 0.5).astype(bool)
                ts = bin_t
            else:
                dt0 = float(np.median(np.diff(ts)))
                edges = np.concatenate([ts, [ts[-1] + dt0]])
                neural = _load_spike_counts(f, edges)

        dt = float(np.median(np.diff(ts)))
        if target_field in ("openloop_vel", "kin_vel", "velocity"):
            target = vel
        else:
            target = kin

        behavior_rate = 1.0 / dt
        neural_rate_use = behavior_rate
        align_rate = max(neural_rate_use, behavior_rate)
        if not np.isclose(neural_rate_use, align_rate):
            neural = resample_to_rate(neural, neural_rate_use, align_rate)
        if not np.isclose(behavior_rate, align_rate):
            target = resample_to_rate(target, behavior_rate, align_rate)
            vel = resample_to_rate(vel, behavior_rate, align_rate)
            eval_mask = resample_to_rate(eval_mask.astype(np.float32), behavior_rate, align_rate) > 0.5
        dt = 1.0 / align_rate

        if spike_preprocess == "log1p":
            neural = np.log1p(neural)
        elif spike_preprocess == "falcon":
            neural = exp_smooth(neural.astype(np.float32), dt=dt, tau_sec=exp_tau_sec)
        else:
            raise ValueError(f"Unknown spike_preprocess '{spike_preprocess}'")

        raw_target = target.copy()
        if target_field not in ("openloop_vel", "kin_vel", "velocity"):
            target = reconstruct_target(target, dt, preprocess_cfg.get("target_reconstruct"))
            vel = finite_diff(target, dt)

        recordings.append(
            Recording(
                subject_id=os.path.basename(os.path.dirname(path)).split("-held-", 1)[0],
                dataset="h1",
                neural=neural.astype(np.float32),
                target=target.astype(np.float32),
                target_vel=vel.astype(np.float32),
                dt=dt,
                valid_mask=eval_mask,
                raw_target=raw_target.astype(np.float32),
            )
        )

    return recordings
