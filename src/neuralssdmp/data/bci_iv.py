"""BCI Competition IV Dataset 4 loader."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List

import numpy as np
from scipy.io import loadmat

from .base import Recording
from .utils import (
    bandpass_filter,
    car_reference,
    finite_diff,
    notch_filter,
    reconstruct_target,
    resample_to_rate,
)


def _preprocess_neural(x: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    if cfg.get("car", True):
        x = car_reference(x)
    bp = cfg.get("bandpass", {})
    if bp.get("enabled", False):
        x = bandpass_filter(x, cfg["sample_rate_hz"], bp["low_hz"], bp["high_hz"], bp["order"])
    notch = cfg.get("notch", {})
    if notch.get("enabled", False):
        x = notch_filter(x, cfg["sample_rate_hz"], notch.get("freq_hz", 60.0), notch.get("q", 30.0), notch.get("harmonics", True))
    return x


def load_bci_iv_recordings(cfg: Dict[str, Any], preprocess_cfg: Dict[str, Any]) -> List[Recording]:
    data_dir = cfg["data_dir"]
    subjects = cfg.get("subjects")
    sample_rate = cfg["sample_rate_hz"]

    local_preprocess = dict(preprocess_cfg)
    if "bandpass" in cfg:
        local_preprocess["bandpass"] = cfg["bandpass"]
    if "car" in cfg:
        local_preprocess["car"] = cfg["car"]
    if "standardize" in cfg:
        local_preprocess["standardize"] = cfg["standardize"]
    if "notch" in cfg:
        local_preprocess["notch"] = cfg["notch"]

    files = sorted(glob.glob(os.path.join(data_dir, "sub*_comp.mat")))
    recordings: List[Recording] = []

    for path in files:
        name = os.path.basename(path)
        subject = name.split("_")[0]
        if subjects and subject not in subjects:
            continue
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        neural = mat["train_data"].astype(np.float32)
        target = mat["train_dg"].astype(np.float32)

        neural = _preprocess_neural(neural, {**local_preprocess, "sample_rate_hz": sample_rate})
        behavior_rate = cfg.get("behavior_rate_hz", sample_rate)
        if target.shape[0] != neural.shape[0]:
            behavior_rate = sample_rate * (target.shape[0] / neural.shape[0])
        align_rate = max(sample_rate, behavior_rate)

        if not np.isclose(sample_rate, align_rate):
            neural = resample_to_rate(neural, sample_rate, align_rate)
        if not np.isclose(behavior_rate, align_rate):
            target = resample_to_rate(target, behavior_rate, align_rate)

        dt = 1.0 / align_rate
        raw_target = target.copy()
        target = reconstruct_target(target, dt, preprocess_cfg.get("target_reconstruct"))
        target_vel = finite_diff(target, dt)

        recordings.append(
            Recording(
                subject_id=subject,
                dataset="bci_iv",
                neural=neural,
                target=target,
                target_vel=target_vel,
                dt=dt,
                valid_mask=None,
                raw_target=raw_target,
            )
        )

    return recordings
