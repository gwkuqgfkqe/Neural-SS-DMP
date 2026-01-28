"""Base data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import compute_psd_features

@dataclass
class Recording:
    subject_id: str
    dataset: str
    neural: np.ndarray
    target: np.ndarray
    target_vel: Optional[np.ndarray]
    dt: float
    valid_mask: Optional[np.ndarray] = None
    raw_target: Optional[np.ndarray] = None
    neural_standardized: bool = False


class WindowedDataset(Dataset):
    """Windowed dataset for causal decoding."""

    def __init__(
        self,
        recording: Recording,
        history_steps: int,
        horizon_steps: int,
        subject_a: Optional[np.ndarray] = None,
        subject_b: Optional[np.ndarray] = None,
        stride: int = 1,
        target_mode: str = "future",
        feature_cfg: Optional[Dict[str, any]] = None,
    ) -> None:
        self.recording = recording
        self.history_steps = history_steps
        self.horizon_steps = horizon_steps
        self.subject_a = subject_a
        self.subject_b = subject_b
        self.stride = max(int(stride), 1)
        self.target_mode = target_mode
        self.feature_cfg = feature_cfg or {}
        if self.target_mode == "within" and self.horizon_steps >= self.history_steps:
            raise ValueError("target_mode='within' requires horizon_steps < history_steps.")

        self._build_indices()

    def _build_indices(self) -> None:
        n = self.recording.target.shape[0]
        if self.target_mode == "within":
            total = self.history_steps
        else:
            total = self.history_steps + self.horizon_steps
        valid = np.ones(n, dtype=bool)
        if self.recording.valid_mask is not None:
            valid = valid & self.recording.valid_mask
        ok = np.convolve(valid.astype(int), np.ones(total, dtype=int), mode="valid")
        indices = np.where(ok == total)[0]
        self.indices = indices[:: self.stride]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t0 = self.indices[idx]
        t1 = t0 + self.history_steps
        t2 = t1 + self.horizon_steps
        if self.target_mode == "within":
            target_start = t1 - self.horizon_steps
            target_end = t1
            denom = max(self.history_steps - 1, 1)
            target_start_ratio = (self.history_steps - self.horizon_steps - 1) / denom
            target_start_ratio = float(max(0.0, min(1.0, target_start_ratio)))
        else:
            target_start = t1
            target_end = t2
            target_start_ratio = None

        x = self.recording.neural[t0:t1]
        feature_mode = self.feature_cfg.get("ecog_features")
        if self.recording.dataset in ("mouse", "bci_iv", "ajile12"):
            if feature_mode == "psd":
                x = compute_psd_features(x, fs=1.0 / self.recording.dt, cfg=self.feature_cfg.get("psd"))
            elif feature_mode in ("raw+psd", "raw_psd"):
                x_psd = compute_psd_features(x, fs=1.0 / self.recording.dt, cfg=self.feature_cfg.get("psd"))
        y_future = self.recording.target[target_start:target_end]
        y_future_raw = None
        if self.recording.raw_target is not None:
            y_future_raw = self.recording.raw_target[target_start:target_end]

        y0 = self.recording.target[target_start - 1]
        if self.recording.target_vel is not None:
            dy0 = self.recording.target_vel[target_start - 1]
        else:
            if target_start - 2 < 0:
                dy0 = np.zeros_like(y0)
            else:
                dy0 = (self.recording.target[target_start - 1] - self.recording.target[target_start - 2]) / self.recording.dt

        batch = {
            "x": torch.from_numpy(x).float(),
            "y_future": torch.from_numpy(y_future).float(),
            "y0": torch.from_numpy(y0).float(),
            "dy0": torch.from_numpy(dy0).float(),
        }
        if target_start_ratio is not None:
            batch["target_start_ratio"] = torch.tensor(target_start_ratio, dtype=torch.float32)
        if feature_mode in ("raw+psd", "raw_psd") and self.recording.dataset in ("mouse", "bci_iv", "ajile12"):
            batch["x_psd"] = torch.from_numpy(x_psd).float()
        if y_future_raw is not None:
            batch["y_future_raw"] = torch.from_numpy(y_future_raw).float()
        if self.subject_a is not None:
            batch["a_mat"] = torch.from_numpy(self.subject_a).float()
        if self.subject_b is not None:
            batch["b_mat"] = torch.from_numpy(self.subject_b).float()
        return batch


class MultiDataset(Dataset):
    """Combine multiple datasets with sampling by concatenation."""

    def __init__(self, datasets: List[Dataset]) -> None:
        self.datasets = datasets
        self.offsets = np.cumsum([0] + [len(d) for d in datasets])

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, idx: int):
        ds_idx = np.searchsorted(self.offsets, idx, side="right") - 1
        local_idx = idx - self.offsets[ds_idx]
        return self.datasets[ds_idx][local_idx]
