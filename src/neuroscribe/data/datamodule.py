"""Dataset assembly and splits."""

from __future__ import annotations


import random
from dataclasses import is_dataclass, replace
from typing import Any, Dict, List, Tuple

import numpy as np
from torch.utils.data import DataLoader

try:
    from .ajile12 import AJILE12WindowedDataset, load_ajile12_recordings
except ModuleNotFoundError:
    AJILE12WindowedDataset = None
    load_ajile12_recordings = None
from .base import MultiDataset, Recording, WindowedDataset
try:
    from .bci_iv import load_bci_iv_recordings
except ModuleNotFoundError:
    load_bci_iv_recordings = None
try:
    from .h1 import load_h1_recordings
except ModuleNotFoundError:
    load_h1_recordings = None
try:
    from .m2 import load_m2_recordings
except ModuleNotFoundError:
    load_m2_recordings = None
try:
    from .mouse import load_mouse_recordings
except ModuleNotFoundError:
    load_mouse_recordings = None
from .utils import Normalizer
from ..train.dynamics import fit_subject_dynamics_mats


def _split_list(items: List[Any], train_frac: float, val_frac: float, seed: int) -> Tuple[List[Any], List[Any], List[Any]]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test


def _slice_recording(rec: Any, slc: slice) -> Any:
    if not is_dataclass(rec):
        raise TypeError(f"Expected dataclass recording, got {type(rec)}")
    kwargs: Dict[str, Any] = {}
    if hasattr(rec, "neural"):
        kwargs["neural"] = rec.neural[slc]
    if hasattr(rec, "target"):
        kwargs["target"] = rec.target[slc]
    if hasattr(rec, "target_vel") and getattr(rec, "target_vel") is not None:
        kwargs["target_vel"] = rec.target_vel[slc]
    if hasattr(rec, "valid_mask") and getattr(rec, "valid_mask") is not None:
        kwargs["valid_mask"] = rec.valid_mask[slc]
    if hasattr(rec, "raw_target") and getattr(rec, "raw_target") is not None:
        kwargs["raw_target"] = rec.raw_target[slc]
    if hasattr(rec, "offset_steps"):
        start = 0 if slc.start is None else int(slc.start)
        kwargs["offset_steps"] = int(getattr(rec, "offset_steps")) + start
    return replace(rec, **kwargs)


def _split_recording_blocks(
    rec: Any,
    train_frac: float,
    val_frac: float,
    preprocess_cfg: Dict[str, Any],
) -> Tuple[Any, Any, Any]:
    n = rec.target.shape[0]
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = slice(0, n_train)
    val = slice(n_train, n_train + n_val)
    test = slice(n_train + n_val, n)
    tr = _slice_recording(rec, train)
    va = _slice_recording(rec, val)
    te = _slice_recording(rec, test)

    if preprocess_cfg.get("standardize", True) and hasattr(tr, "neural"):
        norm = Normalizer().fit(tr.neural)
        tr.neural = norm.transform(tr.neural).astype(np.float32)
        va.neural = norm.transform(va.neural).astype(np.float32)
        te.neural = norm.transform(te.neural).astype(np.float32)
        tr.neural_standardized = True
        va.neural_standardized = True
        te.neural_standardized = True
    return tr, va, te


def _split_recording_train_val(
    rec: Any,
    train_frac: float,
    val_frac: float,
) -> Tuple[Any, Any]:
    n = rec.target.shape[0]
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = slice(0, n_train)
    val = slice(n_train, n_train + n_val)
    tr = _slice_recording(rec, train)
    va = _slice_recording(rec, val)
    return tr, va


def _fit_neural_stats(train_recs: List[Any], eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mean = None
    m2 = None
    n = 0
    for rec in train_recs:
        if not hasattr(rec, "neural"):
            continue
        x = rec.neural
        if x.size == 0:
            continue
        if rec.valid_mask is not None:
            mask = rec.valid_mask
            if mask.any():
                x = x[mask]
        batch_n = x.shape[0]
        batch_mean = np.mean(x, axis=0)
        batch_m2 = np.sum((x - batch_mean) ** 2, axis=0)

        if mean is None:
            mean = batch_mean
            m2 = batch_m2
            n = batch_n
            continue

        if batch_mean.shape != mean.shape:
            raise ValueError(f"Neural dim mismatch: got {batch_mean.shape}, expected {mean.shape}")

        delta = batch_mean - mean
        total = n + batch_n
        mean = mean + delta * batch_n / total
        m2 = m2 + batch_m2 + (delta ** 2) * n * batch_n / total
        n = total

    if mean is None:
        raise ValueError("No neural data found to fit normalization stats.")

    var = m2 / max(n - 1, 1)
    std = np.sqrt(var) + eps
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_neural_normalization(recs: List[Any], mean: np.ndarray, std: np.ndarray) -> None:
    for rec in recs:
        if not hasattr(rec, "neural"):
            continue
        rec.neural = ((rec.neural - mean) / std).astype(np.float32)
        rec.neural_standardized = True


def _apply_local_neural_stats(recs: List[Any], eps: float = 1e-6) -> None:
    for rec in recs:
        if not hasattr(rec, "neural"):
            continue
        x = rec.neural
        if x.size == 0:
            continue
        if rec.valid_mask is not None:
            mask = rec.valid_mask
            if mask.any():
                x_use = x[mask]
            else:
                x_use = x
        else:
            x_use = x
        mean = np.mean(x_use, axis=0)
        std = np.std(x_use, axis=0) + eps
        rec.neural = ((rec.neural - mean) / std).astype(np.float32)
        rec.neural_standardized = True


def _build_falcon_splits(
    load_fn,
    dataset_cfg: Dict[str, Any],
    preprocess_cfg: Dict[str, Any],
) -> Tuple[List[Any], List[Any], List[Any]]:
    falcon_split = dataset_cfg.get("falcon_split", {})
    train_split = falcon_split.get("train_split", "held-in-calib")
    test_split = falcon_split.get("test_split", "held-out-calib")
    val_frac = float(falcon_split.get("val_frac", 0.1))
    val_frac = max(0.0, min(val_frac, 1.0))
    train_frac = max(0.0, 1.0 - val_frac)

    train_cfg = dict(dataset_cfg)
    train_cfg["split"] = train_split
    held_in_recs = load_fn(train_cfg, preprocess_cfg)

    test_cfg = dict(dataset_cfg)
    test_cfg["split"] = test_split
    test_recs = load_fn(test_cfg, preprocess_cfg)

    train_recs: List[Any] = []
    val_recs: List[Any] = []
    for rec in held_in_recs:
        tr, va = _split_recording_train_val(rec, train_frac, val_frac)
        if tr.target.shape[0] > 0:
            train_recs.append(tr)
        if va.target.shape[0] > 0:
            val_recs.append(va)

    if preprocess_cfg.get("standardize", True) and train_recs:
        mean, std = _fit_neural_stats(train_recs)
        _apply_neural_normalization(train_recs + val_recs, mean, std)
        falcon_preprocess = dataset_cfg.get("falcon_preprocess", {})
        local_x_stats = bool(falcon_preprocess.get("local_x_stats", False))
        if local_x_stats:
            _apply_local_neural_stats(test_recs)
        else:
            _apply_neural_normalization(test_recs, mean, std)

    return train_recs, val_recs, test_recs


def _fit_target_stats(train_recs: List[Any], eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    mean = None
    m2 = None
    n = 0
    for rec in train_recs:
        if not hasattr(rec, "target"):
            continue
        x = rec.target
        if x.size == 0:
            continue
        batch_n = x.shape[0]
        batch_mean = np.mean(x, axis=0)
        batch_m2 = np.sum((x - batch_mean) ** 2, axis=0)

        if mean is None:
            mean = batch_mean
            m2 = batch_m2
            n = batch_n
            continue

        if batch_mean.shape != mean.shape:
            raise ValueError(f"Target dim mismatch: got {batch_mean.shape}, expected {mean.shape}")

        delta = batch_mean - mean
        total = n + batch_n
        mean = mean + delta * batch_n / total
        m2 = m2 + batch_m2 + (delta ** 2) * n * batch_n / total
        n = total

    if mean is None:
        raise ValueError("No targets found to fit target normalization stats.")

    var = m2 / max(n - 1, 1)
    std = np.sqrt(var) + eps
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_target_normalization(recs: List[Any], mean: np.ndarray, std: np.ndarray) -> None:
    for rec in recs:
        if not hasattr(rec, "target"):
            continue
        rec.target = ((rec.target - mean) / std).astype(np.float32)
        if rec.target_vel is not None:
            rec.target_vel = (rec.target_vel / std).astype(np.float32)


def _resolve_stride_steps(
    stride_cfg: Any,
    history_steps: int,
    horizon_steps: int,
    dt: float,
) -> int:
    if isinstance(stride_cfg, str):
        if stride_cfg == "window":
            return history_steps + horizon_steps
        if stride_cfg == "history":
            return history_steps
        if stride_cfg == "horizon":
            return horizon_steps
        raise ValueError(f"Unknown window_stride '{stride_cfg}'")
    stride_sec = float(stride_cfg)
    if stride_sec <= 0:
        return 1
    return max(int(round(stride_sec / dt)), 1)


def _fit_subject_coeffs(
    train_recs: List[Any],
    cfg: Dict[str, Any],
    history_steps: int,
    horizon_steps: int,
    stride_steps: int,
    target_mode: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    subject_cfg = cfg["model"].get("subject_dynamics", {})
    if not subject_cfg.get("enabled", True):
        return {}
    ridge = float(subject_cfg.get("ridge_lambda", 1e-3))
    subject_map: Dict[str, List[np.ndarray]] = {}
    subject_b_map: Dict[str, List[np.ndarray]] = {}
    for rec in train_recs:
        if not hasattr(rec, "target") or not hasattr(rec, "dt"):
            continue
        a_mat, b_mat = fit_subject_dynamics_mats(
            rec,
            history_steps,
            horizon_steps,
            stride_steps,
            target_mode,
            cfg["model"]["dmp"],
            subject_cfg,
        )
        subject_map.setdefault(rec.subject_id, []).append(a_mat)
        subject_b_map.setdefault(rec.subject_id, []).append(b_mat)

    subject_coeffs: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for subject, coeffs in subject_map.items():
        a_mat = np.mean(coeffs, axis=0)
        b_mat = np.mean(subject_b_map[subject], axis=0)
        subject_coeffs[subject] = (a_mat.astype(np.float32), b_mat.astype(np.float32))
    return subject_coeffs


def build_datasets(cfg: Dict[str, Any]) -> Tuple[MultiDataset, MultiDataset, MultiDataset, float]:
    datasets = cfg["data"]["datasets"]
    history_sec = float(cfg["data"]["history_sec"])
    horizon_sec_raw = float(cfg["data"]["horizon_sec"])
    horizon_sec = abs(horizon_sec_raw)
    target_mode = cfg["data"].get("target_mode")
    if target_mode is None:
        target_mode = "within" if horizon_sec_raw < 0 else "future"
    stride_cfg = cfg["data"].get("window_stride", 1)

    preprocess_cfg = cfg.get("preprocess", {})
    seed = cfg["project"]["seed"]

    split_mode = cfg["data"]["split"]["mode"]
    train_frac = cfg["data"]["split"]["train_frac"]
    val_frac = cfg["data"]["split"]["val_frac"]

    train_recs: List[Any] = []
    val_recs: List[Any] = []
    test_recs: List[Any] = []
    recordings: List[Any] = []

    for name in datasets:
        if name == "mouse":
            if load_mouse_recordings is None:
                raise FileNotFoundError("mouse loader missing: src/neuroscribe/data/mouse.py")
            recordings.extend(load_mouse_recordings(cfg["data"]["mouse"], preprocess_cfg))
        elif name == "m2":
            if load_m2_recordings is None:
                raise FileNotFoundError("m2 loader missing: src/neuroscribe/data/m2.py")
            m2_cfg = cfg["data"]["m2"]
            if m2_cfg.get("split_strategy", "legacy") == "falcon":
                tr, va, te = _build_falcon_splits(load_m2_recordings, m2_cfg, preprocess_cfg)
                train_recs.extend(tr)
                val_recs.extend(va)
                test_recs.extend(te)
            else:
                recordings.extend(load_m2_recordings(m2_cfg, preprocess_cfg))
        elif name == "h1":
            if load_h1_recordings is None:
                raise FileNotFoundError("h1 loader missing: src/neuroscribe/data/h1.py")
            h1_cfg = cfg["data"]["h1"]
            if h1_cfg.get("split_strategy", "legacy") == "falcon":
                tr, va, te = _build_falcon_splits(load_h1_recordings, h1_cfg, preprocess_cfg)
                train_recs.extend(tr)
                val_recs.extend(va)
                test_recs.extend(te)
            else:
                recordings.extend(load_h1_recordings(h1_cfg, preprocess_cfg))
        elif name == "ajile12":
            if load_ajile12_recordings is None:
                raise FileNotFoundError("ajile12 loader missing: src/neuroscribe/data/ajile12.py")
            recordings.extend(load_ajile12_recordings(cfg["data"]["ajile12"], preprocess_cfg))
        elif name == "bci_iv":
            if load_bci_iv_recordings is None:
                raise FileNotFoundError("bci_iv loader missing: src/neuroscribe/data/bci_iv.py")
            recordings.extend(load_bci_iv_recordings(cfg["data"]["bci_iv"], preprocess_cfg))
        else:
            raise ValueError(f"Unknown dataset {name}")

    if recordings:
        if split_mode == "recording":
            tr, va, te = _split_list(recordings, train_frac, val_frac, seed)
            train_recs.extend(tr)
            val_recs.extend(va)
            test_recs.extend(te)
        else:
            for rec in recordings:
                if hasattr(rec, "target"):
                    tr, va, te = _split_recording_blocks(rec, train_frac, val_frac, preprocess_cfg)
                    train_recs.append(tr)
                    val_recs.append(va)
                    test_recs.append(te)
                else:
                    train_recs.append(rec)

    if not val_recs:
        val_recs = list(train_recs)
    if not test_recs:
        test_recs = list(val_recs)

    target_cfg = cfg.get("preprocess", {})
    if target_cfg.get("target_standardize", False):
        mean, std = _fit_target_stats(train_recs, eps=float(target_cfg.get("target_eps", 1e-6)))
        _apply_target_normalization(train_recs + val_recs + test_recs, mean, std)
        target_cfg["target_mean"] = mean.tolist()
        target_cfg["target_std"] = std.tolist()

    if preprocess_cfg.get("standardize", True) and split_mode == "recording":
        for rec in train_recs + val_recs + test_recs:
            if not hasattr(rec, "neural"):
                continue
            if getattr(rec, "neural_standardized", False):
                continue
            norm = Normalizer().fit(rec.neural)
            rec.neural = norm.transform(rec.neural).astype(np.float32)
            rec.neural_standardized = True

    history_steps = max(int(round(history_sec / train_recs[0].dt)), 1) if train_recs else 1
    horizon_steps = max(int(round(horizon_sec / train_recs[0].dt)), 1) if train_recs else 1
    stride_steps = _resolve_stride_steps(stride_cfg, history_steps, horizon_steps, train_recs[0].dt) if train_recs else 1

    subject_coeffs = _fit_subject_coeffs(
        train_recs,
        cfg,
        history_steps,
        horizon_steps,
        stride_steps,
        target_mode,
    )
    use_subject = cfg["model"].get("subject_dynamics", {}).get("enabled", True)

    def _make_ds(rec):
        history_steps = max(int(round(history_sec / rec.dt)), 1)
        horizon_steps = max(int(round(horizon_sec / rec.dt)), 1)
        stride_steps = _resolve_stride_steps(stride_cfg, history_steps, horizon_steps, rec.dt)
        coeff = subject_coeffs.get(rec.subject_id) if use_subject else None
        a_coeff = coeff[0] if coeff is not None else None
        b_coeff = coeff[1] if coeff is not None else None
        if hasattr(rec, "path"):
            return AJILE12WindowedDataset(
                rec,
                history_steps,
                horizon_steps,
                subject_a=a_coeff,
                subject_b=b_coeff,
                stride=stride_steps,
                target_mode=target_mode,
                feature_cfg=preprocess_cfg,
            )
        return WindowedDataset(
            rec,
            history_steps,
            horizon_steps,
            subject_a=a_coeff,
            subject_b=b_coeff,
            stride=stride_steps,
            target_mode=target_mode,
            feature_cfg=preprocess_cfg,
        )

    train_ds = MultiDataset([_make_ds(r) for r in train_recs])
    val_ds = MultiDataset([_make_ds(r) for r in val_recs])
    test_ds = MultiDataset([_make_ds(r) for r in test_recs])
    if len(val_ds) == 0:
        val_ds = train_ds
    if len(test_ds) == 0:
        test_ds = val_ds

    all_recs = train_recs + val_recs + test_recs
    dts = {round(float(rec.dt), 9) for rec in all_recs}
    if len(dts) != 1:
        raise ValueError(
            "Multiple sampling rates detected across recordings. "
            "Train/eval expects a single dt; please use a single dataset or resample to a common rate."
        )
    dt = float(dts.pop())
    return train_ds, val_ds, test_ds, dt
