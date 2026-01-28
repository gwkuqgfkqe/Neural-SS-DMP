"""Training entrypoint."""

from __future__ import annotations

import os
import sys
if __package__ is None or __package__ == "":  # pragma: no cover
    # Allow running this file directly without -m.
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    __package__ = "neuroscribe"

import argparse
import json
import time
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import apply_overrides, load_yaml
from .data.datamodule import build_datasets
from .data.utils import compute_psd_features
from .models import NeuralDMPModel
from .train.trainer import evaluate, get_device, train_one_epoch
from .train.visualizer import plot_mean_window, plot_predictions
from .utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuroScribe training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[], help="Override key=value")
    return parser.parse_args()


def _safe_torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _get_target_stats(cfg):
    target_cfg = cfg.get("preprocess", {})
    mean = target_cfg.get("target_mean")
    std = target_cfg.get("target_std")
    if mean is None or std is None:
        return None, None
    return np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32)


def _denorm(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return arr * std + mean


def _infer_input_type(cfg) -> str:
    input_type = cfg.get("model", {}).get("input_type", "auto")
    if input_type != "auto":
        return input_type
    datasets = cfg.get("data", {}).get("datasets", [])
    if "m2" in datasets:
        signal_mode = cfg.get("data", {}).get("m2", {}).get("use_signal", "spikes")
        if signal_mode == "spikes":
            return "spike"
    if "h1" in datasets:
        signal_mode = cfg.get("data", {}).get("h1", {}).get("use_signal", "spikes")
        if signal_mode == "spikes":
            return "spike"
    return "raw"


def _infer_feature_mode_for_viz(cfg) -> str:
    """Match the actual model input expectations for visualization.

    For spike datasets (m2/h1), visualization must not attempt PSD feature extraction even if
    preprocess.ecog_features is set to "psd", because the spike model expects raw spike counts.
    """
    datasets = cfg.get("data", {}).get("datasets", [])
    input_type = _infer_input_type(cfg)
    if input_type == "spike" or "m2" in datasets or "h1" in datasets:
        return "raw"
    return cfg.get("preprocess", {}).get("ecog_features", "raw")


def _normalize_splits(value, default):
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value.lower()]
    return [str(item).lower() for item in value]


def _resolve_full_sequence_cfg(cfg):
    viz_cfg = cfg.get("viz", {})
    test_cfg = viz_cfg.get("test", {})
    full_cfg = test_cfg.get("full_sequence")
    if isinstance(full_cfg, dict):
        enabled = full_cfg.get("enabled", True)
        max_recordings = full_cfg.get("max_recordings")
        smooth_cfg = full_cfg.get("smooth", {})
        smooth_enabled = smooth_cfg.get("enabled", False)
        smooth_window_sec = smooth_cfg.get("window_sec", 0.05)
    else:
        enabled = viz_cfg.get("full_sequence", False) if full_cfg is None else bool(full_cfg)
        max_recordings = None
        smooth_enabled = False
        smooth_window_sec = 0.05
    return {
        "enabled": bool(enabled),
        "max_recordings": int(max_recordings) if max_recordings is not None else None,
        "smooth": {"enabled": bool(smooth_enabled), "window_sec": float(smooth_window_sec)},
    }


def _resolve_per_epoch_full_sequence_cfg(cfg):
    viz_cfg = cfg.get("viz", {})
    per_epoch_cfg = viz_cfg.get("per_epoch", {})
    full_cfg = per_epoch_cfg.get("full_sequence")
    if isinstance(full_cfg, dict):
        enabled = full_cfg.get("enabled", False)
        splits = _normalize_splits(full_cfg.get("splits"), ["val"])
        max_recordings = full_cfg.get("max_recordings")
        smooth_cfg = full_cfg.get("smooth", {})
        smooth_enabled = smooth_cfg.get("enabled", False)
        smooth_window_sec = smooth_cfg.get("window_sec", 0.05)
    else:
        enabled = False
        splits = ["val"]
        max_recordings = None
        smooth_enabled = False
        smooth_window_sec = 0.05
    return {
        "enabled": bool(enabled),
        "splits": splits,
        "max_recordings": int(max_recordings) if max_recordings is not None else None,
        "smooth": {"enabled": bool(smooth_enabled), "window_sec": float(smooth_window_sec)},
    }


def _full_sequence_tag(split_name: str, epoch: int | None) -> str:
    if epoch is None:
        return split_name
    return f"{split_name} epoch {epoch+1}"


def _smooth_series(y: np.ndarray, dt: float, window_sec: float) -> np.ndarray:
    if y is None:
        return y
    window_sec = float(window_sec)
    if window_sec <= 0 or y.shape[0] < 3:
        return y
    win = int(round(window_sec / dt))
    if win < 2:
        return y
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32)
    out = np.empty_like(y, dtype=np.float32)
    for d in range(y.shape[1]):
        vec = y[:, d].astype(np.float32)
        mask = np.isfinite(vec).astype(np.float32)
        if mask.sum() == 0:
            out[:, d] = vec
            continue
        num = np.convolve(np.where(mask > 0, vec, 0.0), kernel, mode="same")
        den = np.convolve(mask, kernel, mode="same")
        out[:, d] = np.where(den > 0, num / den, np.nan)
    return out


def _subsample_indices_by_min_step(indices: np.ndarray, stride_steps: int) -> np.ndarray:
    if indices.size == 0:
        return indices
    stride_steps = int(stride_steps)
    if stride_steps <= 1 or indices.size == 1:
        return indices
    kept = [int(indices[0])]
    last = kept[0]
    for t in indices[1:]:
        t = int(t)
        if t - last >= stride_steps:
            kept.append(t)
            last = t
    return np.asarray(kept, dtype=np.int64)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.set)

    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg["project"]["output_dir"], f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(out_dir, ".mpl"))

    train_ds, val_ds, test_ds, dt = build_datasets(cfg)
    target_mean, target_std = _get_target_stats(cfg)

    train_shuffle = True
    if cfg.get("train", {}).get("roll_init", False):
        train_shuffle = False
        print("roll_init enabled: disabling shuffle for train loader.")
    val_shuffle = False
    test_shuffle = False
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=train_shuffle,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=val_shuffle,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=test_shuffle,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
    )

    sample = next(iter(train_loader))
    input_dim = sample["x"].shape[-1]
    input_dim_psd = None
    if "x_psd" in sample:
        input_dim_psd = sample["x_psd"].shape[-1]
    out_dim = sample["y_future"].shape[-1]

    device = get_device(cfg)
    model_cfg = dict(cfg["model"])
    model_cfg["feature_mode"] = cfg.get("preprocess", {}).get("ecog_features", "raw")
    model_cfg["input_type"] = _infer_input_type(cfg)
    if model_cfg["input_type"] == "spike":
        model_cfg["feature_mode"] = "raw"
    model_cfg["psd"] = cfg.get("preprocess", {}).get("psd", {})
    model = NeuralDMPModel(input_dim, out_dim, model_cfg, input_dim_psd=input_dim_psd).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = None
    schedule = cfg["train"].get("lr_schedule")
    if schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=cfg["train"]["num_epochs"],
            eta_min=cfg["train"].get("lr_min", 0.0),
        )

    best_val = float("inf")
    for epoch in range(cfg["train"]["num_epochs"]):
        tr_loss = train_one_epoch(model, train_loader, optim, dt, cfg, device)
        va_loss, va_metrics, _ = evaluate(model, val_loader, dt, cfg, device)
        print(f"epoch {epoch+1} train {tr_loss:.4f} val {va_loss:.4f} metrics {va_metrics}")

        if va_loss < best_val:
            best_val = va_loss

        viz_cfg = cfg.get("viz", {})
        save_every = int(viz_cfg.get("save_every_epochs", 1))
        if viz_cfg.get("enabled", True) and save_every > 0 and (epoch + 1) % save_every == 0:
            per_epoch_cfg = viz_cfg.get("per_epoch", {})
            samples_cfg = per_epoch_cfg.get("samples", {})
            samples_enabled = samples_cfg.get("enabled", True)
            if samples_enabled:
                default_splits = ["val", "train"] if viz_cfg.get("include_train", False) else ["val"]
                splits = _normalize_splits(samples_cfg.get("splits"), default_splits)
                num_examples = samples_cfg.get("num_examples", viz_cfg.get("num_examples", 6))
                if "val" in splits:
                    _save_viz(
                        model,
                        val_loader,
                        dt,
                        out_dir,
                        epoch,
                        device,
                        cfg,
                        split_name="val",
                        target_mean=target_mean,
                        target_std=target_std,
                        num_examples=num_examples,
                    )
                if "train" in splits:
                    _save_viz(
                        model,
                        train_loader,
                        dt,
                        out_dir,
                        epoch,
                        device,
                        cfg,
                        split_name="train",
                        target_mean=target_mean,
                        target_std=target_std,
                        num_examples=num_examples,
                    )

            tail_cfg = per_epoch_cfg.get("train_tail", {})
            tail_enabled = tail_cfg.get("enabled", viz_cfg.get("train_tail", True))
            if tail_enabled:
                _save_train_tail_sequence_avg(
                    model,
                    train_ds,
                    dt,
                    out_dir,
                    epoch,
                    device,
                    cfg,
                    target_mean=target_mean,
                    target_std=target_std,
                )

            mean_cfg = per_epoch_cfg.get("mean_window", {})
            mean_enabled = mean_cfg.get("enabled", viz_cfg.get("mean_window", False))
            if mean_enabled:
                default_splits = ["val", "train"] if viz_cfg.get("include_train", False) else ["val"]
                splits = _normalize_splits(mean_cfg.get("splits"), default_splits)
                if "val" in splits:
                    _save_mean_window(
                        model,
                        val_loader,
                        dt,
                        out_dir,
                        epoch,
                        device,
                        cfg,
                        split_name="val",
                        target_mean=target_mean,
                        target_std=target_std,
                    )
                if "train" in splits:
                    _save_mean_window(
                        model,
                        train_loader,
                        dt,
                        out_dir,
                        epoch,
                        device,
                        cfg,
                        split_name="train",
                        target_mean=target_mean,
                        target_std=target_std,
                    )
            full_epoch_cfg = _resolve_per_epoch_full_sequence_cfg(cfg)
            if full_epoch_cfg["enabled"]:
                splits = full_epoch_cfg["splits"]
                if "val" in splits:
                    _save_full_sequence(
                        model,
                        val_ds,
                        dt,
                        out_dir,
                        epoch,
                        device,
                        cfg,
                        split_name="val",
                        target_mean=target_mean,
                        target_std=target_std,
                        full_seq_cfg=full_epoch_cfg,
                    )
                if "train" in splits:
                    _save_full_sequence(
                        model,
                        train_ds,
                        dt,
                        out_dir,
                        epoch,
                        device,
                        cfg,
                        split_name="train",
                        target_mean=target_mean,
                        target_std=target_std,
                        full_seq_cfg=full_epoch_cfg,
                    )
        if scheduler is not None:
            scheduler.step()

    last_path = os.path.join(out_dir, "last.pt")
    torch.save(model.state_dict(), last_path)
    model.load_state_dict(_safe_torch_load(last_path, map_location=device))
    te_loss, te_metrics, te_sequences = evaluate(model, test_loader, dt, cfg, device)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"val_best_loss": best_val, "test_loss": te_loss, **te_metrics}, f, indent=2)

    print(f"test loss {te_loss:.4f} metrics {te_metrics}")
    viz_cfg = cfg.get("viz", {})
    test_cfg = viz_cfg.get("test", {})
    test_mean_window = test_cfg.get("mean_window")
    if test_mean_window is None:
        test_mean_window = viz_cfg.get("mean_window", False)
    if viz_cfg.get("enabled", True) and test_mean_window:
        _save_mean_window(
            model,
            test_loader,
            dt,
            out_dir,
            None,
            device,
            cfg,
            split_name="test",
            target_mean=target_mean,
            target_std=target_std,
        )
    full_seq_cfg = _resolve_full_sequence_cfg(cfg)
    if viz_cfg.get("enabled", True) and full_seq_cfg["enabled"]:
        _save_full_sequence(
            model,
            test_ds,
            dt,
            out_dir,
            None,
            device,
            cfg,
            split_name="test",
            target_mean=target_mean,
            target_std=target_std,
            full_seq_cfg=full_seq_cfg,
        )


def _save_viz(
    model,
    loader,
    dt,
    out_dir,
    epoch,
    device,
    cfg,
    split_name: str,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
    num_examples: int | None = None,
) -> None:
    model.eval()
    n = num_examples if num_examples is not None else cfg.get("viz", {}).get("num_examples", 6)
    preds = []
    targets = []
    raw_targets = []
    all_have_raw = True
    target_mode = _resolve_target_mode(cfg)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            pred, _, _ = model(
                batch["x"],
                batch["y0"],
                batch["dy0"],
                dt,
                batch["y_future"].shape[1],
                a_mat=batch.get("a_mat"),
                b_mat=batch.get("b_mat"),
                target_mode=target_mode,
                x_psd=batch.get("x_psd"),
                target_start_ratio=batch.get("target_start_ratio"),
            )
            preds.append(pred.cpu().numpy())
            targets.append(batch["y_future"].cpu().numpy())
            if "y_future_raw" in batch:
                raw_targets.append(batch["y_future_raw"].cpu().numpy())
            else:
                all_have_raw = False
            if sum(p.shape[0] for p in preds) >= n:
                break

    y_pred = np.concatenate(preds, axis=0)[:n]
    y_true = np.concatenate(targets, axis=0)[:n]
    y_raw = None
    if all_have_raw and raw_targets:
        y_raw = np.concatenate(raw_targets, axis=0)[:n]
    if y_raw is not None and target_mean is not None and target_std is not None:
        y_raw = (y_raw - target_mean) / target_std

    for i in range(y_true.shape[0]):
        out_path = os.path.join(out_dir, "plots", f"{split_name}_epoch_{epoch+1}_sample_{i}.png")
        plot_predictions(
            y_true[i],
            y_pred[i],
            out_path,
            title=f"{split_name} epoch {epoch+1} sample {i}",
            y_raw=y_raw[i] if y_raw is not None else None,
        )


def _save_mean_window(
    model,
    loader,
    dt,
    out_dir,
    epoch,
    device,
    cfg,
    split_name: str,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
) -> None:
    model.eval()
    pred_sum = None
    target_sum = None
    raw_sum = None
    count = 0
    all_have_raw = True
    target_mode = _resolve_target_mode(cfg)
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            pred, _, _ = model(
                batch["x"],
                batch["y0"],
                batch["dy0"],
                dt,
                batch["y_future"].shape[1],
                a_mat=batch.get("a_mat"),
                b_mat=batch.get("b_mat"),
                target_mode=target_mode,
                x_psd=batch.get("x_psd"),
                target_start_ratio=batch.get("target_start_ratio"),
            )
            pred_np = pred.cpu().numpy()
            target_np = batch["y_future"].cpu().numpy()
            if pred_sum is None:
                pred_sum = pred_np.sum(axis=0)
                target_sum = target_np.sum(axis=0)
            else:
                pred_sum += pred_np.sum(axis=0)
                target_sum += target_np.sum(axis=0)
            if "y_future_raw" in batch:
                raw_np = batch["y_future_raw"].cpu().numpy()
                raw_sum = raw_np.sum(axis=0) if raw_sum is None else raw_sum + raw_np.sum(axis=0)
            else:
                all_have_raw = False
            count += pred_np.shape[0]

    if count == 0 or pred_sum is None or target_sum is None:
        return

    y_pred = pred_sum / count
    y_true = target_sum / count
    y_raw = raw_sum / count if all_have_raw and raw_sum is not None else None
    if y_raw is not None and target_mean is not None and target_std is not None:
        y_raw = (y_raw - target_mean) / target_std

    if epoch is None:
        out_name = f"{split_name}_mean_window.png"
        title = f"{split_name} mean window"
    else:
        out_name = f"{split_name}_epoch_{epoch+1}_mean_window.png"
        title = f"{split_name} epoch {epoch+1} mean window"
    out_path = os.path.join(out_dir, "plots", out_name)
    plot_mean_window(y_true, y_pred, out_path, title=title, y_raw=y_raw)

def _get_neural_window(rec, t0: int, t1: int) -> np.ndarray:
    if hasattr(rec, "get_neural_window"):
        return rec.get_neural_window(t0, t1)
    return rec.neural[t0:t1]


def _get_dy0(rec, t: int, dt: float) -> np.ndarray:
    if rec.target_vel is not None:
        return rec.target_vel[t]
    if t - 1 < 0:
        return np.zeros_like(rec.target[t])
    return (rec.target[t] - rec.target[t - 1]) / dt


def _resolve_stride(value, history, horizon) -> int:
    if isinstance(value, str):
        if value == "window":
            return history + horizon
        if value == "history":
            return history
        if value == "horizon":
            return horizon
        raise ValueError(f"Unknown full_sequence_stride '{value}'")
    return max(int(value), 1)


def _resolve_target_mode(cfg) -> str:
    target_mode = cfg.get("data", {}).get("target_mode")
    if target_mode is None:
        horizon_sec = float(cfg.get("data", {}).get("horizon_sec", 0.0))
        target_mode = "within" if horizon_sec < 0 else "future"
    return target_mode


def _resolve_target_start(t1: int, horizon: int, target_mode: str) -> int:
    if target_mode == "within":
        return t1 - horizon
    return t1


def _build_full_indices(rec, history, horizon, stride, target_mode: str) -> np.ndarray:
    n = rec.target.shape[0]
    total = history if target_mode == "within" else history + horizon
    valid = np.ones(n, dtype=bool)
    if rec.valid_mask is not None:
        valid = valid & rec.valid_mask
    ok = np.convolve(valid.astype(int), np.ones(total, dtype=int), mode="valid")
    indices = np.where(ok == total)[0]
    return indices[::stride]


def _save_train_tail_sequence_avg(
    model,
    train_ds,
    dt,
    out_dir,
    epoch,
    device,
    cfg,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
) -> None:
    """Save a multi-window averaged prediction over the last fraction of the train sequence(s)."""
    model.eval()
    viz_cfg = cfg.get("viz", {})
    per_epoch_cfg = viz_cfg.get("per_epoch", {})
    tail_cfg = per_epoch_cfg.get("train_tail", {})
    frac = float(tail_cfg.get("fraction", viz_cfg.get("train_tail_fraction", 0.125)))
    frac = max(1.0e-6, min(frac, 1.0))
    feature_mode = _infer_feature_mode_for_viz(cfg)
    psd_cfg_global = cfg.get("preprocess", {}).get("psd")

    max_recordings = tail_cfg.get("max_recordings")
    datasets = list(train_ds.datasets)
    if max_recordings is not None:
        datasets = datasets[: max(int(max_recordings), 0)]

    for ds_idx, ds in enumerate(datasets):
        rec = ds.recording
        n = rec.target.shape[0]
        if n < 2:
            continue
        history = ds.history_steps
        horizon = ds.horizon_steps
        target_mode = getattr(ds, "target_mode", "future")
        indices = ds.indices
        if len(indices) == 0:
            continue

        tail_len = max(int(round(n * frac)), 1)
        tail_start = max(n - tail_len, 0)

        ratio = None
        if target_mode == "within":
            denom = max(history - 1, 1)
            ratio = (history - horizon - 1) / denom
            ratio = max(0.0, min(1.0, ratio))

        pred_sum = np.zeros_like(rec.target, dtype=np.float32)
        pred_count = np.zeros_like(rec.target, dtype=np.float32)
        a_mat = ds.subject_a if getattr(ds, "subject_a", None) is not None else None
        b_mat = getattr(ds, "subject_b", None)

        psd_cfg = psd_cfg_global
        if getattr(rec, "preprocess_cfg", None) is not None:
            psd_cfg = rec.preprocess_cfg.get("psd", psd_cfg)

        with torch.no_grad():
            for t0 in indices:
                t0 = int(t0)
                t1 = t0 + history
                if t1 >= n:
                    break
                target_start = _resolve_target_start(t1, horizon, target_mode)
                if target_start - 1 < 0:
                    continue
                if target_start < tail_start:
                    continue
                x = _get_neural_window(rec, t0, t1)
                x_psd = None
                if feature_mode == "psd":
                    x = compute_psd_features(x, fs=1.0 / dt, cfg=psd_cfg)
                elif feature_mode in ("raw+psd", "raw_psd"):
                    x_psd = compute_psd_features(x, fs=1.0 / dt, cfg=psd_cfg)

                y0_np = rec.target[target_start - 1]
                dy0_np = _get_dy0(rec, target_start - 1, dt)
                x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
                y0_t = torch.from_numpy(y0_np).float().unsqueeze(0).to(device)
                dy0_t = torch.from_numpy(dy0_np).float().unsqueeze(0).to(device)
                a_t = torch.from_numpy(a_mat).float().to(device) if a_mat is not None else None
                b_t = torch.from_numpy(b_mat).float().to(device) if b_mat is not None else None
                x_psd_t = torch.from_numpy(x_psd).float().unsqueeze(0).to(device) if x_psd is not None else None

                pred, _, _ = model(
                    x_t,
                    y0_t,
                    dy0_t,
                    dt,
                    horizon,
                    a_mat=a_t,
                    b_mat=b_t,
                    target_mode=target_mode,
                    x_psd=x_psd_t,
                    target_start_ratio=torch.tensor(ratio, device=device) if ratio is not None else None,
                )
                pred_np = pred[0].detach().cpu().numpy()
                if not np.all(np.isfinite(pred_np)):
                    continue
                t2 = min(target_start + pred_np.shape[0], n)
                chunk = pred_np[: t2 - target_start].astype(np.float32)
                pred_sum[target_start:t2] += chunk
                pred_count[target_start:t2] += 1.0

        mask = pred_count > 0
        if not np.any(mask[tail_start:]):
            continue
        pred_mean = np.full_like(rec.target, np.nan, dtype=np.float32)
        pred_mean[mask] = pred_sum[mask] / pred_count[mask]

        y_true = rec.target[tail_start:]
        y_pred = pred_mean[tail_start:]
        smooth_cfg = tail_cfg.get("smooth", {})
        if smooth_cfg.get("enabled", False):
            y_pred = _smooth_series(y_pred, dt, smooth_cfg.get("window_sec", 0.05))
        y_raw = getattr(rec, "raw_target", None)
        if y_raw is not None:
            y_raw = y_raw[tail_start:]
        if y_raw is not None and target_mean is not None and target_std is not None:
            y_raw = (y_raw - target_mean) / target_std

        out_path = os.path.join(out_dir, "plots", f"train_tail_avg_epoch_{epoch+1}_{ds_idx}.png")
        plot_predictions(
            y_true,
            y_pred,
            out_path,
            title=f"train tail avg epoch {epoch+1} ds={ds_idx} (last {frac:.3g})",
            y_raw=y_raw,
        )


def _save_full_sequence_viz(
    model,
    ds_bundle,
    dt,
    out_dir,
    device,
    cfg,
    split_name: str = "test",
    epoch: int | None = None,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
    full_seq_cfg: dict | None = None,
) -> None:
    model.eval()
    if full_seq_cfg is None:
        full_seq_cfg = _resolve_full_sequence_cfg(cfg)
    modes = ["teacher"]
    feature_mode = _infer_feature_mode_for_viz(cfg)
    psd_cfg_global = cfg.get("preprocess", {}).get("psd")

    datasets = list(ds_bundle.datasets)
    max_recordings = full_seq_cfg.get("max_recordings")
    if max_recordings is not None:
        datasets = datasets[: max(int(max_recordings), 0)]

    for ds_idx, ds in enumerate(datasets):
        rec = ds.recording
        n = rec.target.shape[0]
        history = ds.history_steps
        horizon = ds.horizon_steps
        target_mode = getattr(ds, "target_mode", "future")
        stride_steps = 1
        indices = np.asarray(getattr(ds, "indices", []), dtype=np.int64)
        if indices.size == 0:
            continue
        indices = _subsample_indices_by_min_step(indices, stride_steps)

        ratio = None
        if target_mode == "within":
            denom = max(history - 1, 1)
            ratio = (history - horizon - 1) / denom
            ratio = max(0.0, min(1.0, ratio))

        for run_mode in modes:
            pred_sum = np.zeros_like(rec.target, dtype=np.float32)
            pred_count = np.zeros_like(rec.target, dtype=np.float32)
            a_mat = ds.subject_a
            b_mat = getattr(ds, "subject_b", None)
            psd_cfg = psd_cfg_global
            if getattr(rec, "preprocess_cfg", None) is not None:
                psd_cfg = rec.preprocess_cfg.get("psd", psd_cfg)

            with torch.no_grad():
                for t0 in indices:
                    t1 = t0 + history
                    target_start = _resolve_target_start(t1, horizon, target_mode)
                    if t1 >= n:
                        break
                    x = _get_neural_window(rec, t0, t1)
                    x_psd = None
                    if feature_mode == "psd":
                        x = compute_psd_features(x, fs=1.0 / dt, cfg=psd_cfg)
                    elif feature_mode in ("raw+psd", "raw_psd"):
                        x_psd = compute_psd_features(x, fs=1.0 / dt, cfg=psd_cfg)

                    x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
                    y0_np = rec.target[target_start - 1]
                    dy0_np = _get_dy0(rec, target_start - 1, dt)
                    y0_t = torch.from_numpy(y0_np).float().unsqueeze(0).to(device)
                    dy0_t = torch.from_numpy(dy0_np).float().unsqueeze(0).to(device)
                    a_t = None
                    if a_mat is not None:
                        a_t = torch.from_numpy(a_mat).float().to(device)
                    b_t = None
                    if b_mat is not None:
                        b_t = torch.from_numpy(b_mat).float().to(device)

                    pred, _, _ = model(
                        x_t,
                        y0_t,
                        dy0_t,
                        dt,
                        horizon,
                        a_mat=a_t,
                        b_mat=b_t,
                        target_mode=target_mode,
                        x_psd=torch.from_numpy(x_psd).float().unsqueeze(0).to(device) if x_psd is not None else None,
                        target_start_ratio=torch.tensor(ratio, device=device) if ratio is not None else None,
                    )
                    pred_np = pred[0].detach().cpu().numpy()
                    if not np.all(np.isfinite(pred_np)):
                        continue
                    t2 = min(target_start + pred_np.shape[0], n)
                    pred_sum[target_start:t2] += pred_np[: t2 - target_start]
                    pred_count[target_start:t2] += 1

            pred_avg = np.full_like(rec.target, np.nan, dtype=np.float32)
            mask = pred_count > 0
            pred_avg[mask] = pred_sum[mask] / pred_count[mask]
            time_mask = np.any(mask, axis=1) if mask.ndim == 2 else mask
            if not np.any(time_mask):
                continue
            start = int(np.argmax(time_mask))
            end = int(time_mask.shape[0] - np.argmax(time_mask[::-1]))
            y_true = rec.target[start:end]
            y_pred = pred_avg[start:end]
            smooth_cfg = full_seq_cfg.get("smooth", {})
            if smooth_cfg.get("enabled", False):
                y_pred = _smooth_series(y_pred, dt, smooth_cfg.get("window_sec", 0.05))
            y_raw = getattr(rec, "raw_target", None)
            if y_raw is not None:
                y_raw = y_raw[start:end]
            if y_raw is not None and target_mean is not None and target_std is not None:
                y_raw = (y_raw - target_mean) / target_std
            tag = _full_sequence_tag(split_name, epoch)
            out_path = os.path.join(out_dir, "plots", f"{split_name}_full_seq_{ds_idx}_{run_mode}.png")
            if epoch is not None:
                out_path = os.path.join(
                    out_dir,
                    "plots",
                    f"{split_name}_full_seq_epoch_{epoch+1}_{ds_idx}_{run_mode}.png",
                )
            plot_predictions(
                y_true,
                y_pred,
                out_path,
                title=f"{tag} full seq {ds_idx} ({run_mode})",
                y_raw=y_raw,
            )


def _save_full_sequence_direct(
    model,
    ds_bundle,
    dt,
    out_dir,
    device,
    cfg,
    split_name: str = "test",
    epoch: int | None = None,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
    full_seq_cfg: dict | None = None,
) -> None:
    model.eval()
    if full_seq_cfg is None:
        full_seq_cfg = _resolve_full_sequence_cfg(cfg)
    modes = ["teacher"]
    feature_mode = _infer_feature_mode_for_viz(cfg)
    psd_cfg_global = cfg.get("preprocess", {}).get("psd")

    datasets = list(ds_bundle.datasets)
    max_recordings = full_seq_cfg.get("max_recordings")
    if max_recordings is not None:
        datasets = datasets[: max(int(max_recordings), 0)]

    for ds_idx, ds in enumerate(datasets):
        rec = ds.recording
        history = ds.history_steps
        horizon = ds.horizon_steps
        target_mode = getattr(ds, "target_mode", "future")
        indices = ds.indices
        if len(indices) == 0:
            continue
        ratio = None
        if target_mode == "within":
            denom = max(history - 1, 1)
            ratio = (history - horizon - 1) / denom
            ratio = max(0.0, min(1.0, ratio))

        for run_mode in modes:
            parts = []
            true_parts = []
            raw_parts = []
            prev_y = None
            prev_prev = None
            psd_cfg = psd_cfg_global
            if getattr(rec, "preprocess_cfg", None) is not None:
                psd_cfg = rec.preprocess_cfg.get("psd", psd_cfg)
            with torch.no_grad():
                for t0 in indices:
                    t1 = t0 + history
                    if t1 >= rec.target.shape[0]:
                        break
                    x = _get_neural_window(rec, t0, t1)
                    x_psd = None
                    if feature_mode == "psd":
                        x = compute_psd_features(x, fs=1.0 / dt, cfg=psd_cfg)
                    elif feature_mode in ("raw+psd", "raw_psd"):
                        x_psd = compute_psd_features(x, fs=1.0 / dt, cfg=psd_cfg)
                    target_start = _resolve_target_start(t1, horizon, target_mode)
                    init_override = None
                    if run_mode == "closed_loop" and prev_y is not None:
                        y0 = prev_y
                        if prev_prev is not None:
                            dy0 = (prev_y - prev_prev) / dt
                        else:
                            dy0 = torch.zeros_like(prev_y)
                        init_override = (y0, dy0)

                    x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)
                    y0_np = rec.target[target_start - 1]
                    dy0_np = _get_dy0(rec, target_start - 1, dt)
                    y0_t = torch.from_numpy(y0_np).float().unsqueeze(0).to(device)
                    dy0_t = torch.from_numpy(dy0_np).float().unsqueeze(0).to(device)
                    a_t = None
                    if getattr(ds, "subject_a", None) is not None:
                        a_t = torch.from_numpy(ds.subject_a).float().to(device)
                    b_t = None
                    if getattr(ds, "subject_b", None) is not None:
                        b_t = torch.from_numpy(ds.subject_b).float().to(device)
                    pred, _, _ = model(
                        x_t,
                        y0_t,
                        dy0_t,
                        dt,
                        horizon,
                        a_mat=a_t,
                        b_mat=b_t,
                        init_override=init_override,
                        target_mode=target_mode,
                        x_psd=torch.from_numpy(x_psd).float().unsqueeze(0).to(device) if x_psd is not None else None,
                        target_start_ratio=torch.tensor(ratio, device=device) if ratio is not None else None,
                    )
                    pred_np = pred[0].detach().cpu().numpy()
                    parts.append(pred_np)
                    t_end = target_start + pred_np.shape[0]
                    true_parts.append(rec.target[target_start:t_end])
                    if getattr(rec, "raw_target", None) is not None:
                        raw_parts.append(rec.raw_target[target_start:t_end])
                    if run_mode == "closed_loop":
                        prev_prev = prev_y
                        prev_y = pred[:, -1].detach()

            if not parts:
                continue
            y_pred = np.concatenate(parts, axis=0)
            y_true = np.concatenate(true_parts, axis=0)
            smooth_cfg = full_seq_cfg.get("smooth", {})
            if smooth_cfg.get("enabled", False):
                y_pred = _smooth_series(y_pred, dt, smooth_cfg.get("window_sec", 0.05))
            y_raw = np.concatenate(raw_parts, axis=0) if raw_parts else None
            if y_raw is not None and target_mean is not None and target_std is not None:
                y_raw = (y_raw - target_mean) / target_std
            tag = _full_sequence_tag(split_name, epoch)
            out_path = os.path.join(out_dir, "plots", f"{split_name}_full_seq_direct_{ds_idx}_{run_mode}.png")
            if epoch is not None:
                out_path = os.path.join(
                    out_dir,
                    "plots",
                    f"{split_name}_full_seq_direct_epoch_{epoch+1}_{ds_idx}_{run_mode}.png",
                )
            plot_predictions(
                y_true,
                y_pred,
                out_path,
                title=f"{tag} full seq direct {ds_idx} ({run_mode})",
                y_raw=y_raw,
            )


def _save_full_sequence_direct_from_eval(
    te_sequences,
    out_dir,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
) -> None:
    if te_sequences is None:
        return
    y_true, y_pred = te_sequences
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    out_path = os.path.join(out_dir, "plots", "test_full_seq_direct_eval.png")
    plot_predictions(y_true_flat, y_pred_flat, out_path, title="test full seq direct (eval)")


def _save_full_sequence(
    model,
    ds_bundle,
    dt,
    out_dir,
    epoch,
    device,
    cfg,
    split_name: str,
    target_mean: np.ndarray | None = None,
    target_std: np.ndarray | None = None,
    full_seq_cfg: dict | None = None,
) -> None:
    if full_seq_cfg is None:
        full_seq_cfg = _resolve_full_sequence_cfg(cfg)
    _save_full_sequence_viz(
        model,
        ds_bundle,
        dt,
        out_dir,
        device,
        cfg,
        split_name=split_name,
        epoch=epoch,
        target_mean=target_mean,
        target_std=target_std,
        full_seq_cfg=full_seq_cfg,
    )


if __name__ == "__main__":
    main()
