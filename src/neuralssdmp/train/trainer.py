"""Training loop."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .losses import endpoint_loss, position_loss, smoothness_loss, startpoint_loss, velocity_loss
from .metrics import compute_metrics
from ..data.utils import compute_psd_features


def get_device(cfg: Dict[str, any]) -> torch.device:
    dev = cfg.get("project", {}).get("device", "auto")
    if dev != "auto":
        return torch.device(dev)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _resolve_target_mode(cfg: Dict[str, any]) -> str:
    target_mode = cfg.get("data", {}).get("target_mode")
    if target_mode is None:
        horizon_sec = float(cfg.get("data", {}).get("horizon_sec", 0.0))
        target_mode = "within" if horizon_sec < 0 else "future"
    return target_mode


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


def _use_direct_concat_from_eval(cfg: Dict[str, any]) -> bool:
    viz_cfg = cfg.get("viz", {})
    if not viz_cfg.get("enabled", True):
        return False
    test_cfg = viz_cfg.get("test", {})
    full_cfg = test_cfg.get("full_sequence")
    if isinstance(full_cfg, dict):
        if not full_cfg.get("enabled", False):
            return False
        direct_cfg = full_cfg.get("direct_concat", {})
        return bool(direct_cfg.get("enabled", False) and direct_cfg.get("from_eval", False))
    full_enabled = viz_cfg.get("full_sequence", False) if full_cfg is None else bool(full_cfg)
    if not full_enabled:
        return False
    return bool(viz_cfg.get("direct_concat", False) and viz_cfg.get("direct_concat_from_eval", False))


def _compute_recording_metrics(
    model: nn.Module,
    dataset,
    dt: float,
    cfg: Dict[str, any],
    device: torch.device,
) -> Dict[str, float]:
    feature_mode = cfg.get("preprocess", {}).get("ecog_features", "raw")
    psd_cfg_global = cfg.get("preprocess", {}).get("psd")
    metrics_list = []
    rmse_raw_list = []
    target_cfg = cfg.get("preprocess", {})
    mean = target_cfg.get("target_mean")
    std = target_cfg.get("target_std")
    mean_arr = np.asarray(mean, dtype=np.float32) if mean is not None else None
    std_arr = np.asarray(std, dtype=np.float32) if std is not None else None

    model.eval()
    with torch.no_grad():
        for ds in dataset.datasets:
            rec = ds.recording
            n = rec.target.shape[0]
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

            pred_sum = np.zeros_like(rec.target, dtype=np.float32)
            pred_count = np.zeros_like(rec.target, dtype=np.float32)
            a_mat = ds.subject_a if getattr(ds, "subject_a", None) is not None else None
            b_mat = getattr(ds, "subject_b", None)

            for t0 in indices:
                t1 = t0 + history
                if t1 >= n:
                    break
                target_start = t1 - horizon if target_mode == "within" else t1
                if target_start - 1 < 0:
                    continue
                x = _get_neural_window(rec, t0, t1)
                x_psd = None
                psd_cfg = psd_cfg_global
                if getattr(rec, "preprocess_cfg", None) is not None:
                    psd_cfg = rec.preprocess_cfg.get("psd", psd_cfg)
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
                x_psd_t = None
                if x_psd is not None:
                    x_psd_t = torch.from_numpy(x_psd).float().unsqueeze(0).to(device)

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
                t_end = min(target_start + pred_np.shape[0], n)
                pred_sum[target_start:t_end] += pred_np[: t_end - target_start]
                pred_count[target_start:t_end] += 1

            valid = np.all(pred_count > 0, axis=-1)
            if rec.valid_mask is not None:
                valid = valid & rec.valid_mask[: valid.shape[0]]
            if not np.any(valid):
                continue

            y_true = rec.target[valid]
            y_pred = pred_sum[valid] / pred_count[valid]
            metrics_list.append(compute_metrics(y_true, y_pred, aggregate="micro"))
            if mean_arr is not None and std_arr is not None:
                y_true_raw = y_true * std_arr + mean_arr
                y_pred_raw = y_pred * std_arr + mean_arr
                rmse_raw = float(np.sqrt(np.mean((y_true_raw - y_pred_raw) ** 2)))
                rmse_raw_list.append(rmse_raw)

    if not metrics_list:
        return {"r2": float("nan"), "corr": float("nan"), "rmse": float("nan")}

    r2 = float(np.nanmean([m["r2"] for m in metrics_list]))
    corr = float(np.nanmean([m["corr"] for m in metrics_list]))
    rmse = float(np.nanmean([m["rmse"] for m in metrics_list]))
    metrics = {"r2": r2, "corr": corr, "rmse": rmse}
    if rmse_raw_list:
        metrics["rmse_raw"] = float(np.nanmean(rmse_raw_list))
    return metrics


def _compute_basis_over_time(dmp, dt: float, n_steps: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if dmp.tau_mode == "horizon":
        tau = max(n_steps * dt, 1e-6)
    elif dmp.tau_mode in ("fixed", "const", "constant"):
        tau = max(float(dmp.tau), 1e-6)
    else:
        raise ValueError(f"Unknown tau_mode '{dmp.tau_mode}'")
    centers, widths = dmp._compute_time_basis_params(dt, n_steps)
    centers = centers.to(device)
    widths = widths.to(device)
    x = torch.ones((1, 1), device=device)
    x_series = []
    for _ in range(n_steps):
        x_series.append(x)
        x = x + dt * (-(dmp.alpha_x / tau) * x)
    x_t = torch.cat(x_series, dim=0)  # (T, 1)
    psi = dmp._basis(x_t, centers=centers, widths=widths)  # (T, K)
    return x_t, psi, tau


def _infer_wg_targets(model: nn.Module, y: torch.Tensor, y0: torch.Tensor, dy0: torch.Tensor, dt: float, cfg: Dict[str, any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Infer per-window w and g targets from behavior using ridge regression."""
    batch, steps, out_dim = y.shape
    if steps < 2:
        raise ValueError("Need at least 2 steps to infer w/g targets.")
    device = y.device
    dmp = model.dmp
    with torch.no_grad():
        x_t, psi, tau = _compute_basis_over_time(dmp, dt, steps, device)
        psi = psi[: steps - 1]
        x_t = x_t[: steps - 1]
        psi_sum = psi.sum(dim=-1, keepdim=True) + 1e-6
        base_phi = (psi / psi_sum) * x_t  # (T-1, K)

        v = torch.zeros_like(y)
        v[:, 0] = dy0
        v[:, 1:] = (y[:, 1:] - y[:, :-1]) / dt

        s_t_y = y[:, :-1]
        s_t_v = v[:, :-1]
        s_next_v = v[:, 1:]
        s_dot_v = (s_next_v - s_t_v) / dt  # (B, T-1, D)

        tau2 = tau * tau
        a21 = -dmp.alpha_z * dmp.beta_z / tau2
        a22 = -dmp.alpha_z / tau
        b_g = dmp.alpha_z * dmp.beta_z / tau2
        b_f = 1.0 / tau2

        g_target = y[:, -1]  # (B, D)
        f_target = (s_dot_v - a21 * s_t_y - a22 * s_t_v - b_g * g_target.unsqueeze(1)) / b_f

        scale = g_target - y0  # (B, D)
        phi = base_phi[None, None, :, :] * scale[:, :, None, None]  # (B, D, T-1, K)

        f_bt = f_target.permute(0, 2, 1)  # (B, D, T-1)
        phi_t = phi.transpose(-2, -1)  # (B, D, K, T-1)
        lhs = phi_t @ phi  # (B, D, K, K)
        rhs = phi_t @ f_bt.unsqueeze(-1)  # (B, D, K, 1)

        ridge = float(cfg.get("loss", {}).get("wg_ridge", 1e-3))
        eye = torch.eye(lhs.shape[-1], device=device).view(1, 1, lhs.shape[-1], lhs.shape[-1])
        lhs = lhs + ridge * eye
        w_target = torch.linalg.solve(lhs, rhs).squeeze(-1)  # (B, D, K)

        mask = scale.abs() < 1e-6
        if mask.any():
            w_target = w_target.masked_fill(mask.unsqueeze(-1), 0.0)
    return w_target, g_target


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    dt: float,
    cfg: Dict[str, any],
    device: torch.device,
) -> float:
    model.train()
    losses = []
    total_steps = len(loader)
    target_mode = _resolve_target_mode(cfg)
    loss_cfg = cfg.get("loss", {})
    mode = loss_cfg.get("mode", "full")
    if mode != "full":
        unused = []
        if float(loss_cfg.get("vel_weight", 0.0)) > 0.0:
            unused.append("vel_weight")
        if float(loss_cfg.get("smooth_weight", 0.0)) > 0.0:
            unused.append("smooth_weight")
        if float(loss_cfg.get("w_weight", 0.0)) > 0.0:
            unused.append("w_weight")
        if float(loss_cfg.get("g_weight", 0.0)) > 0.0:
            unused.append("g_weight")
        if unused:
            print(f"warning: loss.mode='{mode}' ignores {', '.join(unused)}.")
    use_roll_init = cfg.get("train", {}).get("roll_init", False)
    prev_pred = None
    prev_prev = None
    for step, batch in enumerate(loader):
        batch = _to_device(batch, device)
        init_override = None
        if use_roll_init and prev_pred is not None:
            batch_size = batch["x"].shape[0]
            if prev_pred.shape[0] != batch_size:
                prev_pred = None
                prev_prev = None
            else:
                y0 = prev_pred
                if prev_prev is not None:
                    dy0 = (prev_pred - prev_prev) / dt
                else:
                    dy0 = torch.zeros_like(prev_pred)
                init_override = (y0, dy0)
        pred, g_pred, w_pred = model(
            batch["x"],
            batch["y0"],
            batch["dy0"],
            dt,
            batch["y_future"].shape[1],
            a_mat=batch.get("a_mat"),
            b_mat=batch.get("b_mat"),
            init_override=init_override,
            target_mode=target_mode,
            x_psd=batch.get("x_psd"),
            target_start_ratio=batch.get("target_start_ratio"),
        )
        if use_roll_init:
            prev_prev = prev_pred
            prev_pred = pred[:, -1].detach()

        pos = position_loss(pred, batch["y_future"])
        loss_cfg = cfg.get("loss", {})
        mode = loss_cfg.get("mode", "full")
        if mode == "pos":
            end = endpoint_loss(pred, batch["y_future"])
            start = startpoint_loss(pred, batch["y_future"])
            end_w = float(loss_cfg.get("endpoint_weight", 0.0))
            start_w = float(loss_cfg.get("startpoint_weight", 0.0))
            end_w = max(0.0, min(1.0, end_w))
            start_w = max(0.0, min(1.0, start_w))
            total_w = end_w + start_w
            if total_w > 1.0:
                end_w = end_w / total_w
                start_w = start_w / total_w
            mid_w = 1.0 - end_w - start_w
            mix = mid_w * pos + end_w * end + start_w * start
            loss = loss_cfg["pos_weight"] * mix
        elif mode == "endpoint":
            end = endpoint_loss(pred, batch["y_future"])
            loss = loss_cfg["pos_weight"] * end
        elif mode == "full":
            vel = velocity_loss(pred, batch["y_future"], dt)
            smooth = smoothness_loss(pred)
            w_weight = float(loss_cfg.get("w_weight", 0.0))
            g_weight = float(loss_cfg.get("g_weight", 0.0))
            w_loss = torch.tensor(0.0, device=device)
            g_loss = torch.tensor(0.0, device=device)
            if w_weight > 0.0 or g_weight > 0.0:
                w_target, g_target = _infer_wg_targets(
                    model,
                    batch["y_future"],
                    batch["y0"],
                    batch["dy0"],
                    dt,
                    cfg,
                )
                w_loss = torch.mean((w_pred - w_target) ** 2)
                g_loss = torch.mean((g_pred - g_target) ** 2)
            loss = (
                loss_cfg["pos_weight"] * pos
                + loss_cfg["vel_weight"] * vel
                + loss_cfg["smooth_weight"] * smooth
                + w_weight * w_loss
                + g_weight * g_loss
            )
        else:
            raise ValueError(f"Unknown loss.mode '{mode}'")

        optimizer.zero_grad()
        loss.backward()
        if cfg["train"].get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        optimizer.step()

        losses.append(loss.item())

        log_every = cfg["train"]["log_every"]
        is_log_step = log_every and (step + 1) % log_every == 0
        is_last_step = (step + 1) == total_steps
        if log_every and (is_log_step or is_last_step):
            print(f"step {step+1} loss {np.mean(losses):.4f}")

    return float(np.mean(losses))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    dt: float,
    cfg: Dict[str, any],
    device: torch.device,
) -> Tuple[float, Dict[str, float], Optional[Tuple[np.ndarray, np.ndarray]]]:
    model.eval()
    losses = []
    preds = []
    targets = []
    target_mode = _resolve_target_mode(cfg)
    use_roll_init = cfg.get("eval", {}).get("roll_init", False)
    prev_pred = None
    prev_prev = None

    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            init_override = None
            if use_roll_init and prev_pred is not None:
                batch_size = batch["x"].shape[0]
                if prev_pred.shape[0] != batch_size:
                    prev_pred = None
                    prev_prev = None
                else:
                    y0 = prev_pred
                    if prev_prev is not None:
                        dy0 = (prev_pred - prev_prev) / dt
                    else:
                        dy0 = torch.zeros_like(prev_pred)
                    init_override = (y0, dy0)
            pred, g_pred, w_pred = model(
                batch["x"],
                batch["y0"],
                batch["dy0"],
                dt,
                batch["y_future"].shape[1],
                a_mat=batch.get("a_mat"),
                b_mat=batch.get("b_mat"),
                init_override=init_override,
                target_mode=target_mode,
                x_psd=batch.get("x_psd"),
                target_start_ratio=batch.get("target_start_ratio"),
            )
            if use_roll_init:
                prev_prev = prev_pred
                prev_pred = pred[:, -1].detach()
            pos = position_loss(pred, batch["y_future"])
            loss_cfg = cfg.get("loss", {})
            mode = loss_cfg.get("mode", "full")
            if mode == "pos":
                end = endpoint_loss(pred, batch["y_future"])
                start = startpoint_loss(pred, batch["y_future"])
                end_w = float(loss_cfg.get("endpoint_weight", 0.0))
                start_w = float(loss_cfg.get("startpoint_weight", 0.0))
                end_w = max(0.0, min(1.0, end_w))
                start_w = max(0.0, min(1.0, start_w))
                total_w = end_w + start_w
                if total_w > 1.0:
                    end_w = end_w / total_w
                    start_w = start_w / total_w
                mid_w = 1.0 - end_w - start_w
                mix = mid_w * pos + end_w * end + start_w * start
                loss = loss_cfg["pos_weight"] * mix
            elif mode == "endpoint":
                end = endpoint_loss(pred, batch["y_future"])
                loss = loss_cfg["pos_weight"] * end
            elif mode == "full":
                vel = velocity_loss(pred, batch["y_future"], dt)
                smooth = smoothness_loss(pred)
                w_weight = float(loss_cfg.get("w_weight", 0.0))
                g_weight = float(loss_cfg.get("g_weight", 0.0))
                w_loss = torch.tensor(0.0, device=device)
                g_loss = torch.tensor(0.0, device=device)
                if w_weight > 0.0 or g_weight > 0.0:
                    w_target, g_target = _infer_wg_targets(
                        model,
                        batch["y_future"],
                        batch["y0"],
                        batch["dy0"],
                        dt,
                        cfg,
                    )
                    w_loss = torch.mean((w_pred - w_target) ** 2)
                    g_loss = torch.mean((g_pred - g_target) ** 2)
                loss = (
                    loss_cfg["pos_weight"] * pos
                    + loss_cfg["vel_weight"] * vel
                    + loss_cfg["smooth_weight"] * smooth
                    + w_weight * w_loss
                    + g_weight * g_loss
                )
            else:
                raise ValueError(f"Unknown loss.mode '{mode}'")
            losses.append(loss.item())
            preds.append(pred.cpu().numpy())
            targets.append(batch["y_future"].cpu().numpy())

    if not preds:
        metrics = {"r2": float("nan"), "corr": float("nan"), "rmse": float("nan")}
        return float("nan"), metrics, None
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    metrics_agg = cfg.get("eval", {}).get("metrics_agg", "micro")
    if metrics_agg == "recording":
        metrics = _compute_recording_metrics(model, loader.dataset, dt, cfg, device)
    else:
        metrics = compute_metrics(y_true, y_pred, aggregate=metrics_agg)
        target_cfg = cfg.get("preprocess", {})
        mean = target_cfg.get("target_mean")
        std = target_cfg.get("target_std")
        if mean is not None and std is not None:
            mean_arr = np.asarray(mean, dtype=np.float32)
            std_arr = np.asarray(std, dtype=np.float32)
            y_pred_raw = y_pred * std_arr + mean_arr
            y_true_raw = y_true * std_arr + mean_arr
            rmse_raw = float(np.sqrt(np.mean((y_true_raw - y_pred_raw) ** 2)))
            metrics["rmse_raw"] = rmse_raw
    sequences = None
    if _use_direct_concat_from_eval(cfg):
        sequences = (y_true, y_pred)
    return float(np.mean(losses)), metrics, sequences
