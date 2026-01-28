"""Subject-specific linear dynamics fitting."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from ..data.base import Recording


def _build_indices(
    rec: Recording,
    history_steps: int,
    horizon_steps: int,
    stride: int,
    target_mode: str,
) -> np.ndarray:
    n = rec.target.shape[0]
    total = history_steps if target_mode == "within" else history_steps + horizon_steps
    valid = np.ones(n, dtype=bool)
    if rec.valid_mask is not None:
        valid = valid & rec.valid_mask
    ok = np.convolve(valid.astype(int), np.ones(total, dtype=int), mode="valid")
    indices = np.where(ok == total)[0]
    return indices[:: max(int(stride), 1)]


def _compute_time_basis(
    dmp_cfg: Dict[str, Any],
    dt: float,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    n_basis = int(dmp_cfg["n_basis"])
    alpha_x = float(dmp_cfg.get("alpha_x", 4.0))
    tau_mode = dmp_cfg.get("tau_mode", "fixed")
    tau = float(dmp_cfg.get("tau", 1.0))
    basis_width_scale = float(dmp_cfg.get("basis_width_scale", 1.0))
    basis_eps = float(dmp_cfg.get("basis_eps", 1.0e-3))

    if tau_mode == "horizon":
        tau_use = max(n_steps * dt, 1e-6)
    elif tau_mode in ("fixed", "const", "constant"):
        tau_use = max(tau, 1e-6)
    else:
        raise ValueError(f"Unknown tau_mode '{tau_mode}'")

    if n_basis > 1:
        t = np.linspace(0.0, n_steps * dt, n_basis)
        centers = np.exp(-(alpha_x / tau_use) * t)
        delta_t = (n_steps * dt) / float(n_basis - 1)
        sigma_t = max(delta_t / (2.0 ** 0.5), 1e-6)
        dxdt = (alpha_x / tau_use) * centers
        sigma_x = np.clip(dxdt * sigma_t, 1e-6, None)
    else:
        centers = np.array([1.0], dtype=np.float64)
        sigma_x = np.array([1.0], dtype=np.float64)
    widths = 1.0 / (2.0 * sigma_x ** 2)
    x_end = centers[-1]
    base_term = widths[0] * (x_end - centers[0]) ** 2
    if base_term > 0:
        eps = max(basis_eps, 1.0e-12)
        width_scale = (-np.log(eps)) / base_term
        widths = widths * width_scale * basis_width_scale

    x = 1.0
    x_series = []
    for _ in range(n_steps):
        x_series.append(x)
        x = x + dt * (-(alpha_x / tau_use) * x)
    x_t = np.array(x_series, dtype=np.float64)

    return x_t, (centers.astype(np.float64), widths.astype(np.float64)), tau_use


def _infer_w_and_f(
    y: np.ndarray,
    y0: np.ndarray,
    dy0: np.ndarray,
    dt: float,
    dmp_cfg: Dict[str, Any],
    ridge: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer w and f(t) from a target segment. Returns (w, f_t, g)."""
    steps, out_dim = y.shape
    if steps < 2:
        raise ValueError("Target segment must have at least 2 steps.")
    g = y[-1]

    x_t, (centers, widths), tau = _compute_time_basis(dmp_cfg, dt, steps)
    psi = np.exp(-widths[None, :] * (x_t[:, None] - centers[None, :]) ** 2)
    psi_sum = psi.sum(axis=1, keepdims=True) + 1e-6
    phi_base = (psi / psi_sum) * x_t[:, None]  # (T, K)

    v = np.zeros_like(y)
    v[0] = dy0
    v[1:] = (y[1:] - y[:-1]) / dt
    v_dot = (v[1:] - v[:-1]) / dt  # (T-1, D)

    alpha_z = float(dmp_cfg.get("alpha_z", 25.0))
    beta_z = float(dmp_cfg.get("beta_z", 6.25))
    tau2 = tau * tau
    a21 = -alpha_z * beta_z / tau2
    a22 = -alpha_z / tau
    b_g = alpha_z * beta_z / tau2
    b_f = 1.0 / tau2

    y_t = y[:-1]
    v_t = v[:-1]

    w = np.zeros((out_dim, psi.shape[1]), dtype=np.float64)
    f_hat = np.zeros((steps - 1, out_dim), dtype=np.float64)
    for d in range(out_dim):
        scale = g[d] - y0[d]
        if abs(scale) < 1e-6:
            continue
        f_target = (v_dot[:, d] - a21 * y_t[:, d] - a22 * v_t[:, d] - b_g * g[d]) / b_f
        phi = phi_base[:-1] * scale  # (T-1, K)
        lhs = phi.T @ phi + ridge * np.eye(phi.shape[1])
        rhs = phi.T @ f_target
        w_d = np.linalg.solve(lhs, rhs)
        w[d] = w_d
        f_hat[:, d] = (phi_base[:-1] @ w_d) * scale
    return w.astype(np.float32), f_hat.astype(np.float32), g.astype(np.float32)


def fit_subject_dynamics_mats(
    rec: Recording,
    history_steps: int,
    horizon_steps: int,
    stride: int,
    target_mode: str,
    dmp_cfg: Dict[str, Any],
    subject_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit continuous-time dynamics with inputs: s_dot = A s + B_g g + B_f f."""
    mode = subject_cfg.get("mode", "per_dim")
    learn = subject_cfg.get("learn", "a_only")
    ridge = float(subject_cfg.get("ridge_lambda", 1e-3))
    w_ridge = float(subject_cfg.get("w_ridge", ridge))
    enforce_structure = bool(subject_cfg.get("enforce_structure", True))
    max_abs = subject_cfg.get("max_abs", 500.0)

    indices = _build_indices(rec, history_steps, horizon_steps, stride, target_mode)
    if len(indices) == 0:
        raise ValueError("No valid windows for dynamics fitting.")

    steps = horizon_steps
    _, _, tau = _compute_time_basis(dmp_cfg, rec.dt, steps)
    alpha_z = float(dmp_cfg.get("alpha_z", 25.0))
    beta_z = float(dmp_cfg.get("beta_z", 6.25))
    tau2 = tau * tau
    a21_base = -alpha_z * beta_z / tau2
    a22_base = -alpha_z / tau
    b_g_base = alpha_z * beta_z / tau2
    b_f_base = 1.0 / tau2

    D = rec.target.shape[-1]
    x_list: List[List[np.ndarray]] = [[] for _ in range(D)]
    y_list: List[List[np.ndarray]] = [[] for _ in range(D)]

    for t0 in indices:
        t1 = t0 + history_steps
        t2 = t1 + horizon_steps
        if target_mode == "within":
            target_start = t1 - horizon_steps
            target_end = t1
        else:
            target_start = t1
            target_end = t2
        if target_end > rec.target.shape[0]:
            continue
        if target_start - 1 < 0:
            continue

        y_seg = rec.target[target_start:target_end]
        y0 = rec.target[target_start - 1]
        if rec.target_vel is not None:
            dy0 = rec.target_vel[target_start - 1]
        else:
            if target_start - 2 < 0:
                dy0 = np.zeros_like(y0)
            else:
                dy0 = (rec.target[target_start - 1] - rec.target[target_start - 2]) / rec.dt

        w, f_hat, g = _infer_w_and_f(y_seg, y0, dy0, rec.dt, dmp_cfg, w_ridge)

        v = np.zeros_like(y_seg)
        v[0] = dy0
        v[1:] = (y_seg[1:] - y_seg[:-1]) / rec.dt
        v_dot = (v[1:] - v[:-1]) / rec.dt

        y_t = y_seg[:-1]
        v_t = v[:-1]
        for d in range(D):
            X = np.stack(
                [y_t[:, d], v_t[:, d], np.full_like(y_t[:, d], g[d]), f_hat[:, d]],
                axis=1,
            )
            Y = v_dot[:, d]
            x_list[d].append(X)
            y_list[d].append(Y)

    a_mat = np.zeros((D, 2, 2), dtype=np.float32)
    b_mat = np.zeros((D, 2, 2), dtype=np.float32)
    for d in range(D):
        if not x_list[d]:
            a21 = a21_base
            a22 = a22_base
            b_g = b_g_base
            b_f = b_f_base
        else:
            X = np.concatenate(x_list[d], axis=0)
            Y = np.concatenate(y_list[d], axis=0)
            if learn == "a_only":
                A = X[:, :2]
                lhs = A.T @ A + ridge * np.eye(2)
                rhs = A.T @ Y
                a21, a22 = np.linalg.solve(lhs, rhs)
                b_g, b_f = b_g_base, b_f_base
            else:
                lhs = X.T @ X + ridge * np.eye(4)
                rhs = X.T @ Y
                a21, a22, b_g, b_f = np.linalg.solve(lhs, rhs)
                if learn == "b_only":
                    a21, a22 = a21_base, a22_base

        if enforce_structure:
            a21 = -abs(a21)
            a22 = -abs(a22)
            b_g = abs(b_g)
            b_f = abs(b_f)
        if max_abs is not None:
            a21 = np.clip(a21, -max_abs, max_abs)
            a22 = np.clip(a22, -max_abs, max_abs)
            b_g = np.clip(b_g, -max_abs, max_abs)
            b_f = np.clip(b_f, -max_abs, max_abs)

        a_mat[d, 0, 0] = 0.0
        a_mat[d, 0, 1] = 1.0
        a_mat[d, 1, 0] = a21
        a_mat[d, 1, 1] = a22

        b_mat[d, 0, 0] = 0.0
        b_mat[d, 0, 1] = 0.0
        b_mat[d, 1, 0] = b_g
        b_mat[d, 1, 1] = b_f

    if mode == "shared":
        shared_a = np.mean(a_mat, axis=0, keepdims=True)
        shared_b = np.mean(b_mat, axis=0, keepdims=True)
        a_mat = np.repeat(shared_a, D, axis=0)
        b_mat = np.repeat(shared_b, D, axis=0)

    return a_mat.astype(np.float32), b_mat.astype(np.float32)
