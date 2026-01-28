"""DMP rollout module."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class DMP(nn.Module):
    def __init__(
        self,
        n_basis: int,
        alpha_x: float = 4.0,
        alpha_z: float = 25.0,
        beta_z: float = 6.25,
        tau_mode: str = "fixed",
        tau: float = 1.0,
        basis_width_scale: float = 1.0,
        basis_eps: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.n_basis = n_basis
        self.alpha_x = alpha_x
        self.alpha_z = alpha_z
        self.beta_z = beta_z
        self.tau_mode = tau_mode
        self.tau = float(tau)
        self.basis_width_scale = float(basis_width_scale)
        self.basis_eps = float(basis_eps)
        centers = torch.linspace(0.0, 1.0, n_basis)
        widths = torch.ones_like(centers)
        self.register_buffer("centers", centers)
        self.register_buffer("widths", widths)

    def _compute_time_basis_params(self, dt: float, n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        if self.tau_mode == "horizon":
            tau_use = max(n_steps * dt, 1e-6)
        elif self.tau_mode in ("fixed", "const", "constant"):
            tau_use = max(self.tau, 1e-6)
        else:
            raise ValueError(f"Unknown tau_mode '{self.tau_mode}'")

        if self.n_basis > 1:
            t = torch.linspace(0.0, n_steps * dt, self.n_basis, device=self.centers.device)
            centers = torch.exp(-(self.alpha_x / tau_use) * t)
            delta_t = (n_steps * dt) / float(self.n_basis - 1)
            sigma_t = max(delta_t / (2.0 ** 0.5), 1e-6)
            dxdt = (self.alpha_x / tau_use) * centers
            sigma_x = torch.clamp(dxdt * sigma_t, min=1e-6)
        else:
            centers = torch.tensor([1.0], device=self.centers.device)
            sigma_x = torch.tensor([1.0], device=self.centers.device)
        widths = 1.0 / (2.0 * sigma_x ** 2)
        x_end = centers[-1]
        base_term = widths[0] * (x_end - centers[0]) ** 2
        if base_term > 0:
            eps = max(self.basis_eps, 1.0e-12)
            width_scale = (-torch.log(torch.tensor(eps, device=centers.device))) / base_term
            widths = widths * width_scale * self.basis_width_scale
        return centers, widths

    def _basis(self, x: torch.Tensor, centers: Optional[torch.Tensor] = None, widths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if centers is None:
            centers = self.centers
        if widths is None:
            widths = self.widths
        return torch.exp(-widths * (x - centers) ** 2)

    def rollout(
        self,
        y0: torch.Tensor,
        dy0: torch.Tensor,
        g: torch.Tensor,
        w: torch.Tensor,
        dt: float,
        n_steps: int,
        a_mat: Optional[torch.Tensor] = None,
        b_mat: Optional[torch.Tensor] = None,
        a_blend: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rollout a DMP trajectory in SSM form with state [y, v=y_dot]."""
        batch, out_dim = y0.shape
        device = y0.device

        if self.tau_mode == "horizon":
            tau = max(n_steps * dt, 1e-6)
        elif self.tau_mode in ("fixed", "const", "constant"):
            tau = max(self.tau, 1e-6)
        else:
            raise ValueError(f"Unknown tau_mode '{self.tau_mode}'")

        tau2 = tau * tau
        a11 = torch.zeros(out_dim, device=device)
        a12 = torch.ones(out_dim, device=device)
        a21 = torch.full((out_dim,), -self.alpha_z * self.beta_z / tau2, device=device)
        a22 = torch.full((out_dim,), -self.alpha_z / tau, device=device)
        a_base = torch.stack(
            [
                torch.stack([a11, a12], dim=-1),
                torch.stack([a21, a22], dim=-1),
            ],
            dim=-2,
        )  # (D, 2, 2) continuous-time base matrix

        if a_mat is None:
            a_eff = a_base
        else:
            if a_mat.dim() == 3:
                a_mat = a_mat.unsqueeze(0).expand(batch, -1, -1, -1)
            a_base = a_base.unsqueeze(0).expand(batch, -1, -1, -1)
            blend = 1.0 if a_blend is None else float(a_blend)
            blend = max(0.0, min(1.0, blend))
            a_eff = (1.0 - blend) * a_base + blend * a_mat

        if a_eff.dim() == 3:
            a_eff = a_eff.unsqueeze(0).expand(batch, -1, -1, -1)

        b_g = torch.tensor([0.0, self.alpha_z * self.beta_z / tau2], device=device)
        b_f = torch.tensor([0.0, 1.0 / tau2], device=device)
        b_base = torch.stack([b_g, b_f], dim=-1).view(1, 1, 2, 2)
        if b_mat is None:
            b_eff = b_base.expand(batch, out_dim, -1, -1)
        else:
            if b_mat.dim() == 3:
                b_mat = b_mat.unsqueeze(0).expand(batch, -1, -1, -1)
            b_eff = b_mat

        centers, widths = self._compute_time_basis_params(dt, n_steps)
        x = torch.ones((batch, 1), device=device)
        y = y0
        v = dy0
        traj = []

        for _ in range(n_steps):
            psi = self._basis(x, centers=centers, widths=widths)  # (B, K)
            psi_sum = psi.sum(dim=-1, keepdim=True) + 1e-6
            f = (psi.unsqueeze(1) * w).sum(dim=-1) / psi_sum
            f = f * x
            f = f * (g - y0)

            s = torch.stack([y, v], dim=-1)  # (B, D, 2)
            s_dot = torch.einsum("bdij,bdj->bdi", a_eff, s)
            u = torch.stack([g, f], dim=-1)  # (B, D, 2)
            s_dot = s_dot + torch.einsum("bdij,bdj->bdi", b_eff, u)
            s_next = s + dt * s_dot

            y = s_next[..., 0]
            v = s_next[..., 1]
            x = x + dt * (-(self.alpha_x / tau) * x)

            traj.append(y.unsqueeze(1))

        return torch.cat(traj, dim=1), v

    def plot_basis_time(self, out_path: str, dt: float, n_steps: int) -> None:
        """Save basis functions over time using the same phase update as rollout."""
        import os
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import numpy as np

        if n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        centers, widths = self._compute_time_basis_params(dt, n_steps)
        if self.tau_mode == "horizon":
            tau = max(n_steps * dt, 1e-6)
        elif self.tau_mode in ("fixed", "const", "constant"):
            tau = max(self.tau, 1e-6)
        else:
            raise ValueError(f"Unknown tau_mode '{self.tau_mode}'")

        x = torch.ones((1, 1), device=self.centers.device)
        x_series = []
        for _ in range(n_steps):
            x_series.append(x.item())
            x = x + dt * (-(self.alpha_x / tau) * x)

        x_t = torch.tensor(x_series, device=self.centers.device).view(-1, 1)
        psi = self._basis(x_t, centers=centers, widths=widths).detach().cpu().numpy()
        psi_sum = psi.sum(axis=-1, keepdims=True) + 1e-6
        psi_norm = psi / psi_sum

        t = np.arange(n_steps) * dt
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t, psi, alpha=0.7)
        axes[0].set_title("DMP basis functions (psi)")
        axes[0].set_ylabel("activation")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, psi_norm, alpha=0.7)
        axes[1].set_title("Normalized basis functions (psi / sum)")
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("weight")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
