"""Loss functions."""

from __future__ import annotations

import torch


def position_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def velocity_loss(pred: torch.Tensor, target: torch.Tensor, dt: float) -> torch.Tensor:
    pred_vel = (pred[:, 1:] - pred[:, :-1]) / dt
    target_vel = (target[:, 1:] - target[:, :-1]) / dt
    return torch.mean((pred_vel - target_vel) ** 2)


def smoothness_loss(pred: torch.Tensor) -> torch.Tensor:
    accel = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    return torch.mean(accel ** 2)


def endpoint_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred[:, -1] - target[:, -1]) ** 2)


def startpoint_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred[:, 0] - target[:, 0]) ** 2)
