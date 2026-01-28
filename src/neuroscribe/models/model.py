"""Neural encoder + DMP decoder model."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .dmp import DMP
from .encoder import Encoder, PSDEncoder, SpikeEncoder


class NeuralDMPModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        cfg: Dict[str, any],
        input_dim_psd: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.n_basis = cfg["dmp"]["n_basis"]
        self.init_mode = cfg["dmp"].get("init_mode", "true")

        feature_mode = cfg.get("feature_mode") or cfg.get("preprocess", {}).get("ecog_features", "raw")
        self.feature_mode = feature_mode
        self.input_type = cfg.get("input_type", "auto")

        if self.input_type == "spike":
            if feature_mode in ("psd", "raw+psd", "raw_psd"):
                raise ValueError("input_type='spike' cannot be combined with psd feature modes.")
            self.encoder = SpikeEncoder(input_dim, cfg)
            hidden = self.encoder.hidden_size
        elif feature_mode in ("raw+psd", "raw_psd"):
            if input_dim_psd is None:
                raise ValueError("input_dim_psd is required for raw+psd mode.")
            self.encoder_raw = Encoder(input_dim, cfg)
            self.encoder_psd = PSDEncoder(input_dim_psd, cfg)
            hidden_raw = self.encoder_raw.hidden_size
            hidden_psd = self.encoder_psd.hidden_size
            fusion_hidden = int(cfg.get("fusion_hidden", hidden_raw))
            fusion_dropout = float(cfg.get("fusion_dropout", cfg.get("rnn_dropout", 0.1)))
            self.fuse_global = nn.Sequential(
                nn.Linear(hidden_raw + hidden_psd, fusion_hidden),
                nn.GELU(),
                nn.Dropout(fusion_dropout),
            )
            self.fuse_init = nn.Sequential(
                nn.Linear(hidden_raw + hidden_psd, fusion_hidden),
                nn.GELU(),
                nn.Dropout(fusion_dropout),
            )
            hidden = fusion_hidden
        elif feature_mode == "psd":
            self.encoder = PSDEncoder(input_dim, cfg)
            hidden = self.encoder.hidden_size
        else:
            self.encoder = Encoder(input_dim, cfg)
            hidden = self.encoder.hidden_size

        self.head_w = nn.Linear(hidden, out_dim * self.n_basis)
        self.head_g = nn.Linear(hidden, out_dim)
        self.head_y0 = nn.Linear(hidden, out_dim)
        self.head_dy0 = nn.Linear(hidden, out_dim)
        self.dmp = DMP(
            self.n_basis,
            alpha_x=cfg["dmp"]["alpha_x"],
            alpha_z=cfg["dmp"]["alpha_z"],
            beta_z=cfg["dmp"]["beta_z"],
            tau_mode=cfg["dmp"].get("tau_mode", "fixed"),
            tau=cfg["dmp"].get("tau", 1.0),
            basis_width_scale=cfg["dmp"].get("basis_width_scale", 1.0),
            basis_eps=cfg["dmp"].get("basis_eps", 1.0e-3),
        )
        self.a_blend = cfg.get("subject_dynamics", {}).get("blend", 1.0)

    def forward(
        self,
        x: torch.Tensor,
        y0: torch.Tensor,
        dy0: torch.Tensor,
        dt: float,
        n_steps: int,
        a_mat: Optional[torch.Tensor] = None,
        b_mat: Optional[torch.Tensor] = None,
        init_override: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        target_mode: str = "future",
        x_psd: Optional[torch.Tensor] = None,
        target_start_ratio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ratio = None
        if target_start_ratio is not None:
            ratio = float(target_start_ratio.view(-1)[0].item())
        elif target_mode == "within":
            history_steps = x.shape[1]
            denom = max(history_steps - 1, 1)
            ratio = (history_steps - n_steps - 1) / denom
            ratio = max(0.0, min(1.0, ratio))

        if self.input_type == "spike":
            z_global, z_init = self.encoder(
                x,
                target_mode=target_mode,
                horizon_steps=n_steps,
                target_start_ratio=ratio,
            )
        elif self.feature_mode in ("raw+psd", "raw_psd"):
            if x_psd is None:
                raise ValueError("x_psd is required for raw+psd mode.")
            z_raw, z_raw_init = self.encoder_raw(
                x,
                target_mode=target_mode,
                horizon_steps=n_steps,
                target_start_ratio=ratio,
            )
            z_psd, z_psd_init = self.encoder_psd(
                x_psd,
                target_mode=target_mode,
                horizon_steps=n_steps,
                target_start_ratio=ratio,
            )
            z_global = self.fuse_global(torch.cat([z_raw, z_psd], dim=-1))
            z_init = self.fuse_init(torch.cat([z_raw_init, z_psd_init], dim=-1))
        else:
            z_global, z_init = self.encoder(
                x,
                target_mode=target_mode,
                horizon_steps=n_steps,
                target_start_ratio=ratio,
            )
        w = self.head_w(z_global).view(-1, self.out_dim, self.n_basis) #* 100
        g = self.head_g(z_global)
        if init_override is not None:
            y0_use, dy0_use = init_override
        elif self.init_mode == "predict":
            y0_use = self.head_y0(z_init)
            dy0_use = self.head_dy0(z_init)
        else:
            y0_use = y0
            dy0_use = dy0
        traj, _ = self.dmp.rollout(
            y0_use,
            dy0_use,
            g,
            w,
            dt,
            n_steps,
            a_mat=a_mat,
            b_mat=b_mat,
            a_blend=self.a_blend,
        )
        return traj, g, w
