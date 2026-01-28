"""Encoder backbones."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn


def _as_ratio(target_start_ratio: Optional[torch.Tensor | float]) -> Optional[float]:
    if target_start_ratio is None:
        return None
    if isinstance(target_start_ratio, torch.Tensor):
        return float(target_start_ratio.view(-1)[0].item())
    return float(target_start_ratio)


def _resolve_within_index(seq_len: int, ratio: Optional[float], horizon_steps: int) -> int:
    if seq_len <= 1:
        return 0
    if ratio is not None:
        idx = int(round(ratio * (seq_len - 1)))
    else:
        idx = seq_len - int(horizon_steps) - 1
    return int(max(0, min(seq_len - 1, idx)))


def _sinusoidal_positional_encoding(
    length: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if length <= 0:
        return torch.zeros((1, 0, dim), device=device, dtype=dtype)
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(dim, 1))
    )
    pe = torch.zeros((length, dim), device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe.unsqueeze(0)


def _make_activation(name: Optional[str]) -> Optional[nn.Module]:
    if name is None:
        return None
    name = str(name).lower()
    if name in ("none", "linear", "identity"):
        return None
    if name == "relu":
        return nn.ReLU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation '{name}'")


class _ConvBlock1d(nn.Module):
    def __init__(self, hidden: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.act(y)
        y = self.drop(y)
        return self.norm(x + y)


class _CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class _TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.res = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.act(y)
        y = self.drop(y)
        return y + self.res(x)


class _TCNStack(nn.Module):
    def __init__(self, hidden: int, num_layers: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.blocks.append(_TCNBlock(hidden, hidden, kernel_size, dilation, dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2)


class _SequenceEncoder(nn.Module):
    def __init__(self, input_dim: int, cfg: Dict[str, any], section: str) -> None:
        super().__init__()
        enc_cfg = cfg.get(section, {})
        shared_cfg = cfg.get("encoder", {})

        self.encoder_type = enc_cfg.get("encoder_type", shared_cfg.get("encoder_type", cfg.get("encoder_type", "lstm")))
        hidden = int(enc_cfg.get("hidden_size", shared_cfg.get("hidden_size", cfg.get("hidden_size", 256))))
        num_layers = int(enc_cfg.get("num_layers", shared_cfg.get("num_layers", cfg.get("num_layers", 2))))
        dropout = float(enc_cfg.get("rnn_dropout", shared_cfg.get("rnn_dropout", cfg.get("rnn_dropout", 0.1))))
        input_dropout = float(enc_cfg.get("input_dropout", shared_cfg.get("input_dropout", cfg.get("input_dropout", 0.1))))
        self.within_pool = enc_cfg.get("within_pool", shared_cfg.get("within_pool", "query_attn"))

        pool_heads = int(enc_cfg.get("pool_heads", enc_cfg.get("attn_heads", shared_cfg.get("attn_heads", 1))))
        if hidden % pool_heads != 0:
            raise ValueError(f"hidden_size ({hidden}) must be divisible by pool_heads ({pool_heads}).")

        n_heads = int(enc_cfg.get("n_heads", enc_cfg.get("attn_heads", pool_heads)))
        if self.encoder_type in ("transformer", "conv_transformer") and hidden % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden}) must be divisible by n_heads ({n_heads}).")
        ff_mult = int(enc_cfg.get("ff_mult", 4))
        conv_layers = int(enc_cfg.get("conv_layers", 2))
        conv_kernel = int(enc_cfg.get("conv_kernel", 5))
        tcn_layers = int(enc_cfg.get("tcn_layers", num_layers))
        tcn_kernel = int(enc_cfg.get("tcn_kernel", enc_cfg.get("kernel_size", 3)))

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden)
        self.input_act = _make_activation(enc_cfg.get("input_activation", "gelu"))
        self.input_drop = nn.Dropout(input_dropout)

        self.rnn = None
        self.tcn = None
        self.conv_blocks = None
        self.transformer = None

        if self.encoder_type in ("rnn", "gru", "lstm"):
            if self.encoder_type == "gru":
                self.rnn = nn.GRU(hidden, hidden, num_layers=num_layers, dropout=dropout, batch_first=True)
            elif self.encoder_type == "rnn":
                self.rnn = nn.RNN(hidden, hidden, num_layers=num_layers, dropout=dropout, batch_first=True)
            else:
                self.rnn = nn.LSTM(hidden, hidden, num_layers=num_layers, dropout=dropout, batch_first=True)
        elif self.encoder_type == "tcn":
            self.tcn = _TCNStack(hidden, tcn_layers, tcn_kernel, dropout)
        elif self.encoder_type in ("transformer", "conv_transformer"):
            if self.encoder_type == "conv_transformer":
                self.conv_blocks = nn.ModuleList(
                    [_ConvBlock1d(hidden, conv_kernel, dropout) for _ in range(conv_layers)]
                )
            layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * ff_mult,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            raise ValueError(f"Unknown encoder_type '{self.encoder_type}'")

        self.attn = nn.MultiheadAttention(hidden, pool_heads, dropout=dropout, batch_first=True)
        self.output_drop = nn.Dropout(dropout)
        self.hidden_size = hidden

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        x = self.input_norm(x)
        x = self.input_proj(x)
        if self.input_act is not None:
            x = self.input_act(x)
        x = self.input_drop(x)

        if self.rnn is not None:
            out, _ = self.rnn(x)
            return out
        if self.tcn is not None:
            return self.tcn(x)
        if self.transformer is not None:
            if self.conv_blocks is not None:
                for block in self.conv_blocks:
                    x = block(x)
            x = x + _sinusoidal_positional_encoding(x.shape[1], x.shape[2], x.device, x.dtype)
            return self.transformer(x)
        raise RuntimeError("Encoder core not initialized.")

    def _pool(
        self,
        out: torch.Tensor,
        target_mode: str,
        horizon_steps: Optional[int],
        target_start_ratio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_mode == "within":
            ratio = _as_ratio(target_start_ratio)
            if ratio is None and horizon_steps is None:
                raise ValueError("horizon_steps is required when target_mode='within'.")
            idx = _resolve_within_index(out.shape[1], ratio, int(horizon_steps or 0))
            z_init = out[:, idx]
            if self.within_pool == "query_attn":
                query = out[:, idx : idx + 1, :]
                pooled, _ = self.attn(query, out, out, need_weights=False)
                pooled = pooled[:, 0]
            else:
                raise ValueError(f"Unknown within_pool '{self.within_pool}'")
        else:
            pooled = out[:, -1]
            z_init = out[:, -1]
        return pooled, z_init

    def forward(
        self,
        x: torch.Tensor,
        target_mode: str = "future",
        horizon_steps: Optional[int] = None,
        target_start_ratio: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._encode(x)
        pooled, z_init = self._pool(out, target_mode, horizon_steps, target_start_ratio)
        return self.output_drop(pooled), self.output_drop(z_init)


class Encoder(_SequenceEncoder):
    """Encoder for raw neural inputs (ECoG or spikes)."""

    def __init__(self, input_dim: int, cfg: Dict[str, any]) -> None:
        super().__init__(input_dim, cfg, section="raw_encoder")


class PSDEncoder(_SequenceEncoder):
    """Encoder for PSD features."""

    def __init__(self, input_dim: int, cfg: Dict[str, any]) -> None:
        super().__init__(input_dim, cfg, section="psd_encoder")
        self.band_mixer = None
        self._band_count = None
        self._band_channels = None
        psd_cfg = cfg.get("psd", {})
        bands = psd_cfg.get("bands")
        enc_cfg = cfg.get("psd_encoder", {})
        if bands:
            band_count = len(bands)
            if band_count > 0 and input_dim % band_count == 0:
                band_hidden = int(enc_cfg.get("band_hidden", band_count))
                self.band_mixer = nn.Sequential(
                    nn.Linear(band_count, band_hidden),
                    nn.GELU(),
                    nn.Linear(band_hidden, band_count),
                )
                self._band_count = band_count
                self._band_channels = int(input_dim // band_count)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.band_mixer is None or self._band_count is None or self._band_channels is None:
            return x
        b, t, f = x.shape
        x = x.reshape(b, t, self._band_channels, self._band_count)
        x = self.band_mixer(x)
        return x.reshape(b, t, f)


class SpikeEncoder(_SequenceEncoder):
    """Encoder for spike-based inputs."""

    def __init__(self, input_dim: int, cfg: Dict[str, any]) -> None:
        super().__init__(input_dim, cfg, section="spike_encoder")
