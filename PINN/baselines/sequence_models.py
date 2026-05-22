"""Recurrent and temporal-convolution baselines.

All models output position offsets with shape [B, P, 3].  The caller is
responsible for adding the shared anchor position.  This keeps anchor usage
identical across data-driven baselines and VRT-PINN variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RecurrentOffsetPredictor(nn.Module):
    """LSTM/GRU/BiLSTM offset predictor."""

    def __init__(
        self,
        n_features: int,
        pred_len: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        cell: str = "lstm",
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if cell not in {"lstm", "gru"}:
            raise ValueError(f"cell must be 'lstm' or 'gru', got {cell!r}")
        self.pred_len = pred_len
        self.n_targets = 3
        self.bidirectional = bidirectional

        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.rnn = rnn_cls(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, pred_len * self.n_targets),
        )
        self._init_last()

    def _init_last(self) -> None:
        last_linear = None
        for module in self.decoder:
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            nn.init.normal_(last_linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(last_linear.bias)

    def forward(self, x_seq: Tensor, validity: Tensor | None = None) -> Tensor:
        x = self.input_proj(x_seq)
        out, _ = self.rnn(x)
        pooled = out[:, -1, :]
        offsets = self.decoder(pooled)
        return offsets.view(x_seq.shape[0], self.pred_len, self.n_targets)


class TemporalBlock(nn.Module):
    """Causal residual temporal convolution block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.trim = padding

    def _causal_trim(self, x: Tensor) -> Tensor:
        if self.trim == 0:
            return x
        return x[:, :, :-self.trim]

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        y = self._causal_trim(self.conv1(x))
        y = self.dropout(self.activation(self.norm1(y)))
        y = self._causal_trim(self.conv2(y))
        y = self.dropout(self.activation(self.norm2(y)))
        return residual + y


class TCNOffsetPredictor(nn.Module):
    """Causal TCN offset predictor."""

    def __init__(
        self,
        n_features: int,
        pred_len: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.n_targets = 3
        self.input_proj = nn.Conv1d(n_features, hidden_dim, kernel_size=1)
        self.blocks = nn.Sequential(*[
            TemporalBlock(
                channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=2 ** i,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * self.n_targets),
        )
        self._init_last()

    def _init_last(self) -> None:
        last_linear = None
        for module in self.decoder:
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            nn.init.normal_(last_linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(last_linear.bias)

    def forward(self, x_seq: Tensor, validity: Tensor | None = None) -> Tensor:
        x = x_seq.transpose(1, 2)
        x = self.input_proj(x)
        x = self.blocks(x)
        pooled = x[:, :, -1]
        offsets = self.decoder(pooled)
        return offsets.view(x_seq.shape[0], self.pred_len, self.n_targets)
