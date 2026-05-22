"""Transformer benchmark variants for AUV trajectory prediction."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from model import AUVTransformerPredictor, build_causal_mask


class LearnablePositionalEncoding(nn.Module):
    """Learnable absolute positional encoding."""

    def __init__(self, d_model: int, max_seq_len: int) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(max_seq_len, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos[: x.shape[1]].unsqueeze(0)


class VanillaTransformerOffsetPredictor(nn.Module):
    """Standard Transformer encoder baseline with optional validity key mask."""

    def __init__(
        self,
        n_features: int,
        pred_len: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        use_validity_mask: bool = False,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.n_targets = 3
        self.d_model = d_model
        self.use_validity_mask = use_validity_mask

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos = LearnablePositionalEncoding(d_model, max_seq_len=seq_len + 64)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.pool_proj = nn.Linear(d_model, d_model, bias=False)
        self.pool_pos_bias = nn.Parameter(torch.zeros(seq_len + 64))
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len * self.n_targets),
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

    def forward(self, x_seq: Tensor, validity: Optional[Tensor] = None) -> Tensor:
        B, T, _ = x_seq.shape
        x = self.pos(self.input_proj(x_seq))
        causal = build_causal_mask(T, x.device)
        key_padding_mask = None
        if self.use_validity_mask and validity is not None:
            key_padding_mask = validity.squeeze(-1) <= 0.0
        x = self.encoder(x, mask=causal, src_key_padding_mask=key_padding_mask)
        x = self.final_norm(x)

        q = self.pool_query / math.sqrt(self.d_model)
        scores = (self.pool_proj(x) * q).sum(dim=-1)
        scores = scores + self.pool_pos_bias[:T]
        if self.use_validity_mask and validity is not None:
            key_valid = validity.squeeze(-1)
            scores = scores + (1.0 - key_valid) * (-1e9)
            attn = torch.softmax(scores, dim=-1) * key_valid
            denom = attn.sum(dim=-1, keepdim=True)
            attn = torch.where(
                denom > 0,
                attn / denom.clamp_min(1e-12),
                torch.zeros_like(attn),
            )
        else:
            attn = torch.softmax(scores, dim=-1)
        pooled = (x * attn.unsqueeze(-1)).sum(dim=1)
        offsets = self.decoder(pooled)
        return offsets.view(B, self.pred_len, self.n_targets)


class RoPEOffsetPredictor(nn.Module):
    """RoPE Transformer offset predictor with optional validity masking."""

    def __init__(
        self,
        n_features: int,
        pred_len: int,
        seq_len: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        use_validity_mask: bool = True,
    ) -> None:
        super().__init__()
        self.use_validity_mask = use_validity_mask
        self.predictor = AUVTransformerPredictor(
            n_features=n_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            pred_len=pred_len,
            max_seq_len=seq_len + 64,
        )

    def forward(self, x_seq: Tensor, validity: Optional[Tensor] = None) -> Tensor:
        if self.use_validity_mask:
            if validity is None:
                validity = torch.ones(
                    x_seq.shape[0], x_seq.shape[1], 1,
                    device=x_seq.device,
                    dtype=x_seq.dtype,
                )
            effective_validity = validity
        else:
            effective_validity = torch.ones(
                x_seq.shape[0], x_seq.shape[1], 1,
                device=x_seq.device,
                dtype=x_seq.dtype,
            )
        return self.predictor(x_seq, effective_validity)
