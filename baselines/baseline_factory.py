"""Factory and registry for benchmark models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch.nn as nn

from model import ModelConfig, create_model

from .sequence_models import RecurrentOffsetPredictor, TCNOffsetPredictor
from .transformer_models import RoPEOffsetPredictor, VanillaTransformerOffsetPredictor


@dataclass(frozen=True)
class BenchmarkModelSpec:
    name: str
    label: str
    group: str
    trainable: bool
    kind: str
    physics_mode: str = "none"
    control_mode: str = "none"
    use_validity_mask: bool = False
    uses_rope: bool = False
    description: str = ""


_SPECS: Dict[str, BenchmarkModelSpec] = {
    "cv": BenchmarkModelSpec(
        name="cv",
        label="Constant Velocity",
        group="Classical baseline",
        trainable=False,
        kind="analytic",
    ),
    "lstm": BenchmarkModelSpec(
        name="lstm",
        label="LSTM",
        group="Generic data-driven baselines",
        trainable=True,
        kind="data",
    ),
    "gru": BenchmarkModelSpec(
        name="gru",
        label="GRU",
        group="Generic data-driven baselines",
        trainable=True,
        kind="data",
    ),
    "bilstm": BenchmarkModelSpec(
        name="bilstm",
        label="BiLSTM",
        group="Generic data-driven baselines",
        trainable=True,
        kind="data",
    ),
    "tcn": BenchmarkModelSpec(
        name="tcn",
        label="TCN",
        group="Generic data-driven baselines",
        trainable=True,
        kind="data",
    ),
    "vanilla_transformer": BenchmarkModelSpec(
        name="vanilla_transformer",
        label="Vanilla Transformer",
        group="Generic data-driven baselines",
        trainable=True,
        kind="data",
    ),
    "vanilla_transformer_mask": BenchmarkModelSpec(
        name="vanilla_transformer_mask",
        label="Vanilla Transformer + Validity Mask",
        group="Method-specific ablations",
        trainable=True,
        kind="data",
        use_validity_mask=True,
    ),
    "rope_transformer_nomask": BenchmarkModelSpec(
        name="rope_transformer_nomask",
        label="RoPE Transformer w/o Validity Mask",
        group="Method-specific ablations",
        trainable=True,
        kind="data",
        uses_rope=True,
    ),
    "rope_transformer_mask_nophysics": BenchmarkModelSpec(
        name="rope_transformer_mask_nophysics",
        label="RoPE Transformer + Validity Mask, w/o Physics",
        group="Method-specific ablations",
        trainable=True,
        kind="data",
        use_validity_mask=True,
        uses_rope=True,
    ),
    "vrt_pinn_tau0": BenchmarkModelSpec(
        name="vrt_pinn_tau0",
        label="VRT-PINN tau=0 Physics",
        group="Method-specific ablations",
        trainable=True,
        kind="pinn",
        physics_mode="inference",
        control_mode="none",
        use_validity_mask=True,
        uses_rope=True,
    ),
    "vrt_pinn_controlled": BenchmarkModelSpec(
        name="vrt_pinn_controlled",
        label="Controlled VRT-PINN",
        group="Method-specific ablations",
        trainable=True,
        kind="pinn",
        physics_mode="inference",
        control_mode="anchor_hold",
        use_validity_mask=True,
        uses_rope=True,
    ),
}

BENCHMARK_MODEL_NAMES = tuple(_SPECS.keys())


def get_benchmark_spec(model_name: str) -> BenchmarkModelSpec:
    try:
        return _SPECS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported AUV_MODEL_NAME={model_name!r}; "
            f"expected one of {BENCHMARK_MODEL_NAMES}"
        ) from exc


def is_pinn_model(model_name: str) -> bool:
    return get_benchmark_spec(model_name).kind == "pinn"


def is_data_driven_model(model_name: str) -> bool:
    return get_benchmark_spec(model_name).kind == "data"


def create_benchmark_model(model_name: str, config: ModelConfig) -> nn.Module:
    """Create a trainable model for the requested benchmark entry."""
    spec = get_benchmark_spec(model_name)
    if spec.kind == "analytic":
        raise ValueError("Analytic baselines do not have trainable modules.")
    if spec.kind == "pinn":
        return create_model(config)

    kwargs = dict(
        n_features=config.n_features,
        pred_len=config.pred_len,
    )
    if model_name == "lstm":
        return RecurrentOffsetPredictor(
            **kwargs,
            hidden_dim=config.d_model,
            num_layers=2,
            dropout=config.dropout,
            cell="lstm",
            bidirectional=False,
        )
    if model_name == "gru":
        return RecurrentOffsetPredictor(
            **kwargs,
            hidden_dim=config.d_model,
            num_layers=2,
            dropout=config.dropout,
            cell="gru",
            bidirectional=False,
        )
    if model_name == "bilstm":
        return RecurrentOffsetPredictor(
            **kwargs,
            hidden_dim=config.d_model,
            num_layers=2,
            dropout=config.dropout,
            cell="lstm",
            bidirectional=True,
        )
    if model_name == "tcn":
        return TCNOffsetPredictor(
            **kwargs,
            hidden_dim=config.d_model,
            num_layers=4,
            dropout=config.dropout,
        )
    if model_name in {"vanilla_transformer", "vanilla_transformer_mask"}:
        return VanillaTransformerOffsetPredictor(
            **kwargs,
            seq_len=config.seq_len,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            use_validity_mask=spec.use_validity_mask,
        )
    if model_name in {"rope_transformer_nomask", "rope_transformer_mask_nophysics"}:
        return RoPEOffsetPredictor(
            **kwargs,
            seq_len=config.seq_len,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            use_validity_mask=spec.use_validity_mask,
        )

    raise ValueError(f"No factory implementation for {model_name!r}")
