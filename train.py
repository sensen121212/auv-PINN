# =============================================================================
# train.py  —  Dynamics-Informed PINN Training Pipeline (v2.0)
# =============================================================================
"""
Production-grade training pipeline for AUV trajectory prediction with
Fossen dynamics-informed regularisation and unified uncertainty weighting.

v2.0 Breaking Changes (aligned with model.py v2.0):
    - ConfidenceGate abolished: no confidence-weighted loss modulation.
    - Three-term adaptive loss: data + kinematics + Fossen dynamics.
    - Dynamics loss module has learnable damping and control parameters
      that require their own optimiser parameter group.
    - All uncertainty is managed by σ_data, σ_phy, σ_dyn in log-variance
      space (strict MLE formulation).

Mathematical Foundation:
    L = 1/(2σ²_data)·MSE + 1/(2σ²_phy)·L_trap + 1/(2σ²_dyn)·L_fossen
      + log(σ_data) + log(σ_phy) + log(σ_dyn)

References:
    [1] Kendall et al., CVPR 2018 (homoscedastic uncertainty).
    [2] Fossen, "Handbook of Marine Craft Hydrodynamics", Wiley, 2011.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import AUVDataPipeline, DatasetConfig
from baselines import (
    create_benchmark_model,
    get_benchmark_spec,
    is_data_driven_model,
    is_pinn_model,
)
from model import (
    AdaptiveRobustLoss,
    FossenDynamicsLoss,
    KinematicPhysicsLoss,
    ModelConfig,
    RobustPINN,
    create_loss_modules,
    create_model,
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def configure_training_file_logging(cfg: Config) -> Path:
    """Attach a per-run file handler for training logs.

    The console logger is configured at import time.  This function adds a
    UTF-8 file handler after runtime configuration is known, so ablation runs
    can be separated by their experiment settings.
    """
    log_dir = cfg.paths.project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    spec = get_benchmark_spec(cfg.MODEL_NAME)
    if spec.kind == 'pinn':
        log_physics_mode = spec.physics_mode
        log_control_mode = spec.control_mode
    else:
        log_physics_mode = 'none'
        log_control_mode = 'none'

    exp_suffix = (
        f"{cfg.MODEL_NAME}_"
        f"p{cfg.PRED_LEN}"
        f"_anchor_{cfg.ANCHOR_POS_SOURCE}"
        f"_phys_{log_physics_mode}"
        f"_ctrl_{log_control_mode}"
        f"_deg_{cfg.DEGRADATION_LEVEL}"
        f"_s{cfg.WINDOW_STRIDE}"
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{exp_suffix}_{timestamp}.log"

    root_logger = logging.getLogger()
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_path


# =============================================================================
# Training History Container
# =============================================================================

@dataclass
class TrainingHistory:
    """Container for training metrics with type-safe access.

    Attributes:
        train_loss: Total training loss per epoch.
        train_data: Data loss component per epoch.
        train_phys: Kinematic physics loss per epoch.
        train_dyn: Fossen dynamics loss per epoch.
        train_dyn_residual: Fossen residual term before damping prior.
        train_dyn_prior: Damping prior term inside dynamics loss.
        val_loss: Total validation loss per epoch.
        val_data: Validation data loss per epoch.
        val_phys: Validation kinematic loss per epoch.
        val_dyn: Validation dynamics loss per epoch.
        val_dyn_residual: Validation Fossen residual term before prior.
        val_dyn_prior: Validation damping prior term.
        sigma_data: Learned data uncertainty per epoch.
        sigma_phy: Learned kinematic uncertainty per epoch.
        sigma_dyn: Learned dynamics uncertainty per epoch.
        weight_data: Effective data weight per epoch.
        weight_phy: Effective kinematic weight per epoch.
        weight_dyn: Effective dynamics weight per epoch.
        learning_rates: Learning rate per epoch.
    """
    train_loss: List[float] = field(default_factory=list)
    train_data: List[float] = field(default_factory=list)
    train_phys: List[float] = field(default_factory=list)
    train_dyn: List[float] = field(default_factory=list)
    train_dyn_residual: List[float] = field(default_factory=list)
    train_dyn_prior: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_data: List[float] = field(default_factory=list)
    val_phys: List[float] = field(default_factory=list)
    val_dyn: List[float] = field(default_factory=list)
    val_dyn_residual: List[float] = field(default_factory=list)
    val_dyn_prior: List[float] = field(default_factory=list)
    sigma_data: List[float] = field(default_factory=list)
    sigma_phy: List[float] = field(default_factory=list)
    sigma_dyn: List[float] = field(default_factory=list)
    weight_data: List[float] = field(default_factory=list)
    weight_phy: List[float] = field(default_factory=list)
    weight_dyn: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert to dictionary for serialization."""
        return {
            'train_loss': self.train_loss,
            'train_data': self.train_data,
            'train_phys': self.train_phys,
            'train_dyn': self.train_dyn,
            'train_dyn_residual': self.train_dyn_residual,
            'train_dyn_prior': self.train_dyn_prior,
            'val_loss': self.val_loss,
            'val_data': self.val_data,
            'val_phys': self.val_phys,
            'val_dyn': self.val_dyn,
            'val_dyn_residual': self.val_dyn_residual,
            'val_dyn_prior': self.val_dyn_prior,
            'sigma_data': self.sigma_data,
            'sigma_phy': self.sigma_phy,
            'sigma_dyn': self.sigma_dyn,
            'weight_data': self.weight_data,
            'weight_phy': self.weight_phy,
            'weight_dyn': self.weight_dyn,
            'learning_rates': self.learning_rates,
        }


# =============================================================================
# Epoch Metrics Container
# =============================================================================

@dataclass
class EpochMetrics:
    """Aggregated metrics for a single epoch."""
    loss_total: float = 0.0
    loss_data: float = 0.0
    loss_physics: float = 0.0
    loss_dynamics: float = 0.0
    loss_dynamics_residual: float = 0.0
    loss_dynamics_prior: float = 0.0
    n_batches: int = 0

    def update(self, log_dict: Dict[str, float]) -> None:
        """Update metrics from loss module output."""
        self.loss_total += log_dict['loss_total']
        self.loss_data += log_dict['loss_data_raw']
        self.loss_physics += log_dict['loss_physics_raw']
        self.loss_dynamics += log_dict['loss_dynamics_raw']
        self.loss_dynamics_residual += log_dict.get('loss_dynamics_residual', 0.0)
        self.loss_dynamics_prior += log_dict.get('loss_dynamics_prior', 0.0)
        self.n_batches += 1

    def average(self) -> Tuple[float, float, float, float, float, float]:
        """Compute epoch averages.

        Returns:
            (avg_total, avg_data, avg_physics, avg_dynamics,
             avg_dynamics_residual, avg_dynamics_prior)
        """
        n = max(self.n_batches, 1)
        return (
            self.loss_total / n,
            self.loss_data / n,
            self.loss_physics / n,
            self.loss_dynamics / n,
            self.loss_dynamics_residual / n,
            self.loss_dynamics_prior / n,
        )


def _add_dynamics_diagnostics(
    log_dict: Dict[str, float],
    dynamics_loss_fn: FossenDynamicsLoss,
) -> None:
    """Attach residual/prior split from the most recent dynamics call."""
    residual = getattr(dynamics_loss_fn, 'last_residual_loss', None)
    prior = getattr(dynamics_loss_fn, 'last_prior_loss', None)
    if residual is not None:
        log_dict['loss_dynamics_residual'] = float(residual.detach().item())
    if prior is not None:
        log_dict['loss_dynamics_prior'] = float(prior.detach().item())


def _select_dynamics_inputs(
    pred_pos: Tensor,
    physics_mode: str,
    control_mode: str,
    anchor_thrust: Tensor,
    target_thrust: Tensor,
    target_body_vel: Tensor,
    target_attitude: Tensor,
    anchor_attitude: Tensor,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """Select Fossen forcing/attitude inputs using train/eval-consistent rules."""
    if physics_mode == 'supervised' or control_mode == 'future_truth':
        return target_thrust, target_body_vel, target_attitude

    dyn_attitude = anchor_attitude.unsqueeze(1).expand(-1, pred_pos.shape[1], -1)
    if control_mode == 'anchor_hold':
        dyn_thrust = anchor_thrust.unsqueeze(1).expand(-1, pred_pos.shape[1], -1)
    elif control_mode == 'none':
        dyn_thrust = None
    else:
        raise ValueError(f"Unsupported control_mode: {control_mode}")

    return dyn_thrust, None, dyn_attitude


def _uses_controlled_residual(physics_mode: str, control_mode: str) -> bool:
    """Whether dynamics residual receives nonzero control tensors."""
    return physics_mode != 'none' and control_mode in ('anchor_hold', 'future_truth')


def _effective_physics_control_modes(cfg: Config, model_name: str) -> Tuple[str, str]:
    """Resolve model-specific physics/control settings for fair benchmarks."""
    spec = get_benchmark_spec(model_name)
    if spec.kind == 'pinn':
        return spec.physics_mode, spec.control_mode
    return 'none', 'none'


def _benchmark_checkpoint_paths(cfg: Config, model_name: str) -> Tuple[str, str]:
    """Return best/last checkpoint paths for a benchmark model."""
    suffix = (
        f"p{cfg.PRED_LEN}_anchor_{cfg.ANCHOR_POS_SOURCE}"
        f"_deg_{cfg.DEGRADATION_LEVEL}_{model_name}"
    )
    return (
        str(cfg.SAVE_DIR / f"best_model_{suffix}.pth"),
        str(cfg.SAVE_DIR / f"last_model_{suffix}.pth"),
    )


def _data_loss(pred_pos: Tensor, target_pos: Tensor) -> Tensor:
    """NaN-safe MSE for position prediction."""
    valid_mask = ~torch.isnan(target_pos)
    if valid_mask.any():
        return F.mse_loss(pred_pos[valid_mask], target_pos[valid_mask], reduction='mean')
    return torch.zeros(1, device=pred_pos.device, dtype=pred_pos.dtype)


def _forward_offset_model(model: nn.Module, x_seq: Tensor, validity: Tensor, last_pos: Tensor) -> Tensor:
    """Run an offset predictor and add the shared anchor position."""
    offsets = model(x_seq, validity)
    return last_pos.unsqueeze(1) + offsets


def train_one_epoch_data_only(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip_norm: float = 5.0,
) -> float:
    """Train a data-driven baseline for one epoch using only position MSE."""
    model.train()
    total = 0.0
    n_batches = 0
    for batch in loader:
        (
            x_seq, validity, target_pos, _last_vel, last_pos, *_rest
        ) = batch
        x_seq = x_seq.to(device, non_blocking=True)
        validity = validity.to(device, non_blocking=True)
        target_pos = target_pos.to(device, non_blocking=True)
        last_pos = last_pos.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred_pos = _forward_offset_model(model, x_seq, validity, last_pos)
        if torch.isnan(pred_pos).any():
            logger.warning("NaN in data-driven model output, skipping batch")
            continue
        loss = _data_loss(pred_pos, target_pos)
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/Inf data loss detected, skipping batch")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()
        total += float(loss.detach().item())
        n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def validate_data_only(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Validate a data-driven baseline using position MSE."""
    model.eval()
    total = 0.0
    n_batches = 0
    for batch in loader:
        (
            x_seq, validity, target_pos, _last_vel, last_pos, *_rest
        ) = batch
        x_seq = x_seq.to(device, non_blocking=True)
        validity = validity.to(device, non_blocking=True)
        target_pos = target_pos.to(device, non_blocking=True)
        last_pos = last_pos.to(device, non_blocking=True)
        pred_pos = _forward_offset_model(model, x_seq, validity, last_pos)
        total += float(_data_loss(pred_pos, target_pos).detach().item())
        n_batches += 1
    return total / max(n_batches, 1)


def save_data_only_checkpoint(
    path: str,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    val_loss: float,
    history: Dict[str, List[float]],
    config: Dict[str, Any],
) -> None:
    """Save a benchmark checkpoint for data-only baselines."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_state': model.state_dict(),  # compatibility with old baseline loader
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'history': history,
        'config': config,
    }, path)


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(
    model: RobustPINN,
    kinematic_loss_fn: KinematicPhysicsLoss,
    dynamics_loss_fn: FossenDynamicsLoss,
    adaptive_loss_fn: AdaptiveRobustLoss,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    dt: float,
    device: torch.device,
    grad_clip_norm: float = 5.0,
    anchor_pos_source: str = 'last_valid',
    physics_mode: str = 'inference',
    control_mode: str = 'none',
) -> Tuple[float, float, float, float, float, float]:
    """Execute one training epoch with three-term physics loss.

    Args:
        model: RobustPINN v2.0 model.
        kinematic_loss_fn: Trapezoidal kinematics residual module.
        dynamics_loss_fn: Fossen dynamics residual module.
        adaptive_loss_fn: Homoscedastic uncertainty loss module.
        loader: Training data loader.
        optimizer: Optimizer with model + loss parameter groups.
        dt: Time step interval.
        device: Computation device.
        grad_clip_norm: Maximum gradient norm for clipping.
        anchor_pos_source: Boundary anchor mode. When 'zero', position
            boundary terms are omitted from physics losses because last_pos is
            a decoder anchor, not a physical boundary measurement.

    Returns:
        Tuple of (avg_loss, avg_data_loss, avg_physics_loss,
        avg_dynamics_loss, avg_dynamics_residual, avg_dynamics_prior).
    """
    model.train()
    adaptive_loss_fn.train()
    dynamics_loss_fn.train()

    metrics = EpochMetrics()

    for batch in loader:
        (
            x_seq, validity, target_pos, last_vel, last_pos, anchor_thrust,
            target_thrust, target_vel, target_body_vel, target_attitude,
            _anchor_valid, _anchor_lag, anchor_attitude
        ) = batch

        x_seq = x_seq.to(device, non_blocking=True)
        validity = validity.to(device, non_blocking=True)
        target_pos = target_pos.to(device, non_blocking=True)
        last_vel = last_vel.to(device, non_blocking=True)
        last_pos = last_pos.to(device, non_blocking=True)
        anchor_thrust = anchor_thrust.to(device, non_blocking=True)
        target_thrust = target_thrust.to(device, non_blocking=True)
        target_vel = target_vel.to(device, non_blocking=True)
        target_body_vel = target_body_vel.to(device, non_blocking=True)
        target_attitude = target_attitude.to(device, non_blocking=True)
        anchor_attitude = anchor_attitude.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass (v2.0: no confidence output).
        pred_pos = model(x_seq, validity, last_pos)

        # NaN guard on model output (e.g. from degenerate attention).
        if torch.isnan(pred_pos).any():
            logger.warning("NaN in model output, skipping batch")
            continue

        if physics_mode == 'none':
            loss_kinematic = torch.zeros(1, device=device, dtype=pred_pos.dtype)
            loss_dynamics = torch.zeros(1, device=device, dtype=pred_pos.dtype)
        else:
            physics_last_pos = None if anchor_pos_source == 'zero' else last_pos
            if physics_mode == 'supervised':
                kin_target_vel = target_vel
            elif physics_mode == 'inference':
                kin_target_vel = None
            else:
                raise ValueError(f"Unsupported physics_mode: {physics_mode}")

            dyn_thrust, dyn_target_vel, dyn_target_attitude = _select_dynamics_inputs(
                pred_pos=pred_pos,
                physics_mode=physics_mode,
                control_mode=control_mode,
                anchor_thrust=anchor_thrust,
                target_thrust=target_thrust,
                target_body_vel=target_body_vel,
                target_attitude=target_attitude,
                anchor_attitude=anchor_attitude,
            )

            loss_kinematic = kinematic_loss_fn(
                pred_pos, last_vel, dt,
                target_vel=kin_target_vel,
                last_pos=physics_last_pos,
            )
            loss_dynamics = dynamics_loss_fn(
                pred_pos, last_vel, dt,
                thrust_data=dyn_thrust,
                target_vel=dyn_target_vel,
                target_attitude=dyn_target_attitude,
                last_pos=physics_last_pos,
            )

        # Unified adaptive loss (three-term MLE weighting).
        loss_total, log_dict = adaptive_loss_fn(
            pred_pos, target_pos, loss_kinematic, loss_dynamics
        )
        _add_dynamics_diagnostics(log_dict, dynamics_loss_fn)

        # NaN/Inf guard — skip batch before backward.
        if torch.isnan(loss_total) or torch.isinf(loss_total):
            logger.warning("NaN/Inf loss detected, skipping batch")
            continue

        loss_total.backward()

        # Gradient clipping across all learnable parameters.
        all_params = (
            list(model.parameters())
            + list(adaptive_loss_fn.parameters())
            + list(dynamics_loss_fn.parameters())
        )
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=grad_clip_norm)

        optimizer.step()

        metrics.update(log_dict)

    return metrics.average()


@torch.no_grad()
def validate(
    model: RobustPINN,
    kinematic_loss_fn: KinematicPhysicsLoss,
    dynamics_loss_fn: FossenDynamicsLoss,
    adaptive_loss_fn: AdaptiveRobustLoss,
    loader: DataLoader,
    dt: float,
    device: torch.device,
    anchor_pos_source: str = 'last_valid',
    physics_mode: str = 'inference',
    control_mode: str = 'none',
) -> Tuple[float, float, float, float, float, float]:
    """Validate model on held-out data.

    Args:
        model: RobustPINN v2.0 model.
        kinematic_loss_fn: Trapezoidal kinematics loss module.
        dynamics_loss_fn: Fossen dynamics loss module.
        adaptive_loss_fn: Homoscedastic uncertainty loss module.
        loader: Validation data loader.
        dt: Time step interval.
        device: Computation device.
        anchor_pos_source: Boundary anchor mode. When 'zero', position
            boundary terms are omitted from physics losses.

    Returns:
        Tuple of (avg_loss, avg_data_loss, avg_physics_loss,
        avg_dynamics_loss, avg_dynamics_residual, avg_dynamics_prior).
    """
    model.eval()
    adaptive_loss_fn.eval()
    dynamics_loss_fn.eval()

    metrics = EpochMetrics()

    for batch in loader:
        (
            x_seq, validity, target_pos, last_vel, last_pos, anchor_thrust,
            target_thrust, target_vel, target_body_vel, target_attitude,
            _anchor_valid, _anchor_lag, anchor_attitude
        ) = batch

        x_seq = x_seq.to(device, non_blocking=True)
        validity = validity.to(device, non_blocking=True)
        target_pos = target_pos.to(device, non_blocking=True)
        last_vel = last_vel.to(device, non_blocking=True)
        last_pos = last_pos.to(device, non_blocking=True)
        anchor_thrust = anchor_thrust.to(device, non_blocking=True)
        target_thrust = target_thrust.to(device, non_blocking=True)
        target_vel = target_vel.to(device, non_blocking=True)
        target_body_vel = target_body_vel.to(device, non_blocking=True)
        target_attitude = target_attitude.to(device, non_blocking=True)
        anchor_attitude = anchor_attitude.to(device, non_blocking=True)

        pred_pos = model(x_seq, validity, last_pos)
        if physics_mode == 'none':
            loss_kinematic = torch.zeros(1, device=device, dtype=pred_pos.dtype)
            loss_dynamics = torch.zeros(1, device=device, dtype=pred_pos.dtype)
        else:
            physics_last_pos = None if anchor_pos_source == 'zero' else last_pos
            if physics_mode == 'supervised':
                kin_target_vel = target_vel
            elif physics_mode == 'inference':
                kin_target_vel = None
            else:
                raise ValueError(f"Unsupported physics_mode: {physics_mode}")

            dyn_thrust, dyn_target_vel, dyn_target_attitude = _select_dynamics_inputs(
                pred_pos=pred_pos,
                physics_mode=physics_mode,
                control_mode=control_mode,
                anchor_thrust=anchor_thrust,
                target_thrust=target_thrust,
                target_body_vel=target_body_vel,
                target_attitude=target_attitude,
                anchor_attitude=anchor_attitude,
            )

            loss_kinematic = kinematic_loss_fn(
                pred_pos, last_vel, dt,
                target_vel=kin_target_vel,
                last_pos=physics_last_pos,
            )
            loss_dynamics = dynamics_loss_fn(
                pred_pos, last_vel, dt,
                thrust_data=dyn_thrust,
                target_vel=dyn_target_vel,
                target_attitude=dyn_target_attitude,
                last_pos=physics_last_pos,
            )

        loss_total, log_dict = adaptive_loss_fn(
            pred_pos, target_pos, loss_kinematic, loss_dynamics
        )
        _add_dynamics_diagnostics(log_dict, dynamics_loss_fn)

        metrics.update(log_dict)

    return metrics.average()


# =============================================================================
# Visualization
# =============================================================================

def plot_training_curves(history: TrainingHistory, save_dir: str) -> None:
    """Generate publication-quality training visualization.

    Creates a 2×3 subplot figure:
        Row 1: Total loss | Data vs Physics vs Dynamics | LR schedule
        Row 2: σ evolution | Effective weights | Loss ratios

    Args:
        history: Training history container.
        save_dir: Directory to save figure.
    """
    epochs = range(1, len(history.train_loss) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Dynamics-Informed PINN Training with Homoscedastic Uncertainty (v2.0)',
        fontsize=14, fontweight='bold'
    )

    # (a) Total loss convergence.
    ax = axes[0, 0]
    ax.plot(epochs, history.train_loss, label='Train', color='#2E86AB', linewidth=2)
    ax.plot(epochs, history.val_loss, label='Val', color='#E94F37',
            linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Total Loss', fontsize=11)
    ax.set_title('(a) Total Loss Convergence', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # (b) Loss component decomposition.
    ax = axes[0, 1]
    ax.plot(epochs, history.train_data, label='$\\mathcal{L}_{data}$',
            color='#1B998B', linewidth=2)
    ax.plot(epochs, history.train_phys, label='$\\mathcal{L}_{kin}$',
            color='#FF6B35', linewidth=2)
    ax.plot(epochs, history.train_dyn, label='$\\mathcal{L}_{dyn}$ (Fossen)',
            color='#6B2D5C', linewidth=2, linestyle='-.')
    if history.train_dyn_residual:
        ax.plot(epochs, history.train_dyn_residual,
                label='$\\mathcal{L}_{dyn,res}$',
                color='#8E44AD', linewidth=1.5, linestyle=':')
    if history.train_dyn_prior:
        ax.plot(epochs, history.train_dyn_prior,
                label='$\\mathcal{L}_{dyn,prior}$',
                color='#7D3C98', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss Component', fontsize=11)
    ax.set_title('(b) Loss Decomposition (Data / Kinematics / Dynamics)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # (c) Learning rate schedule.
    ax = axes[0, 2]
    ax.plot(epochs, history.learning_rates, color='#444', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title('(c) Learning Rate Schedule', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # (d) Learned uncertainty evolution.
    ax = axes[1, 0]
    ax.plot(epochs, history.sigma_data,
            label='$\\sigma_{data}$', color='#2E86AB', linewidth=2.5)
    ax.plot(epochs, history.sigma_phy,
            label='$\\sigma_{kin}$', color='#FF6B35', linewidth=2.5)
    ax.plot(epochs, history.sigma_dyn,
            label='$\\sigma_{dyn}$', color='#6B2D5C', linewidth=2.5, linestyle='-.')
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learned Uncertainty $\\sigma$', fontsize=11)
    ax.set_title('(d) Homoscedastic Uncertainty Evolution', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # (e) Effective weights.
    ax = axes[1, 1]
    ax.plot(epochs, history.weight_data,
            label='$w_{data}$', color='#2E86AB', linewidth=2, linestyle='--')
    ax.plot(epochs, history.weight_phy,
            label='$w_{kin}$', color='#FF6B35', linewidth=2, linestyle='--')
    ax.plot(epochs, history.weight_dyn,
            label='$w_{dyn}$', color='#6B2D5C', linewidth=2, linestyle='-.')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Effective Weight $1/(2\\sigma^2)$', fontsize=11)
    ax.set_title('(e) Adaptive Loss Weights', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # (f) Validation loss components.
    ax = axes[1, 2]
    ax.plot(epochs, history.val_data, label='Val $\\mathcal{L}_{data}$',
            color='#1B998B', linewidth=2, linestyle='--')
    ax.plot(epochs, history.val_phys, label='Val $\\mathcal{L}_{kin}$',
            color='#FF6B35', linewidth=2, linestyle='--')
    ax.plot(epochs, history.val_dyn, label='Val $\\mathcal{L}_{dyn}$',
            color='#6B2D5C', linewidth=2, linestyle='-.')
    if history.val_dyn_residual:
        ax.plot(epochs, history.val_dyn_residual,
                label='Val $\\mathcal{L}_{dyn,res}$',
                color='#8E44AD', linewidth=1.5, linestyle=':')
    if history.val_dyn_prior:
        ax.plot(epochs, history.val_dyn_prior,
                label='Val $\\mathcal{L}_{dyn,prior}$',
                color='#7D3C98', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Val Loss Component', fontsize=11)
    ax.set_title('(f) Validation Loss Decomposition', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'training_curves_uncertainty.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Training curves saved to: {save_path}")


def plot_uncertainty_analysis(history: TrainingHistory, save_dir: str) -> None:
    """Generate dedicated uncertainty analysis figure.

    Args:
        history: Training history container.
        save_dir: Directory to save figure.
    """
    epochs = range(1, len(history.train_loss) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(
        'Homoscedastic Uncertainty Analysis (3-Term)',
        fontsize=13, fontweight='bold'
    )

    # (a) Sigma evolution.
    ax = axes[0]
    ax.plot(epochs, history.sigma_data, label='$\\sigma_{data}$',
            color='#2E86AB', linewidth=2.5, marker='o', markersize=3,
            markevery=max(1, len(epochs)//20))
    ax.plot(epochs, history.sigma_phy, label='$\\sigma_{kin}$',
            color='#E94F37', linewidth=2.5, marker='s', markersize=3,
            markevery=max(1, len(epochs)//20))
    ax.plot(epochs, history.sigma_dyn, label='$\\sigma_{dyn}$',
            color='#6B2D5C', linewidth=2.5, marker='^', markersize=3,
            markevery=max(1, len(epochs)//20))
    ax.fill_between(epochs, history.sigma_data, alpha=0.15, color='#2E86AB')
    ax.fill_between(epochs, history.sigma_phy, alpha=0.15, color='#E94F37')
    ax.fill_between(epochs, history.sigma_dyn, alpha=0.15, color='#6B2D5C')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learned Uncertainty $\\sigma$', fontsize=11)
    ax.set_title('(a) Task Uncertainty Evolution', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # (b) Data / Kinematics ratio.
    ax = axes[1]
    if history.sigma_data and history.sigma_phy:
        ratio_dk = [sd / (sp + 1e-8) for sd, sp in
                    zip(history.sigma_data, history.sigma_phy)]
        ax.plot(epochs, ratio_dk, color='#1B998B', linewidth=2.5,
                label='$\\sigma_{data}/\\sigma_{kin}$')
        ax.fill_between(epochs, ratio_dk, 1.0, alpha=0.2, color='#1B998B')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Ratio', fontsize=11)
    ax.set_title('(b) Data vs Kinematics Uncertainty', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # (c) Data / Dynamics ratio.
    ax = axes[2]
    if history.sigma_data and history.sigma_dyn:
        ratio_dd = [sd / (sdy + 1e-8) for sd, sdy in
                    zip(history.sigma_data, history.sigma_dyn)]
        ax.plot(epochs, ratio_dd, color='#6B2D5C', linewidth=2.5,
                label='$\\sigma_{data}/\\sigma_{dyn}$')
        ax.fill_between(epochs, ratio_dd, 1.0, alpha=0.2, color='#6B2D5C')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Ratio', fontsize=11)
    ax.set_title('(c) Data vs Dynamics Uncertainty', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'uncertainty_analysis.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Uncertainty analysis saved to: {save_path}")


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(
    path: str,
    epoch: int,
    model: RobustPINN,
    kinematic_loss_fn: KinematicPhysicsLoss,
    dynamics_loss_fn: FossenDynamicsLoss,
    adaptive_loss_fn: AdaptiveRobustLoss,
    optimizer: optim.Optimizer,
    scheduler: Any,
    val_loss: float,
    history: TrainingHistory,
    config: Dict[str, Any],
) -> None:
    """Save training checkpoint with all components.

    Args:
        path: Checkpoint file path.
        epoch: Current epoch number.
        model: RobustPINN model.
        kinematic_loss_fn: Kinematic loss module.
        dynamics_loss_fn: Fossen dynamics loss module.
        adaptive_loss_fn: Adaptive loss module.
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        val_loss: Current validation loss.
        history: Training history.
        config: Configuration dictionary.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'kinematic_loss_state_dict': kinematic_loss_fn.state_dict(),
        'dynamics_loss_state_dict': dynamics_loss_fn.state_dict(),
        'adaptive_loss_state_dict': adaptive_loss_fn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'history': history.to_dict(),
        'config': config,
        'sigma_data': adaptive_loss_fn.sigma_data,
        'sigma_phy': adaptive_loss_fn.sigma_phy,
        'sigma_dyn': adaptive_loss_fn.sigma_dyn,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: RobustPINN,
    kinematic_loss_fn: KinematicPhysicsLoss,
    dynamics_loss_fn: FossenDynamicsLoss,
    adaptive_loss_fn: AdaptiveRobustLoss,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device('cpu'),
) -> Tuple[int, float, TrainingHistory]:
    """Load training checkpoint.

    Args:
        path: Checkpoint file path.
        model: RobustPINN model to load into.
        kinematic_loss_fn: Kinematic loss module to load into.
        dynamics_loss_fn: Dynamics loss module to load into.
        adaptive_loss_fn: Adaptive loss module to load into.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        device: Target device.

    Returns:
        Tuple of (epoch, val_loss, history).
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])

    # Backward-compatible: support old checkpoints that used 'physics_loss_state_dict'.
    if 'kinematic_loss_state_dict' in checkpoint:
        kinematic_loss_fn.load_state_dict(checkpoint['kinematic_loss_state_dict'])
    elif 'physics_loss_state_dict' in checkpoint:
        kinematic_loss_fn.load_state_dict(checkpoint['physics_loss_state_dict'])

    if 'dynamics_loss_state_dict' in checkpoint:
        dynamics_loss_fn.load_state_dict(checkpoint['dynamics_loss_state_dict'])

    adaptive_loss_fn.load_state_dict(checkpoint['adaptive_loss_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (ValueError, KeyError):
            logger.warning(
                "Optimizer state incompatible (likely due to new param groups). "
                "Starting optimizer from scratch."
            )

    if scheduler and checkpoint.get('scheduler_state_dict'):
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except (ValueError, KeyError):
            logger.warning("Scheduler state incompatible. Resetting scheduler.")

    history_dict = checkpoint.get('history', {})
    history = TrainingHistory(**{
        k: v for k, v in history_dict.items()
        if k in TrainingHistory.__dataclass_fields__
    })

    return checkpoint['epoch'], checkpoint['val_loss'], history


# =============================================================================
# Main Training Function
# =============================================================================

def main() -> None:
    """Main training entry point."""
    cfg = Config()
    model_name = cfg.MODEL_NAME
    model_spec = get_benchmark_spec(model_name)
    effective_physics_mode, effective_control_mode = _effective_physics_control_modes(
        cfg, model_name
    )
    log_path = configure_training_file_logging(cfg)
    device = torch.device(
        cfg.DEVICE if hasattr(cfg, 'DEVICE')
        else 'cuda' if torch.cuda.is_available() else 'cpu'
    )

    logger.info("=" * 70)
    logger.info("AUV Trajectory Benchmark Training")
    logger.info(f"  Model: {model_spec.label} ({model_name})")
    logger.info("=" * 70)
    logger.info(f"Training log: {log_path}")
    logger.info(f"Device: {device}")

    # --- Data ---
    logger.info("Loading datasets...")
    dataset_config = DatasetConfig(
        seq_len=cfg.SEQ_LEN,
        pred_len=cfg.PRED_LEN,
        train_ratio=cfg.TRAIN_RATIO,
        val_ratio=getattr(cfg, 'VAL_RATIO', 0.15),
        batch_size=cfg.BATCH_SIZE,
        feature_cols=tuple(cfg.FEATURE_COLS),
        target_cols=tuple(cfg.TARGET_COLS),
        vel_cols=tuple(cfg.VEL_COLS),
        anchor_pos_source=getattr(cfg, 'ANCHOR_POS_SOURCE', 'gt'),
        stride=getattr(cfg, 'WINDOW_STRIDE', 1),
        thrust_cols=tuple(getattr(cfg.data, 'thrust_cols',
                                  ('thrust_net_N', 'rudder_rad', 'stern_rad'))),
    )
    pipeline = AUVDataPipeline(cfg.GT_PATH, cfg.COR_PATH, dataset_config)
    train_loader, val_loader, _ = pipeline.get_dataloaders(
        num_workers=0, pin_memory=False
    )
    dt = pipeline.mean_dt
    vel_scale = pipeline.vel_scale
    logger.info(f"Mean dt: {dt:.4f}s")
    logger.info(f"Benchmark group: {model_spec.group}")
    logger.info(f"Anchor source: {dataset_config.anchor_pos_source}")
    logger.info(f"Physics mode: {effective_physics_mode}")
    logger.info(f"Control mode: {effective_control_mode}")
    logger.info(f"Use control as feature: {getattr(cfg, 'USE_CONTROL_AS_FEATURE', False)}")
    logger.info(f"Degradation level: {getattr(cfg, 'DEGRADATION_LEVEL', 'medium')}")
    logger.info(f"Window stride: {dataset_config.stride}")
    logger.info(f"Velocity scale (σ_v): {vel_scale:.4f} m/s")
    logger.info(f"Force scale (F_char): {50.0 * vel_scale / dt:.1f} N")
    logger.info(f"Anchor stats: {pipeline.anchor_stats}")
    logger.info(f"Control columns present: {pipeline.control_columns_present}")
    logger.info(f"Control data fallback to zero: {pipeline.control_fallback_zero}")
    logger.info(f"Control stats: {pipeline.control_stats}")
    logger.info(
        "Using controlled Fossen residual: "
        f"{_uses_controlled_residual(effective_physics_mode, effective_control_mode)}"
    )
    first_batch = next(iter(train_loader))
    logger.info(f"Anchor control tensor shape: {tuple(first_batch[5].shape)}")
    logger.info(f"Target control tensor shape: {tuple(first_batch[6].shape)}")

    # --- Model ---
    model_config = ModelConfig(
        n_features=cfg.N_FEATURES,
        seq_len=cfg.SEQ_LEN,
        pred_len=cfg.PRED_LEN,
        d_model=getattr(cfg, 'D_MODEL', 128),
        nhead=getattr(cfg, 'NHEAD', 8),
        num_encoder_layers=getattr(cfg, 'NUM_LAYERS', 4),
        dim_feedforward=getattr(cfg, 'DIM_FEEDFORWARD', 256),
        dropout=cfg.DROPOUT,
        n_dof=6,  # Full 6-DOF to leverage rotational dynamics.
    )

    if model_name == 'cv':
        logger.info("Constant Velocity is analytic and does not require training.")
        logger.info("Run evaluate.py with AUV_MODEL_NAME=cv to generate metrics.")
        return

    model = create_benchmark_model(model_name, model_config).to(device)
    n_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters:    {n_model_params:,}")

    if is_data_driven_model(model_name):
        best_ckpt_path, last_ckpt_path = _benchmark_checkpoint_paths(cfg, model_name)
        logger.info(f"Best checkpoint: {best_ckpt_path}")
        logger.info(f"Last checkpoint: {last_ckpt_path}")
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.LR,
            weight_decay=getattr(cfg, 'WEIGHT_DECAY', 1e-4),
        )
        scheduler_name = getattr(cfg, 'LR_SCHEDULER', 'plateau')
        if scheduler_name == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=getattr(cfg, 'LR_PLATEAU_FACTOR', 0.5),
                patience=getattr(cfg, 'LR_PLATEAU_PATIENCE', 4),
                threshold=getattr(cfg, 'MIN_DELTA', 1e-5),
                threshold_mode='abs',
                min_lr=getattr(cfg, 'LR_MIN', cfg.LR * 0.01),
            )
        elif scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=getattr(cfg, 'LR_T0', 20),
                T_mult=getattr(cfg, 'LR_TMULT', 2),
                eta_min=getattr(cfg, 'LR_MIN', cfg.LR * 0.01),
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=getattr(cfg, 'LR_STEP', 80),
                gamma=getattr(cfg, 'LR_GAMMA', 0.5),
            )

        history_data: Dict[str, List[float]] = {'train_data': [], 'val_data': []}
        best_val = float('inf')
        best_epoch = 0
        patience_counter = 0
        start_time = time.time()
        epoch = 0
        try:
            logger.info("")
            logger.info("=" * 88)
            logger.info(f"{'Epoch':>6} | {'Train MSE':>10} | {'Val MSE':>10} | {'LR':>9} | {'Time':>7}")
            logger.info("=" * 88)
            for epoch in range(1, cfg.EPOCHS + 1):
                epoch_start = time.time()
                tr_data = train_one_epoch_data_only(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    grad_clip_norm=getattr(cfg, 'GRAD_CLIP', 5.0),
                )
                val_data = validate_data_only(model, val_loader, device)
                history_data['train_data'].append(tr_data)
                history_data['val_data'].append(val_data)
                improved = best_val - val_data > getattr(cfg, 'MIN_DELTA', 1e-5)
                if improved:
                    best_val = val_data
                    best_epoch = epoch
                    patience_counter = 0
                    save_data_only_checkpoint(
                        best_ckpt_path,
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        val_data,
                        history_data,
                        {
                            'model_name': model_name,
                            'label': model_spec.label,
                            'group': model_spec.group,
                            'pred_len': cfg.PRED_LEN,
                            'anchor': cfg.ANCHOR_POS_SOURCE,
                            'degradation': cfg.DEGRADATION_LEVEL,
                            'stride': cfg.WINDOW_STRIDE,
                            'params': n_model_params,
                        },
                    )
                else:
                    patience_counter += 1

                if scheduler_name == 'plateau':
                    scheduler.step(val_data)
                else:
                    scheduler.step()

                should_log = (
                    epoch == 1
                    or epoch % getattr(cfg, 'LOG_INTERVAL', 5) == 0
                    or improved
                    or (epoch >= getattr(cfg, 'MIN_EPOCHS', 5)
                        and patience_counter >= getattr(cfg, 'PATIENCE', 12))
                )
                if should_log:
                    logger.info(
                        f"{epoch:6d} | {tr_data:10.6f} | {val_data:10.6f} | "
                        f"{optimizer.param_groups[0]['lr']:9.2e} | "
                        f"{time.time() - epoch_start:6.1f}s"
                    )
                    logger.info(
                        f"       Monitor: best_val_mse={best_val:.6f} "
                        f"@ epoch {best_epoch}, wait={patience_counter}/{cfg.PATIENCE}"
                    )

                if epoch >= cfg.MIN_EPOCHS and patience_counter >= cfg.PATIENCE:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch}: "
                        f"best Val MSE {best_val:.6f} was at epoch {best_epoch}."
                    )
                    break
        finally:
            if epoch > 0:
                save_data_only_checkpoint(
                    last_ckpt_path,
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    history_data['val_data'][-1] if history_data['val_data'] else float('inf'),
                    history_data,
                    {
                        'model_name': model_name,
                        'label': model_spec.label,
                        'group': model_spec.group,
                        'pred_len': cfg.PRED_LEN,
                        'anchor': cfg.ANCHOR_POS_SOURCE,
                        'degradation': cfg.DEGRADATION_LEVEL,
                        'stride': cfg.WINDOW_STRIDE,
                        'params': n_model_params,
                    },
                )
        logger.info("=" * 88)
        logger.info("Training Complete!")
        logger.info(f"  Total time:   {(time.time() - start_time) / 60:.1f} minutes")
        logger.info(f"  Best val MSE: {best_val:.6f} @ epoch {best_epoch}")
        logger.info(f"  Best model:   {best_ckpt_path}")
        return

    kinematic_loss_fn, dynamics_loss_fn, adaptive_loss_fn = create_loss_modules(
        model_config,
        vel_scale=pipeline.vel_scale,
        dt=dt,
    )
    logger.info(f"Fossen n_dof initialized to: {dynamics_loss_fn.n_dof}")
    dynamics_loss_fn = dynamics_loss_fn.to(device)
    adaptive_loss_fn = adaptive_loss_fn.to(device)

    n_dyn_params = sum(p.numel() for p in dynamics_loss_fn.parameters() if p.requires_grad)
    n_loss_params = sum(p.numel() for p in adaptive_loss_fn.parameters() if p.requires_grad)

    logger.info(f"Dynamics parameters: {n_dyn_params:,} (Damping + control coefficients)")
    logger.info(f"Loss parameters:     {n_loss_params:,} (σ_data, σ_phy, σ_dyn)")
    logger.info(f"Total trainable:     {n_model_params + n_dyn_params + n_loss_params:,}")

    # --- Optimiser (3 parameter groups) ---
    base_lr = cfg.LR
    dynamics_lr_multiplier = getattr(cfg, 'DYNAMICS_LR_MULTIPLIER', 0.1)
    loss_lr_multiplier = getattr(cfg, 'LOSS_LR_MULTIPLIER', 0.1)

    optimizer = optim.AdamW([
        {
            'params': model.parameters(),
            'lr': base_lr,
            'weight_decay': getattr(cfg, 'WEIGHT_DECAY', 1e-4),
        },
        {
            'params': dynamics_loss_fn.parameters(),
            'lr': base_lr * dynamics_lr_multiplier,
            'weight_decay': 0.0,
        },
        {
            'params': adaptive_loss_fn.parameters(),
            'lr': base_lr * loss_lr_multiplier,
            'weight_decay': 0.0,
        },
    ])

    logger.info("Optimizer: AdamW (3 parameter groups)")
    logger.info(f"  Model LR:    {base_lr:.2e}")
    logger.info(f"  Dynamics LR: {base_lr * dynamics_lr_multiplier:.2e}")
    logger.info(f"  Loss sigma LR: {base_lr * loss_lr_multiplier:.2e}")

    scheduler_name = getattr(cfg, 'LR_SCHEDULER', 'plateau')
    if scheduler_name == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=getattr(cfg, 'LR_PLATEAU_FACTOR', 0.5),
            patience=getattr(cfg, 'LR_PLATEAU_PATIENCE', 4),
            threshold=getattr(cfg, 'MIN_DELTA', 1e-5),
            threshold_mode='abs',
            min_lr=getattr(cfg, 'LR_MIN', base_lr * 0.01),
        )
    elif scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=getattr(cfg, 'LR_T0', 20),
            T_mult=getattr(cfg, 'LR_TMULT', 2),
            eta_min=getattr(cfg, 'LR_MIN', base_lr * 0.01),
        )
    elif scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(cfg, 'LR_STEP', 80),
            gamma=getattr(cfg, 'LR_GAMMA', 0.5),
        )
    else:
        raise ValueError(f"Unsupported LR scheduler: {scheduler_name}")
    logger.info(f"Scheduler: {scheduler_name}")

    # --- Training loop ---
    history = TrainingHistory()
    best_val_data_loss = float('inf')  # Early stopping on pure MSE, not MLE total.
    best_epoch = 0
    patience_counter = 0
    patience = getattr(cfg, 'PATIENCE', 12)
    min_delta = getattr(cfg, 'MIN_DELTA', 1e-5)
    min_epochs = getattr(cfg, 'MIN_EPOCHS', 5)
    log_interval = getattr(cfg, 'LOG_INTERVAL', 5)
    cfg.setup_environment(verbose=False)
    anchor_suffix = dataset_config.anchor_pos_source
    physics_suffix = effective_physics_mode
    control_suffix = effective_control_mode
    best_ckpt_path, last_ckpt_path = _benchmark_checkpoint_paths(cfg, model_name)
    logger.info(f"Best checkpoint: {best_ckpt_path}")
    logger.info(f"Last checkpoint: {last_ckpt_path}")

    logger.info("")
    logger.info("=" * 120)
    header = (
        f"{'Epoch':>6} │ {'Train':>9} │ {'Val MSE':>9} │ "
        f"{'L_data':>8} │ {'L_kin':>8} │ {'L_dyn':>10} │ "
        f"{'σ_dat':>6} │ {'σ_kin':>6} │ {'σ_dyn':>6} │ "
        f"{'LR':>9} │ {'Time':>6}"
    )
    logger.info(header)
    logger.info("=" * 115)

    total_epochs = cfg.EPOCHS
    start_time = time.time()
    epoch = 0  # Initialise for finally block safety.
    val_loss = float('inf')  # Safe default if training fails before first validate().

    try:
        for epoch in range(1, total_epochs + 1):
            epoch_start = time.time()

            tr_loss, tr_data, tr_phys, tr_dyn, tr_dyn_res, tr_dyn_prior = train_one_epoch(
                model=model,
                kinematic_loss_fn=kinematic_loss_fn,
                dynamics_loss_fn=dynamics_loss_fn,
                adaptive_loss_fn=adaptive_loss_fn,
                loader=train_loader,
                optimizer=optimizer,
                dt=dt,
                device=device,
                grad_clip_norm=getattr(cfg, 'GRAD_CLIP', 5.0),
                anchor_pos_source=dataset_config.anchor_pos_source,
                physics_mode=effective_physics_mode,
                control_mode=effective_control_mode,
            )

            (
                val_loss, val_data, val_phys, val_dyn,
                val_dyn_res, val_dyn_prior
            ) = validate(
                model=model,
                kinematic_loss_fn=kinematic_loss_fn,
                dynamics_loss_fn=dynamics_loss_fn,
                adaptive_loss_fn=adaptive_loss_fn,
                loader=val_loader,
                dt=dt,
                device=device,
                anchor_pos_source=dataset_config.anchor_pos_source,
                physics_mode=effective_physics_mode,
                control_mode=effective_control_mode,
            )

            # Record history.
            sigma_data = adaptive_loss_fn.sigma_data
            sigma_phy = adaptive_loss_fn.sigma_phy
            sigma_dyn = adaptive_loss_fn.sigma_dyn

            history.train_loss.append(tr_loss)
            history.train_data.append(tr_data)
            history.train_phys.append(tr_phys)
            history.train_dyn.append(tr_dyn)
            history.train_dyn_residual.append(tr_dyn_res)
            history.train_dyn_prior.append(tr_dyn_prior)
            history.val_loss.append(val_loss)
            history.val_data.append(val_data)
            history.val_phys.append(val_phys)
            history.val_dyn.append(val_dyn)
            history.val_dyn_residual.append(val_dyn_res)
            history.val_dyn_prior.append(val_dyn_prior)
            history.sigma_data.append(sigma_data)
            history.sigma_phy.append(sigma_phy)
            history.sigma_dyn.append(sigma_dyn)
            history.weight_data.append(adaptive_loss_fn.weight_data)
            history.weight_phy.append(adaptive_loss_fn.weight_phy)
            history.weight_dyn.append(adaptive_loss_fn.weight_dyn)
            history.learning_rates.append(optimizer.param_groups[0]['lr'])

            epoch_time = time.time() - epoch_start

            # Monitor pure validation MSE for model selection. The total MLE
            # loss can be negative because of learned log-variance terms.
            improved = best_val_data_loss - val_data > min_delta
            if improved:
                best_val_data_loss = val_data
                best_epoch = epoch
                patience_counter = 0

                save_checkpoint(
                    path=best_ckpt_path,
                    epoch=epoch,
                    model=model,
                    kinematic_loss_fn=kinematic_loss_fn,
                    dynamics_loss_fn=dynamics_loss_fn,
                    adaptive_loss_fn=adaptive_loss_fn,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    val_loss=val_data,
                    history=history,
                    config={},
                )
            else:
                patience_counter += 1

            if scheduler_name == 'plateau':
                scheduler.step(val_data)
            else:
                scheduler.step()

            should_log = (
                epoch == 1
                or epoch % log_interval == 0
                or epoch == total_epochs
                or improved
                or (epoch >= min_epochs and patience_counter >= patience)
            )

            if should_log:
                lr_now = optimizer.param_groups[0]['lr']
                log_line = (
                    f"{epoch:>6} │ {tr_loss:>9.4f} │ {val_data:>9.6f} │ "
                    f"{tr_data:>8.4f} │ {tr_phys:>8.4f} │ {tr_dyn:>10.6f} │ "
                    f"{sigma_data:>6.3f} │ {sigma_phy:>6.3f} │ {sigma_dyn:>6.3f} │ "
                    f"{lr_now:>9.2e} │ {epoch_time:>5.1f}s"
                )
                logger.info(log_line)
                logger.info(
                    f"       Dynamics split: residual={tr_dyn_res:.6f}, "
                    f"damping_prior={tr_dyn_prior:.6f} | "
                    f"val_residual={val_dyn_res:.6f}, val_prior={val_dyn_prior:.6f}"
                )
                logger.info(
                    f"       Monitor: best_val_mse={best_val_data_loss:.6f} "
                    f"@ epoch {best_epoch}, wait={patience_counter}/{patience}, "
                    f"min_delta={min_delta:.1e}"
                )

                # v2.2: Periodic Fossen M/D diagnostics (every 10 epochs).
                with torch.no_grad():
                    m_mat = dynamics_loss_fn.mass_matrix
                    m_diag = torch.diag(m_mat).cpu().numpy()
                    d_diag = torch.diag(dynamics_loss_fn.damping_matrix).cpu().numpy()
                    ratio = m_diag[1] / m_diag[0] if m_diag[0] > 0 else 1.0
                    m_trans = m_diag[:3]
                    m_str = f"M_t=[{', '.join(f'{v:.1f}' for v in m_trans)}] kg"
                    if len(m_diag) > 3:
                        m_rot = m_diag[3:]
                        m_str += f"  I_r=[{', '.join(f'{v:.2f}' for v in m_rot)}] kg·m²"
                    
                    rudder_coeff = (15.0 + F.softplus(dynamics_loss_fn.raw_rudder_coeff)).item()
                    stern_coeff = (15.0 + F.softplus(dynamics_loss_fn.raw_stern_coeff)).item()
                    logger.info(
                        f"       │ {m_str} (lat/ax={ratio:.2f})  "
                        f"D=[{', '.join(f'{v:.2f}' for v in d_diag[:3])}|{', '.join(f'{v:.2f}' for v in d_diag[3:])}]  "
                        f"τ_coeff=[rudder:{rudder_coeff:.1f}, stern:{stern_coeff:.1f}]"
                    )

            if epoch >= min_epochs and patience_counter >= patience:
                logger.info("")
                logger.info(
                    f"Early stopping triggered at epoch {epoch}: "
                    f"best Val MSE {best_val_data_loss:.6f} was at epoch "
                    f"{best_epoch}, with no improvement for {patience} epochs."
                )
                break

    except KeyboardInterrupt:
        logger.info("")
        logger.info("Training interrupted by user")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    finally:
        if epoch > 0:
            save_checkpoint(
                path=last_ckpt_path,
                epoch=epoch,
                model=model,
                kinematic_loss_fn=kinematic_loss_fn,
                dynamics_loss_fn=dynamics_loss_fn,
                adaptive_loss_fn=adaptive_loss_fn,
                optimizer=optimizer,
                scheduler=scheduler,
                val_loss=val_loss if epoch > 0 else float('inf'),
                history=history,
                config={},
            )

    total_time = time.time() - start_time
    logger.info("=" * 115)
    logger.info("")
    logger.info("Training Complete!")
    logger.info(f"  Total time:     {total_time/60:.1f} minutes")
    logger.info(f"  Best val MSE:   {best_val_data_loss:.6f} @ epoch {best_epoch}")
    logger.info(f"  Final σ_data:   {history.sigma_data[-1]:.4f}")
    logger.info(f"  Final σ_kin:    {history.sigma_phy[-1]:.4f}")
    logger.info(f"  Final σ_dyn:    {history.sigma_dyn[-1]:.4f}")
    logger.info(f"  Best model:     {best_ckpt_path}")

    # --- Fossen dynamics diagnostics ---
    logger.info("")
    logger.info("Learned Fossen Hydrodynamic Parameters:")
    with torch.no_grad():
        mass_diag = torch.diag(dynamics_loss_fn.mass_matrix).cpu().numpy()
        damp_diag = torch.diag(dynamics_loss_fn.damping_matrix).cpu().numpy()
        damp_quad = torch.diag(dynamics_loss_fn.quad_damping_matrix).cpu().numpy()
        logger.info(f"  Mass M (diag):    [{', '.join(f'{m:.2f}' for m in mass_diag)}] kg")
        logger.info(f"  Damping D_lin:    [{', '.join(f'{d:.2f}' for d in damp_diag)}] N·s/m")
        logger.info(f"  Damping D_quad:   [{', '.join(f'{d:.2f}' for d in damp_quad)}] kg/m")

    logger.info("")
    logger.info("Generating training visualizations...")
    plot_training_curves(history, str(cfg.FIG_DIR))
    plot_uncertainty_analysis(history, str(cfg.FIG_DIR))

    logger.info("")
    logger.info("All done!")


if __name__ == '__main__':
    main()
