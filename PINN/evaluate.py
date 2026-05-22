# =============================================================================
# evaluate.py  —  Robust PINN Evaluation & IEEE-Grade Visualization
# =============================================================================
"""
Production-grade evaluation pipeline for AUV trajectory prediction.

Key Features:
    1. Region-stratified metrics (Normal vs. Sensor-Degraded zones)
    2. IEEE T-RO/RA-L compliant visualization with Times New Roman fonts
    3. Sensor dropout region highlighting via axvspan overlays
    4. Type-safe implementation with comprehensive error handling

Mathematical Metrics:
    RMSE = sqrt(mean((pred - gt)²))
    MAE  = mean(|pred - gt|)

Region Stratification:
    - Normal Region: validity_ratio > 0.5
    - Degraded Region: validity_ratio ≤ 0.5
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import AUVDataPipeline, DatasetConfig
from baselines import create_benchmark_model, get_benchmark_spec, is_data_driven_model, is_pinn_model
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

warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')


# =============================================================================
# IEEE-Grade Font Configuration
# =============================================================================

def configure_ieee_fonts() -> None:
    """Configure matplotlib to use Times New Roman for IEEE compliance.

    Falls back to serif fonts if Times New Roman is unavailable.
    """
    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 13
        plt.rcParams['lines.linewidth'] = 1.5
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['grid.linewidth'] = 0.5
    except Exception as e:
        logger.warning(f"Font configuration failed: {e}. Using default fonts.")


# =============================================================================
# Baseline Model (Pure Data-Driven LSTM)
# =============================================================================

class BaselineLSTM(nn.Module):
    """Baseline BiLSTM model without physics constraints.

    This serves as the data-driven baseline for comparison against
    the physics-informed Robust PINN.

    Architecture:
        Input → BiLSTM → MLP Decoder → Position Offsets

    Attributes:
        pred_len: Prediction horizon length.
        n_targets: Number of output dimensions (3 for x, y, z).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        pred_len: int = 10,
        dropout: float = 0.2
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.n_targets = 3

        self.bilstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, pred_len * self.n_targets)
        )

    def forward(self, x_seq: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x_seq: Input sequence [B, T, F].

        Returns:
            Position offsets [B, pred_len, 3].
        """
        out, _ = self.bilstm(x_seq)
        flat = self.decoder(out[:, -1, :])
        return flat.view(-1, self.pred_len, self.n_targets)


# =============================================================================
# Metrics Computation
# =============================================================================

@dataclass
class RegionMetrics:
    """Container for region-stratified evaluation metrics.

    Attributes:
        rmse: Root Mean Squared Error.
        mae: Mean Absolute Error.
        n_samples: Number of samples in this region.
        region_name: Human-readable region identifier.
    """
    rmse: float
    mae: float
    n_samples: int
    region_name: str

    def __repr__(self) -> str:
        return (f"RegionMetrics(region='{self.region_name}', "
                f"RMSE={self.rmse:.4f}, MAE={self.mae:.4f}, n={self.n_samples})")


def compute_rmse_mae(
    pred: NDArray[np.float32],
    gt: NDArray[np.float32],
    mask: Optional[NDArray[np.bool_]] = None
) -> Tuple[float, float]:
    """Compute RMSE and MAE with NaN-safe handling.

    Args:
        pred: Predicted values, shape [N, ...].
        gt: Ground truth values, shape [N, ...].
        mask: Optional boolean mask for valid samples.

    Returns:
        Tuple of (rmse, mae). Returns (nan, nan) if no valid samples.
    """
    valid_mask = ~np.isnan(gt)

    if mask is not None:
        valid_mask = valid_mask & mask

    if not valid_mask.any():
        return float('nan'), float('nan')

    diff = pred[valid_mask] - gt[valid_mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    return rmse, mae


def compute_per_region_metrics(
    preds: NDArray[np.float32],
    gts: NDArray[np.float32],
    validity_masks: NDArray[np.float32]
) -> Dict[str, RegionMetrics]:
    """Compute metrics stratified by sensor validity regions.

    Stratification Strategy:
        - Normal Region: mean(validity_mask) > 0.5
        - Degraded Region: mean(validity_mask) ≤ 0.5

    Args:
        preds: Predictions, shape [N, pred_len, 3].
        gts: Ground truth, shape [N, pred_len, 3].
        validity_masks: Validity masks, shape [N, seq_len, 1].

    Returns:
        Dictionary mapping region names to RegionMetrics objects.
    """
    n_samples = preds.shape[0]

    validity_ratios = validity_masks.mean(axis=(1, 2))

    normal_mask = validity_ratios > 0.5
    degraded_mask = validity_ratios <= 0.5

    results = {}

    for region_name, region_mask in [
        ('Normal Region', normal_mask),
        ('Degraded Region (Sensor Dropout/Noise)', degraded_mask)
    ]:
        if not region_mask.any():
            results[region_name] = RegionMetrics(
                rmse=float('nan'),
                mae=float('nan'),
                n_samples=0,
                region_name=region_name
            )
            continue

        region_preds = preds[region_mask]
        region_gts = gts[region_mask]

        rmse, mae = compute_rmse_mae(region_preds, region_gts)

        results[region_name] = RegionMetrics(
            rmse=rmse,
            mae=mae,
            n_samples=int(region_mask.sum()),
            region_name=region_name
        )

    return results


# =============================================================================
# Inference Pipeline
# =============================================================================

@dataclass
class InferenceResults:
    """Container for model inference outputs."""
    predictions: NDArray[np.float32]
    ground_truth: NDArray[np.float32]
    validity_masks: NDArray[np.float32]
    input_features: NDArray[np.float32]  # [N, seq_len, F] for maneuver analysis
    anchor_lag: NDArray[np.float32]
    fossen_residual_norm: float = float('nan')
    fossen_residual_eval_mode: str = 'not_available'


def _benchmark_checkpoint_candidates(cfg: Config, model_name: str) -> List[Path]:
    """Checkpoint candidates ordered from benchmark naming to legacy names."""
    anchor = cfg.ANCHOR_POS_SOURCE
    physics = cfg.PHYSICS_MODE
    control = cfg.CONTROL_MODE
    candidates = [
        cfg.SAVE_DIR / (
            f"best_model_p{cfg.PRED_LEN}_anchor_{anchor}"
            f"_deg_{cfg.DEGRADATION_LEVEL}_{model_name}.pth"
        )
    ]
    if model_name == 'vrt_pinn_controlled':
        candidates.append(cfg.SAVE_DIR / (
            f"best_model_p{cfg.PRED_LEN}_anchor_{anchor}"
            f"_phys_inference_ctrl_anchor_hold.pth"
        ))
    elif model_name == 'vrt_pinn_tau0':
        candidates.extend([
            cfg.SAVE_DIR / (
                f"best_model_p{cfg.PRED_LEN}_anchor_{anchor}"
                f"_phys_inference_ctrl_none.pth"
            ),
            cfg.SAVE_DIR / (
                f"best_model_p{cfg.PRED_LEN}_anchor_{anchor}_phys_inference.pth"
            ),
        ])
    elif model_name == 'bilstm':
        candidates.append(cfg.SAVE_DIR / f"baseline_best_p{cfg.PRED_LEN}_anchor_{anchor}.pth")
    candidates.append(cfg.SAVE_DIR / (
        f"best_model_p{cfg.PRED_LEN}_anchor_{anchor}_phys_{physics}_ctrl_{control}.pth"
    ))
    return candidates


def _find_checkpoint(cfg: Config, model_name: str) -> Optional[Path]:
    for path in _benchmark_checkpoint_candidates(cfg, model_name):
        if path.exists():
            return path
    return None


def _common_physics_control(model_name: str) -> Tuple[str, str]:
    """Use a controlled inference residual as the default common metric."""
    spec = get_benchmark_spec(model_name)
    if spec.kind in ('pinn', 'pgt') and spec.use_dynamics_loss:
        return spec.physics_mode, spec.control_mode
    return 'inference', 'anchor_hold'


def _select_physics_eval_checkpoint(
    cfg: Config,
    current_ckpt_path: str,
    anchor_source: str,
    physics_source: str,
    control_source: str,
) -> str:
    """Choose the checkpoint that provides dynamics parameters for evaluation.

    For physics-free ablations, the model checkpoint contains untrained
    dynamics parameters.  To keep FossenResidual_norm comparable across Full,
    no-physics, and analytical baselines, default to the matching Full
    inference checkpoint as the common evaluator when it exists.
    """
    explicit = os.getenv('AUV_PHYSICS_EVAL_CHECKPOINT')
    if explicit:
        return explicit

    if physics_source == 'none':
        candidates = [
            cfg.SAVE_DIR / (
                f'best_model_p{cfg.PRED_LEN}_anchor_{anchor_source}'
                f'_deg_{cfg.DEGRADATION_LEVEL}_vrt_pinn_controlled.pth'
            ),
            cfg.SAVE_DIR / (
                f'best_model_p{cfg.PRED_LEN}_anchor_{anchor_source}'
                f'_phys_inference_ctrl_{control_source}.pth'
            ),
            cfg.SAVE_DIR / (
                f'best_model_p{cfg.PRED_LEN}_anchor_{anchor_source}'
                f'_phys_inference_ctrl_none.pth'
            ),
            cfg.SAVE_DIR / (
                f'best_model_p{cfg.PRED_LEN}_anchor_{anchor_source}_phys_inference.pth'
            ),
        ]
        for full_ckpt in candidates:
            if full_ckpt.exists():
                return str(full_ckpt)
        logger.warning(
            "Full inference checkpoint not found for common physics evaluator; "
            "falling back to the current no-physics checkpoint dynamics state."
        )

    if physics_source == 'inference' and control_source == 'anchor_hold':
        candidates = [
            cfg.SAVE_DIR / (
                f'best_model_p{cfg.PRED_LEN}_anchor_{anchor_source}'
                f'_deg_{cfg.DEGRADATION_LEVEL}_vrt_pinn_controlled.pth'
            ),
            cfg.SAVE_DIR / (
                f'best_model_p{cfg.PRED_LEN}_anchor_{anchor_source}'
                f'_phys_inference_ctrl_anchor_hold.pth'
            ),
        ]
        for controlled_ckpt in candidates:
            if controlled_ckpt.exists():
                return str(controlled_ckpt)

    return current_ckpt_path


def _load_dynamics_evaluator(
    cfg: Config,
    model_config: ModelConfig,
    pipeline: AUVDataPipeline,
    dt: float,
    device: torch.device,
    current_ckpt_path: str,
    anchor_source: str,
    physics_source: str,
    control_source: str,
) -> Tuple[Optional[FossenDynamicsLoss], Optional[Dict[str, object]]]:
    """Create and restore the Fossen dynamics evaluator used by metrics."""
    _, dynamics_loss_fn, _ = create_loss_modules(
        model_config,
        vel_scale=pipeline.vel_scale,
        dt=dt,
    )
    dynamics_loss_fn = dynamics_loss_fn.to(device)

    eval_checkpoint: Optional[Dict[str, object]] = None
    eval_mode = 'not_available'
    eval_ckpt_path = current_ckpt_path
    if current_ckpt_path and os.path.exists(current_ckpt_path):
        maybe_checkpoint = torch.load(current_ckpt_path, map_location=device)
        if 'dynamics_loss_state_dict' in maybe_checkpoint:
            eval_checkpoint = maybe_checkpoint
            eval_mode = 'learned_checkpoint'
        else:
            eval_ckpt_path = _select_physics_eval_checkpoint(
                cfg, current_ckpt_path, anchor_source, physics_source, control_source
            )
    else:
        eval_ckpt_path = _select_physics_eval_checkpoint(
            cfg, current_ckpt_path, anchor_source, physics_source, control_source
        )

    if not os.path.exists(eval_ckpt_path):
        logger.warning(f"Physics evaluator checkpoint not found: {eval_ckpt_path}")
        return None, None

    if eval_checkpoint is None:
        eval_checkpoint = torch.load(eval_ckpt_path, map_location=device)
        eval_mode = 'nominal_evaluator'
    if 'dynamics_loss_state_dict' not in eval_checkpoint:
        logger.warning(
            "Checkpoint has no dynamics_loss_state_dict; "
            "FossenResidual_norm will be reported as NaN."
        )
        return None, None

    dynamics_loss_fn.load_state_dict(eval_checkpoint['dynamics_loss_state_dict'])
    dynamics_loss_fn.fossen_eval_mode = eval_mode
    dynamics_loss_fn.eval()
    logger.info(f"Loaded dynamics evaluator from: {eval_ckpt_path}")
    logger.info(f"Fossen residual eval mode: {eval_mode}")
    logger.info(
        f"Physics evaluator: mode={physics_source if physics_source != 'none' else 'inference(eval-only)'} | "
        f"control={control_source} | "
        f"dt={dt:.6f} | vel_scale={pipeline.vel_scale:.6f} | "
        f"force_scale={float(dynamics_loss_fn.force_scale.detach().cpu()):.6f} | "
        f"n_dof={dynamics_loss_fn.n_dof}"
    )

    return dynamics_loss_fn, eval_checkpoint


def _compute_fossen_residual_for_batch(
    dynamics_loss_fn: FossenDynamicsLoss,
    pred_pos: Tensor,
    last_vel: Tensor,
    last_pos: Tensor,
    target_thrust: Tensor,
    target_body_vel: Tensor,
    target_attitude: Tensor,
    anchor_thrust: Tensor,
    anchor_attitude: Tensor,
    dt: float,
    physics_mode: str,
    control_mode: str,
) -> float:
    """Compute the same residual branch logged by train.py validation."""
    if physics_mode == 'supervised' or control_mode == 'future_truth':
        dyn_target_vel = target_body_vel
        dyn_target_attitude = target_attitude
        dyn_thrust = target_thrust
    else:
        # For inference and no-physics ablations, use deployment-consistent
        # inputs only.  No-physics models are evaluated with this metric only;
        # they were not optimized with it.
        dyn_target_vel = None
        dyn_target_attitude = anchor_attitude.unsqueeze(1).expand(
            -1, pred_pos.shape[1], -1
        )
        if control_mode == 'anchor_hold':
            dyn_thrust = anchor_thrust.unsqueeze(1).expand(-1, pred_pos.shape[1], -1)
        elif control_mode == 'none':
            dyn_thrust = None
        else:
            raise ValueError(f"Unsupported control_mode: {control_mode}")

    _ = dynamics_loss_fn(
        pred_pos,
        last_vel,
        dt,
        thrust_data=dyn_thrust,
        target_vel=dyn_target_vel,
        target_attitude=dyn_target_attitude,
        last_pos=last_pos,
    )
    residual = getattr(dynamics_loss_fn, 'last_residual_loss', None)
    if residual is None:
        return float('nan')
    return float(residual.detach().item())


def _trajectory_accel_norm(pred: NDArray[np.float32], dt: float) -> float:
    """Second-difference trajectory smoothness metric, not Fossen physics."""
    vel = np.diff(pred, axis=1) / dt
    accel = np.diff(vel, axis=1) / dt if vel.shape[1] >= 2 else np.zeros_like(vel)
    return float(np.sqrt(np.nanmean(accel ** 2))) if accel.size else 0.0


@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_pinn: bool = True,
    dynamics_loss_fn: Optional[FossenDynamicsLoss] = None,
    dt: float = 0.05,
    physics_mode: str = 'inference',
    control_mode: str = 'none',
) -> InferenceResults:
    """Execute model inference on validation set.

    Args:
        model: Model to evaluate (RobustPINN v2.0 or BaselineLSTM).
        loader: Validation data loader.
        device: Computation device.
        is_pinn: Whether model is PINN (requires validity and last_pos).

    Returns:
        InferenceResults containing predictions and metadata.
    """
    model.eval()

    preds_list: List[NDArray] = []
    gts_list: List[NDArray] = []
    validity_list: List[NDArray] = []
    features_list: List[NDArray] = []
    anchor_lag_list: List[NDArray] = []
    fossen_residuals: List[float] = []

    for batch in loader:
        (
            x_seq, validity, target_pos, last_vel, last_pos, anchor_thrust,
            target_thrust, _target_vel, target_body_vel, target_attitude,
            _anchor_valid, anchor_lag, anchor_attitude
        ) = batch

        x_seq = x_seq.to(device, non_blocking=True)
        validity = validity.to(device, non_blocking=True)
        last_vel = last_vel.to(device, non_blocking=True)
        last_pos = last_pos.to(device, non_blocking=True)
        anchor_thrust = anchor_thrust.to(device, non_blocking=True)
        target_thrust = target_thrust.to(device, non_blocking=True)
        target_body_vel = target_body_vel.to(device, non_blocking=True)
        target_attitude = target_attitude.to(device, non_blocking=True)
        anchor_attitude = anchor_attitude.to(device, non_blocking=True)

        if is_pinn:
            pred_pos = model(x_seq, validity, last_pos)
        else:
            try:
                pred_offset = model(x_seq, validity)
            except TypeError:
                pred_offset = model(x_seq)
            pred_pos = last_pos.unsqueeze(1) + pred_offset

        if dynamics_loss_fn is not None:
            fossen_residuals.append(_compute_fossen_residual_for_batch(
                dynamics_loss_fn=dynamics_loss_fn,
                pred_pos=pred_pos,
                last_vel=last_vel,
                last_pos=last_pos,
                target_thrust=target_thrust,
                target_body_vel=target_body_vel,
                target_attitude=target_attitude,
                anchor_thrust=anchor_thrust,
                anchor_attitude=anchor_attitude,
                dt=dt,
                physics_mode=physics_mode,
                control_mode=control_mode,
            ))

        preds_list.append(pred_pos.cpu().numpy())
        gts_list.append(target_pos.cpu().numpy())
        validity_list.append(validity.cpu().numpy())
        features_list.append(x_seq.cpu().numpy())
        anchor_lag_list.append(anchor_lag.cpu().numpy())

    return InferenceResults(
        predictions=np.concatenate(preds_list, axis=0),
        ground_truth=np.concatenate(gts_list, axis=0),
        validity_masks=np.concatenate(validity_list, axis=0),
        input_features=np.concatenate(features_list, axis=0),
        anchor_lag=np.concatenate(anchor_lag_list, axis=0),
        fossen_residual_norm=float(np.nanmean(fossen_residuals)) if fossen_residuals else float('nan'),
        fossen_residual_eval_mode=(
            getattr(dynamics_loss_fn, 'fossen_eval_mode', 'not_available')
            if dynamics_loss_fn is not None else 'not_available'
        ),
    )


@torch.no_grad()
def run_constant_velocity(
    loader: DataLoader,
    device: torch.device = torch.device('cpu'),
    dynamics_loss_fn: Optional[FossenDynamicsLoss] = None,
    physics_mode: str = 'inference',
    control_mode: str = 'none',
) -> InferenceResults:
    """Evaluate a deployment-consistent constant-velocity baseline."""
    preds_list: List[NDArray] = []
    gts_list: List[NDArray] = []
    validity_list: List[NDArray] = []
    features_list: List[NDArray] = []
    anchor_lag_list: List[NDArray] = []
    fossen_residuals: List[float] = []
    dt = getattr(loader.dataset, 'mean_dt', 0.05)

    for batch in loader:
        (
            x_seq, validity, target_pos, last_vel, last_pos, anchor_thrust,
            target_thrust, _target_vel, target_body_vel, target_attitude,
            _anchor_valid, anchor_lag, anchor_attitude
        ) = batch
        steps = torch.arange(
            1, target_pos.shape[1] + 1, dtype=last_pos.dtype
        ).view(1, -1, 1)
        pred_pos = last_pos.unsqueeze(1) + steps * dt * last_vel[:, None, :3]

        if dynamics_loss_fn is not None:
            fossen_residuals.append(_compute_fossen_residual_for_batch(
                dynamics_loss_fn=dynamics_loss_fn,
                pred_pos=pred_pos.to(device),
                last_vel=last_vel.to(device),
                last_pos=last_pos.to(device),
                target_thrust=target_thrust.to(device),
                target_body_vel=target_body_vel.to(device),
                target_attitude=target_attitude.to(device),
                anchor_thrust=anchor_thrust.to(device),
                anchor_attitude=anchor_attitude.to(device),
                dt=dt,
                physics_mode=physics_mode,
                control_mode=control_mode,
            ))

        preds_list.append(pred_pos.numpy())
        gts_list.append(target_pos.numpy())
        validity_list.append(validity.numpy())
        features_list.append(x_seq.numpy())
        anchor_lag_list.append(anchor_lag.numpy())

    return InferenceResults(
        predictions=np.concatenate(preds_list, axis=0),
        ground_truth=np.concatenate(gts_list, axis=0),
        validity_masks=np.concatenate(validity_list, axis=0),
        input_features=np.concatenate(features_list, axis=0),
        anchor_lag=np.concatenate(anchor_lag_list, axis=0),
        fossen_residual_norm=float(np.nanmean(fossen_residuals)) if fossen_residuals else float('nan'),
        fossen_residual_eval_mode=(
            getattr(dynamics_loss_fn, 'fossen_eval_mode', 'not_available')
            if dynamics_loss_fn is not None else 'not_available'
        ),
    )


@torch.no_grad()
def run_constant_acceleration(
    loader: DataLoader,
    device: torch.device = torch.device('cpu'),
    dynamics_loss_fn: Optional[FossenDynamicsLoss] = None,
    physics_mode: str = 'inference',
    control_mode: str = 'none',
) -> InferenceResults:
    """Constant-acceleration baseline.

    The current dataset does not expose a clean NED acceleration estimate at
    the boundary. To keep this baseline deployment-consistent and avoid future
    labels, it estimates acceleration as zero when no raw boundary acceleration
    is available. This is intentionally conservative and is logged as such.
    """
    logger.warning(
        "Constant Acceleration baseline currently uses zero acceleration "
        "(CV-equivalent) because no NED boundary acceleration is exposed."
    )
    return run_constant_velocity(
        loader,
        device=device,
        dynamics_loss_fn=dynamics_loss_fn,
        physics_mode=physics_mode,
        control_mode=control_mode,
    )


def compute_paper_metrics(results: InferenceResults, dt: float = 0.05) -> Dict[str, float]:
    """Compute paper-facing metrics on one held-out split."""
    pred = results.predictions
    gt = results.ground_truth
    diff = pred - gt
    euclid = np.linalg.norm(diff, axis=2)  # [N, P]

    rmse, mae = compute_rmse_mae(pred, gt)
    ade = float(np.nanmean(euclid))
    fde = float(np.nanmean(euclid[:, -1]))

    validity_ratios = results.validity_masks.mean(axis=(1, 2))
    normal_mask = validity_ratios > 0.5
    degraded_mask = ~normal_mask
    normal_rmse, _ = compute_rmse_mae(pred[normal_mask], gt[normal_mask]) if normal_mask.any() else (float('nan'), float('nan'))
    degraded_rmse, _ = compute_rmse_mae(pred[degraded_mask], gt[degraded_mask]) if degraded_mask.any() else (float('nan'), float('nan'))

    accel_cols = results.input_features[:, :, 9:12]
    maneuver_intensity = np.sqrt((accel_cols ** 2).mean(axis=(1, 2)))
    high_mask = maneuver_intensity >= np.percentile(maneuver_intensity, 67)
    high_rmse, _ = compute_rmse_mae(pred[high_mask], gt[high_mask]) if high_mask.any() else (float('nan'), float('nan'))

    trajectory_accel_norm = _trajectory_accel_norm(pred, dt)
    sample_rmse = np.sqrt(np.nanmean(diff ** 2, axis=(1, 2)))

    def _trimmed_rmse(percentile: float) -> float:
        if sample_rmse.size == 0:
            return float('nan')
        threshold = np.nanpercentile(sample_rmse, percentile)
        keep = sample_rmse <= threshold
        if not keep.any():
            return float('nan')
        return float(np.sqrt(np.nanmean(diff[keep] ** 2)))

    return {
        'RMSE': rmse,
        'MAE': mae,
        'ADE': ade,
        'FDE': fde,
        'Degraded_RMSE': degraded_rmse,
        'Normal_RMSE': normal_rmse,
        'High_Maneuver_RMSE': high_rmse,
        'FossenResidual_norm': results.fossen_residual_norm,
        'FossenResidual_eval_mode': results.fossen_residual_eval_mode,
        'TrajectoryAccelNorm': trajectory_accel_norm,
        'Sample_RMSE_p50': float(np.nanpercentile(sample_rmse, 50)),
        'Sample_RMSE_p95': float(np.nanpercentile(sample_rmse, 95)),
        'Sample_RMSE_p99': float(np.nanpercentile(sample_rmse, 99)),
        'Sample_RMSE_p995': float(np.nanpercentile(sample_rmse, 99.5)),
        'Sample_RMSE_max': float(np.nanmax(sample_rmse)),
        'Trimmed_RMSE_99': _trimmed_rmse(99.0),
        'Trimmed_RMSE_995': _trimmed_rmse(99.5),
    }


def log_paper_metrics(name: str, results: InferenceResults, dt: float) -> Dict[str, float]:
    metrics = compute_paper_metrics(results, dt)
    logger.info(
        f"{name:28s} | RMSE={metrics['RMSE']:.4f} | MAE={metrics['MAE']:.4f} | "
        f"ADE={metrics['ADE']:.4f} | FDE={metrics['FDE']:.4f} | "
        f"DegRMSE={metrics['Degraded_RMSE']:.4f} | NormRMSE={metrics['Normal_RMSE']:.4f} | "
        f"HighManRMSE={metrics['High_Maneuver_RMSE']:.4f} | "
        f"Fossen={metrics['FossenResidual_norm']:.6f} | "
        f"AccelNorm={metrics['TrajectoryAccelNorm']:.4f}"
    )
    return metrics


def compute_per_step_rmse(results: InferenceResults) -> NDArray[np.float32]:
    """RMSE at each prediction step k=1..P."""
    pred = results.predictions
    gt = results.ground_truth
    diff = pred - gt
    return np.sqrt(np.nanmean(diff ** 2, axis=(0, 2))).astype(np.float32)


def log_per_step_rmse(results_by_name: Dict[str, InferenceResults]) -> None:
    """Print per-step RMSE curves for horizon-growth analysis."""
    if not results_by_name:
        return

    logger.info("")
    logger.info("=" * 70)
    logger.info("Per-Step RMSE")
    logger.info("=" * 70)

    per_step = {name: compute_per_step_rmse(res) for name, res in results_by_name.items()}
    pred_len = next(iter(per_step.values())).shape[0]
    header = "Step" + "".join(f" | {name:>18s}" for name in per_step)
    logger.info(header)
    logger.info("-" * min(len(header), 120))
    for step in range(pred_len):
        row = f"{step + 1:4d}" + "".join(
            f" | {values[step]:18.4f}" for values in per_step.values()
        )
        logger.info(row)

    tail_n = min(5, pred_len)
    summary = "Last-step / tail RMSE: " + "; ".join(
        f"{name}: step{pred_len}={values[-1]:.4f}, last{tail_n}_mean={float(np.mean(values[-tail_n:])):.4f}"
        for name, values in per_step.items()
    )
    logger.info(summary)


def log_anchor_lag_metrics(results_by_name: Dict[str, InferenceResults]) -> None:
    """Report RMSE grouped by anchor lag to expose stale-anchor difficulty."""
    if not results_by_name:
        return

    logger.info("")
    logger.info("=" * 70)
    logger.info("Anchor-Lag Stratified RMSE")
    logger.info("=" * 70)

    bins = [
        ("lag=0", lambda lag: lag == 0),
        ("1<=lag<=5", lambda lag: (lag >= 1) & (lag <= 5)),
        ("6<=lag<=10", lambda lag: (lag >= 6) & (lag <= 10)),
        ("lag>10", lambda lag: lag > 10),
    ]

    for label, mask_fn in bins:
        parts = []
        n_ref: Optional[int] = None
        for name, results in results_by_name.items():
            lag = results.anchor_lag.reshape(-1)
            mask = mask_fn(lag)
            n = int(mask.sum())
            if n_ref is None:
                n_ref = n
            if n == 0:
                parts.append(f"{name}=nan")
                continue
            rmse, _ = compute_rmse_mae(results.predictions[mask], results.ground_truth[mask])
            parts.append(f"{name}={rmse:.4f}")
        logger.info(f"{label:10s} | n={n_ref or 0:6d} | " + " | ".join(parts))


def compute_anchor_lag_rmse(results: InferenceResults) -> Dict[str, float]:
    """Return anchor-lag stratified RMSE values for one model."""
    bins = [
        ("lag=0", lambda lag: lag == 0),
        ("1<=lag<=5", lambda lag: (lag >= 1) & (lag <= 5)),
        ("6<=lag<=10", lambda lag: (lag >= 6) & (lag <= 10)),
        ("lag>10", lambda lag: lag > 10),
    ]
    lag = results.anchor_lag.reshape(-1)
    out: Dict[str, float] = {}
    for label, mask_fn in bins:
        mask = mask_fn(lag)
        if mask.any():
            out[label], _ = compute_rmse_mae(
                results.predictions[mask],
                results.ground_truth[mask],
            )
        else:
            out[label] = float('nan')
    return out


def write_benchmark_result(
    cfg: Config,
    split_name: str,
    model_name: str,
    model_label: str,
    group: str,
    params: int,
    checkpoint: Optional[Dict[str, object]],
    metrics: Dict[str, float],
    per_step: NDArray[np.float32],
    anchor_lag_metrics: Dict[str, float],
) -> Path:
    """Persist one model's evaluation result for collect_results.py."""
    result_dir = cfg.paths.project_root / "benchmark" / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / (
        f"result_p{cfg.PRED_LEN}_anchor_{cfg.ANCHOR_POS_SOURCE}"
        f"_deg_{cfg.DEGRADATION_LEVEL}_{model_name}_{split_name}.json"
    )
    best_epoch = None
    best_val = None
    if checkpoint:
        best_epoch = checkpoint.get('epoch')
        best_val = checkpoint.get('val_loss')
    payload = {
        'Group': group,
        'Model': model_label,
        'ModelName': model_name,
        'Params': int(params),
        'SeqLen': cfg.SEQ_LEN,
        'PredLen': cfg.PRED_LEN,
        'Anchor': cfg.ANCHOR_POS_SOURCE,
        'Degradation': cfg.DEGRADATION_LEVEL,
        'Stride': cfg.WINDOW_STRIDE,
        'TestSplit': split_name,
        'BestEpoch': best_epoch,
        'BestValMSE': best_val,
        'metrics': metrics,
        'per_step_rmse': {f"step_{i + 1}": float(v) for i, v in enumerate(per_step)},
        'anchor_lag_rmse': {k: float(v) for k, v in anchor_lag_metrics.items()},
        'sample_distribution': {
            key: metrics[key]
            for key in (
                'Sample_RMSE_p50',
                'Sample_RMSE_p95',
                'Sample_RMSE_p99',
                'Sample_RMSE_p995',
                'Sample_RMSE_max',
                'Trimmed_RMSE_99',
                'Trimmed_RMSE_995',
            )
            if key in metrics
        },
    }
    import json
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info(f"Benchmark result saved: {result_path}")
    return result_path


# =============================================================================
# IEEE-Grade Visualization
# =============================================================================

def plot_academic_comparison(
    pinn_results: InferenceResults,
    baseline_results: InferenceResults,
    save_dir: str,
    dt: float = 0.1
) -> None:
    """Generate IEEE-compliant comparison visualization.

    Creates a 2-row figure:
        - Row 1: 3D trajectory comparison
        - Row 2: Temporal error evolution with sensor dropout highlighting

    Args:
        pinn_results: PINN inference results.
        baseline_results: Baseline inference results.
        save_dir: Directory to save figures.
        dt: Time step interval for x-axis scaling.
    """
    configure_ieee_fonts()

    os.makedirs(save_dir, exist_ok=True)

    gt = pinn_results.ground_truth[:, 0, :]
    pinn_pred = pinn_results.predictions[:, 0, :]
    base_pred = baseline_results.predictions[:, 0, :]
    validity_ratios = pinn_results.validity_masks.mean(axis=(1, 2))

    n_samples = len(gt)
    time_axis = np.arange(n_samples) * dt

    fig = plt.figure(figsize=(14, 10))

    ax_3d = fig.add_subplot(2, 1, 1, projection='3d')

    valid_gt_mask = ~np.isnan(gt).any(axis=1)
    gt_valid = gt[valid_gt_mask]

    ax_3d.plot(gt_valid[:, 0], gt_valid[:, 1], gt_valid[:, 2],
               color='#2ECC71', linewidth=2.0, label='Ground Truth', zorder=3)
    ax_3d.plot(base_pred[:, 0], base_pred[:, 1], base_pred[:, 2],
               color='#E74C3C', linewidth=1.8, linestyle='--',
               label='Baseline LSTM', alpha=0.8, zorder=2)
    ax_3d.plot(pinn_pred[:, 0], pinn_pred[:, 1], pinn_pred[:, 2],
               color='#3498DB', linewidth=2.0, label='Robust PINN', zorder=1)

    ax_3d.set_xlabel('East (m)', labelpad=8)
    ax_3d.set_ylabel('North (m)', labelpad=8)
    ax_3d.set_zlabel('Up (m)', labelpad=8)
    ax_3d.set_title('(a) 3D Trajectory Comparison', pad=10, fontweight='bold')
    ax_3d.legend(loc='upper left', framealpha=0.95)
    ax_3d.grid(True, alpha=0.3, linewidth=0.5)

    ax_err = fig.add_subplot(2, 1, 2)

    degraded_mask = validity_ratios <= 0.5

    in_degraded = False
    start_idx = None

    for i in range(n_samples):
        if degraded_mask[i] and not in_degraded:
            start_idx = i
            in_degraded = True
        elif not degraded_mask[i] and in_degraded:
            ax_err.axvspan(time_axis[start_idx], time_axis[i - 1],
                          color='#95A5A6', alpha=0.3, zorder=0)
            in_degraded = False

    if in_degraded and start_idx is not None:
        ax_err.axvspan(time_axis[start_idx], time_axis[-1],
                      color='#95A5A6', alpha=0.3, zorder=0,
                      label='Sensor Dropout/Noise Region')

    pinn_error = np.linalg.norm(pinn_pred - gt, axis=1)
    base_error = np.linalg.norm(base_pred - gt, axis=1)

    ax_err.plot(time_axis, base_error, color='#E74C3C', linewidth=1.8,
                linestyle='--', label='Baseline LSTM Error', alpha=0.8, zorder=2)
    ax_err.plot(time_axis, pinn_error, color='#3498DB', linewidth=2.0,
                label='Robust PINN Error', zorder=3)

    ax_err.set_xlabel('Time (s)')
    ax_err.set_ylabel('Euclidean Error (m)')
    ax_err.set_title('(b) Temporal Error Evolution with Sensor Degradation Zones',
                     pad=10, fontweight='bold')
    ax_err.legend(loc='upper right', framealpha=0.95)
    ax_err.grid(True, alpha=0.3, linewidth=0.5)
    ax_err.set_xlim(time_axis[0], time_axis[-1])

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'ieee_comparison_figure.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"IEEE-grade figure saved: {save_path}")


def plot_per_step_rmse(
    pinn_results: InferenceResults,
    baseline_results: InferenceResults,
    save_dir: str
) -> None:
    """Generate per-step RMSE bar chart.

    Args:
        pinn_results: PINN inference results.
        baseline_results: Baseline inference results.
        save_dir: Directory to save figure.
    """
    configure_ieee_fonts()

    pred_len = pinn_results.predictions.shape[1]

    pinn_rmse_per_step = np.zeros(pred_len)
    base_rmse_per_step = np.zeros(pred_len)

    for step in range(pred_len):
        pinn_rmse_per_step[step], _ = compute_rmse_mae(
            pinn_results.predictions[:, step, :],
            pinn_results.ground_truth[:, step, :]
        )
        base_rmse_per_step[step], _ = compute_rmse_mae(
            baseline_results.predictions[:, step, :],
            baseline_results.ground_truth[:, step, :]
        )

    steps = np.arange(1, pred_len + 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    bar_width = 0.35
    ax.bar(steps - bar_width/2, base_rmse_per_step, bar_width,
           label='Baseline LSTM', color='#E74C3C', alpha=0.8)
    ax.bar(steps + bar_width/2, pinn_rmse_per_step, bar_width,
           label='Robust PINN', color='#3498DB', alpha=0.8)

    ax.set_xlabel('Prediction Horizon (step)')
    ax.set_ylabel('RMSE (m)')
    ax.set_title('RMSE by Prediction Horizon', fontweight='bold')
    ax.set_xticks(steps)
    ax.legend(framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    save_path = os.path.join(save_dir, 'rmse_per_step.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    logger.info(f"Per-step RMSE figure saved: {save_path}")


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def _log_maneuver_stratified(results: InferenceResults, cfg) -> None:
    """Stratify metrics by maneuver intensity (angular rate magnitude).

    Feature indices for angular velocity in the standardized input:
    feature_cols = (x, y, z, vn, ve, vu, roll, pitch, yaw, ax, ay, az)
    If angular rates (wx, wy, wz) are appended or in a known position,
    we use the acceleration columns (ax, ay, az at indices 9, 10, 11)
    as a proxy for maneuver intensity.
    """
    preds = results.predictions
    gts = results.ground_truth
    features = results.input_features  # [N, T, F] standardized

    # Use acceleration magnitude (features[:, :, 9:12]) as maneuver proxy.
    # These are standardized, so we use the absolute mean over the window.
    accel_cols = features[:, :, 9:12]  # ax, ay, az
    maneuver_intensity = np.sqrt((accel_cols ** 2).mean(axis=(1, 2)))  # [N]

    # Tertile split: low / medium / high maneuver
    thresholds = np.percentile(maneuver_intensity, [33, 67])

    categories = {
        'Low Maneuver  (bottom 33%)': maneuver_intensity <= thresholds[0],
        'Med Maneuver  (middle 33%)': (maneuver_intensity > thresholds[0]) & (maneuver_intensity <= thresholds[1]),
        'High Maneuver (top 33%)':    maneuver_intensity > thresholds[1],
    }

    validity_ratios = results.validity_masks.mean(axis=(1, 2))

    for cat_name, cat_mask in categories.items():
        if not cat_mask.any():
            continue
        rmse, mae = compute_rmse_mae(preds[cat_mask], gts[cat_mask])
        n = int(cat_mask.sum())

        # Sub-stratify by sensor condition
        normal_in_cat = cat_mask & (validity_ratios > 0.5)
        degraded_in_cat = cat_mask & (validity_ratios <= 0.5)

        line = f"  {cat_name:35s} | RMSE: {rmse:7.4f} m | MAE: {mae:7.4f} m | n={n:6d}"
        logger.info(line)

        if normal_in_cat.any() and degraded_in_cat.any():
            rmse_n, _ = compute_rmse_mae(preds[normal_in_cat], gts[normal_in_cat])
            rmse_d, _ = compute_rmse_mae(preds[degraded_in_cat], gts[degraded_in_cat])
            logger.info(f"    {'Normal':>30s}: RMSE={rmse_n:.4f} m (n={int(normal_in_cat.sum()):d})")
            logger.info(f"    {'Degraded':>30s}: RMSE={rmse_d:.4f} m (n={int(degraded_in_cat.sum()):d})")


def main() -> None:
    """Main evaluation entry point."""
    cfg = Config()
    device = torch.device(cfg.DEVICE if hasattr(cfg, 'DEVICE') else
                         'cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("=" * 70)
    logger.info("Robust PINN Evaluation Pipeline")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")

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
    train_loader, val_loader, test_loader = pipeline.get_dataloaders(
        num_workers=0, pin_memory=False
    )
    split_name = os.getenv('AUV_EVAL_SPLIT', 'test')
    if split_name == 'train':
        eval_loader = train_loader
    elif split_name == 'val':
        eval_loader = val_loader
    elif split_name == 'test':
        eval_loader = test_loader
    else:
        raise ValueError("AUV_EVAL_SPLIT must be one of train/val/test")
    dt = pipeline.mean_dt
    logger.info(
        f"Evaluation split: {split_name} | seq_len={dataset_config.seq_len} | "
        f"pred_len={dataset_config.pred_len} | anchor={dataset_config.anchor_pos_source} | "
        f"stride={dataset_config.stride} | degradation={getattr(cfg, 'DEGRADATION_LEVEL', 'medium')} | "
        f"control={getattr(cfg, 'CONTROL_MODE', 'none')}"
    )
    logger.info(f"Anchor stats: {pipeline.anchor_stats}")
    logger.info(f"Control columns present: {pipeline.control_columns_present}")
    logger.info(f"Control data fallback to zero: {pipeline.control_fallback_zero}")
    logger.info(f"Control stats: {pipeline.control_stats}")
    logger.info(
        "Dynamics evaluator uses control input: "
        f"{getattr(cfg, 'PHYSICS_MODE', 'inference') != 'none' and getattr(cfg, 'CONTROL_MODE', 'none') in ('anchor_hold', 'future_truth')}"
    )

    model_config = ModelConfig(
        n_features=cfg.N_FEATURES,
        seq_len=cfg.SEQ_LEN,
        pred_len=cfg.PRED_LEN,
        d_model=getattr(cfg, 'D_MODEL', 128),
        nhead=getattr(cfg, 'NHEAD', 8),
        num_encoder_layers=getattr(cfg, 'NUM_LAYERS', 4),
        dropout=cfg.DROPOUT,
        n_dof=6,
    )

    anchor_source = getattr(cfg, 'ANCHOR_POS_SOURCE', 'gt')
    physics_source = getattr(cfg, 'PHYSICS_MODE', 'inference')
    control_source = getattr(cfg, 'CONTROL_MODE', 'none')
    model_label = "RoPE w/o physics" if physics_source == 'none' else "Full VRT-PINN"
    if (
        anchor_source == 'gt'
        and physics_source == 'supervised'
        and control_source == 'future_truth'
        and cfg.PRED_LEN == 5
    ):
        pinn_ckpt_path = str(cfg.SAVE_DIR / 'best_model.pth')
    else:
        ckpt_with_ctrl = cfg.SAVE_DIR / (
            f'best_model_p{cfg.PRED_LEN}_anchor_{anchor_source}'
            f'_phys_{physics_source}_ctrl_{control_source}.pth'
        )
        ckpt_legacy = cfg.SAVE_DIR / (
            f'best_model_p{cfg.PRED_LEN}_anchor_{anchor_source}_phys_{physics_source}.pth'
        )
        pinn_ckpt_path = str(ckpt_with_ctrl if ckpt_with_ctrl.exists() else ckpt_legacy)
    if not os.path.exists(pinn_ckpt_path):
        logger.error(f"PINN checkpoint not found: {pinn_ckpt_path}")
        logger.error("Please run train.py first")
        return

    pinn = create_model(model_config).to(device)
    checkpoint = torch.load(pinn_ckpt_path, map_location=device)
    pinn.load_state_dict(checkpoint['model_state_dict'])
    dynamics_loss_fn, dynamics_eval_checkpoint = _load_dynamics_evaluator(
        cfg=cfg,
        model_config=model_config,
        pipeline=pipeline,
        dt=dt,
        device=device,
        current_ckpt_path=pinn_ckpt_path,
        anchor_source=anchor_source,
        physics_source=physics_source,
        control_source=control_source,
    )
    metric_physics_mode = physics_source if physics_source == 'supervised' else 'inference'
    logger.info("✓ Loaded Robust PINN checkpoint")

    if anchor_source == 'gt' and cfg.PRED_LEN == 5:
        baseline_ckpt_path = str(cfg.SAVE_DIR / 'baseline_best.pth')
    else:
        baseline_ckpt_path = str(
            cfg.SAVE_DIR / f'baseline_best_p{cfg.PRED_LEN}_anchor_{anchor_source}.pth'
        )
    if not os.path.exists(baseline_ckpt_path):
        logger.warning(f"Baseline checkpoint not found: {baseline_ckpt_path}")
        logger.warning("Skipping baseline comparison")
        baseline = None
    else:
        baseline = BaselineLSTM(
            n_features=cfg.N_FEATURES,
            hidden_size=getattr(cfg, 'HIDDEN_SIZE', 128),
            num_layers=getattr(cfg, 'NUM_LAYERS', 2),
            pred_len=cfg.PRED_LEN,
            dropout=cfg.DROPOUT
        ).to(device)
        baseline.load_state_dict(
            torch.load(baseline_ckpt_path, map_location=device)['model_state']
        )
        logger.info("✓ Loaded Baseline LSTM checkpoint")

    logger.info("")
    logger.info(f"Running inference on {split_name} set...")

    pinn_results = run_inference(
        pinn,
        eval_loader,
        device,
        is_pinn=True,
        dynamics_loss_fn=dynamics_loss_fn,
        dt=dt,
        physics_mode=metric_physics_mode,
        control_mode=control_source,
    )
    cv_results = run_constant_velocity(
        eval_loader,
        device=device,
        dynamics_loss_fn=dynamics_loss_fn,
        physics_mode=metric_physics_mode,
        control_mode=control_source,
    )
    ca_results = run_constant_acceleration(
        eval_loader,
        device=device,
        dynamics_loss_fn=dynamics_loss_fn,
        physics_mode=metric_physics_mode,
        control_mode=control_source,
    )

    if baseline is not None:
        baseline_results = run_inference(
            baseline,
            eval_loader,
            device,
            is_pinn=False,
            dynamics_loss_fn=dynamics_loss_fn,
            dt=dt,
            physics_mode=metric_physics_mode,
            control_mode=control_source,
        )
    else:
        baseline_results = None

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Global Metrics ({split_name} Set)")
    logger.info("=" * 70)

    pinn_rmse, pinn_mae = compute_rmse_mae(
        pinn_results.predictions,
        pinn_results.ground_truth
    )

    logger.info(f"Robust PINN:    RMSE = {pinn_rmse:.4f} m,  MAE = {pinn_mae:.4f} m")
    cv_metrics = log_paper_metrics("Constant Velocity", cv_results, dt)
    ca_metrics = log_paper_metrics("Constant Acceleration", ca_results, dt)
    pinn_metrics = log_paper_metrics(model_label, pinn_results, dt)

    if split_name == 'val' and dynamics_eval_checkpoint is not None:
        history = dynamics_eval_checkpoint.get('history', {})
        ref_values = history.get('val_dyn_residual') if isinstance(history, dict) else None
        if ref_values:
            train_ref = float(ref_values[-1])
            eval_res = pinn_metrics['FossenResidual_norm']
            logger.info(
                f"Fossen consistency reference: train_val_residual={train_ref:.6f}, "
                f"eval_val_residual={eval_res:.6f}"
            )
            if np.isfinite(eval_res) and train_ref > 0:
                ratio = eval_res / train_ref
                if ratio > 10.0 or ratio < 0.1:
                    force_scale = (
                        float(dynamics_loss_fn.force_scale.detach().cpu())
                        if dynamics_loss_fn is not None else float('nan')
                    )
                    n_dof = dynamics_loss_fn.n_dof if dynamics_loss_fn is not None else None
                    logger.warning(
                        "Evaluation Fossen residual is inconsistent with "
                        "training-time validation residual."
                    )
                    logger.warning(
                        f"dt={dt:.6f}, vel_scale={pipeline.vel_scale:.6f}, "
                        f"force_scale={force_scale:.6f}, n_dof={n_dof}, "
                        f"physics_mode={physics_source}, "
                        f"dynamics_state_loaded={dynamics_loss_fn is not None}"
                    )

    if baseline_results is not None:
        base_rmse, base_mae = compute_rmse_mae(
            baseline_results.predictions,
            baseline_results.ground_truth
        )
        logger.info(f"Baseline LSTM:  RMSE = {base_rmse:.4f} m,  MAE = {base_mae:.4f} m")
        log_paper_metrics("BiLSTM Baseline", baseline_results, dt)

        if base_rmse > 1e-6:
            improvement = (base_rmse - pinn_rmse) / base_rmse * 100
            logger.info(f"RMSE Improvement: {improvement:.1f}%")

    curve_results = {
        "CV": cv_results,
        model_label: pinn_results,
    }
    if baseline_results is not None:
        curve_results["BiLSTM"] = baseline_results
    log_per_step_rmse(curve_results)
    log_anchor_lag_metrics(curve_results)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Region-Stratified Metrics")
    logger.info("=" * 70)

    pinn_region_metrics = compute_per_region_metrics(
        pinn_results.predictions,
        pinn_results.ground_truth,
        pinn_results.validity_masks
    )

    logger.info("\nRobust PINN:")
    for region_name, metrics in pinn_region_metrics.items():
        logger.info(f"  {region_name:40s} | RMSE: {metrics.rmse:7.4f} m | "
                   f"MAE: {metrics.mae:7.4f} m | n={metrics.n_samples:5d}")

    if baseline_results is not None:
        base_region_metrics = compute_per_region_metrics(
            baseline_results.predictions,
            baseline_results.ground_truth,
            baseline_results.validity_masks
        )

        logger.info("\nBaseline LSTM:")
        for region_name, metrics in base_region_metrics.items():
            logger.info(f"  {region_name:40s} | RMSE: {metrics.rmse:7.4f} m | "
                       f"MAE: {metrics.mae:7.4f} m | n={metrics.n_samples:5d}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Maneuver-Intensity Stratified Metrics")
    logger.info("=" * 70)

    _log_maneuver_stratified(pinn_results, cfg)

    logger.info("")
    logger.info("Generating IEEE-grade visualizations...")

    if baseline_results is not None:
        plot_academic_comparison(pinn_results, baseline_results, str(cfg.FIG_DIR), dt)
        plot_per_step_rmse(pinn_results, baseline_results, str(cfg.FIG_DIR))
    else:
        logger.warning("Skipping comparison plots (no baseline available)")

    logger.info("")
    logger.info("Evaluation complete!")
    logger.info("=" * 70)


def main_benchmark() -> None:
    """Benchmark-aware evaluation entry point selected by AUV_MODEL_NAME."""
    cfg = Config()
    model_name = cfg.MODEL_NAME
    model_spec = get_benchmark_spec(model_name)
    device = torch.device(cfg.DEVICE if hasattr(cfg, 'DEVICE') else
                         'cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("=" * 70)
    logger.info("AUV Benchmark Evaluation Pipeline")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Model: {model_spec.label} ({model_name})")

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
    train_loader, val_loader, test_loader = pipeline.get_dataloaders(
        num_workers=0, pin_memory=False
    )
    split_name = os.getenv('AUV_EVAL_SPLIT', 'test')
    if split_name == 'train':
        eval_loader = train_loader
    elif split_name == 'val':
        eval_loader = val_loader
    elif split_name == 'test':
        eval_loader = test_loader
    else:
        raise ValueError("AUV_EVAL_SPLIT must be one of train/val/test")

    dt = pipeline.mean_dt
    physics_source, control_source = _common_physics_control(model_name)
    logger.info(
        f"Evaluation split: {split_name} | seq_len={dataset_config.seq_len} | "
        f"pred_len={dataset_config.pred_len} | anchor={dataset_config.anchor_pos_source} | "
        f"stride={dataset_config.stride} | degradation={getattr(cfg, 'DEGRADATION_LEVEL', 'medium')} | "
        f"physics_metric={physics_source} | control_metric={control_source}"
    )
    logger.info(f"Anchor stats: {pipeline.anchor_stats}")
    logger.info(f"Control columns present: {pipeline.control_columns_present}")
    logger.info(f"Control data fallback to zero: {pipeline.control_fallback_zero}")
    logger.info(f"Control stats: {pipeline.control_stats}")
    logger.info(
        "Dynamics evaluator uses control input: "
        f"{physics_source != 'none' and control_source in ('anchor_hold', 'future_truth')}"
    )

    model_config = ModelConfig(
        n_features=cfg.N_FEATURES,
        seq_len=cfg.SEQ_LEN,
        pred_len=cfg.PRED_LEN,
        d_model=getattr(cfg, 'D_MODEL', 128),
        nhead=getattr(cfg, 'NHEAD', 8),
        num_encoder_layers=getattr(cfg, 'NUM_LAYERS', 4),
        dim_feedforward=getattr(cfg, 'DIM_FEEDFORWARD', 256),
        dropout=cfg.DROPOUT,
        n_dof=6,
    )

    selected_model: Optional[nn.Module] = None
    checkpoint: Optional[Dict[str, object]] = None
    selected_ckpt_path = _find_checkpoint(cfg, model_name) if model_name != 'cv' else None
    n_params = 0
    if model_name != 'cv':
        if selected_ckpt_path is None:
            logger.error(f"Checkpoint not found for model {model_name!r}.")
            for candidate in _benchmark_checkpoint_candidates(cfg, model_name):
                logger.error(f"  tried: {candidate}")
            logger.error("Please run train.py first for this model.")
            return
        selected_model = create_benchmark_model(model_name, model_config).to(device)
        checkpoint = torch.load(selected_ckpt_path, map_location=device)
        state = checkpoint.get('model_state_dict', checkpoint.get('model_state'))
        if state is None:
            logger.error(f"Checkpoint has no model state: {selected_ckpt_path}")
            return
        selected_model.load_state_dict(state)
        n_params = sum(p.numel() for p in selected_model.parameters() if p.requires_grad)
        logger.info(f"Loaded checkpoint: {selected_ckpt_path}")
        logger.info(f"Model parameters: {n_params:,}")
    else:
        logger.info("Constant Velocity is analytic; no checkpoint is required.")

    dynamics_loss_fn, dynamics_eval_checkpoint = _load_dynamics_evaluator(
        cfg=cfg,
        model_config=model_config,
        pipeline=pipeline,
        dt=dt,
        device=device,
        current_ckpt_path=str(selected_ckpt_path) if selected_ckpt_path else "",
        anchor_source=getattr(cfg, 'ANCHOR_POS_SOURCE', 'gt'),
        physics_source=physics_source,
        control_source=control_source,
    )
    metric_physics_mode = physics_source if physics_source == 'supervised' else 'inference'

    logger.info("")
    logger.info(f"Running inference on {split_name} set...")
    cv_results = run_constant_velocity(
        eval_loader,
        device=device,
        dynamics_loss_fn=dynamics_loss_fn,
        physics_mode=metric_physics_mode,
        control_mode=control_source,
    )
    if model_name == 'cv':
        selected_results = cv_results
    else:
        assert selected_model is not None
        selected_results = run_inference(
            selected_model,
            eval_loader,
            device,
            is_pinn=is_pinn_model(model_name),
            dynamics_loss_fn=dynamics_loss_fn,
            dt=dt,
            physics_mode=metric_physics_mode,
            control_mode=control_source,
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Global Metrics ({split_name} Set)")
    logger.info("=" * 70)
    log_paper_metrics("Constant Velocity", cv_results, dt)
    selected_metrics = log_paper_metrics(model_spec.label, selected_results, dt)

    if split_name == 'val' and dynamics_eval_checkpoint is not None:
        history = dynamics_eval_checkpoint.get('history', {})
        ref_values = history.get('val_dyn_residual') if isinstance(history, dict) else None
        if ref_values:
            logger.info(
                f"Fossen consistency reference: train_val_residual={float(ref_values[-1]):.6f}, "
                f"eval_val_residual={selected_metrics['FossenResidual_norm']:.6f}"
            )

    curve_results = {"CV": cv_results, model_spec.label: selected_results}
    log_per_step_rmse(curve_results)
    log_anchor_lag_metrics(curve_results)

    per_step = compute_per_step_rmse(selected_results)
    anchor_lag_result = compute_anchor_lag_rmse(selected_results)
    write_benchmark_result(
        cfg=cfg,
        split_name=split_name,
        model_name=model_name,
        model_label=model_spec.label,
        group=model_spec.group,
        params=n_params,
        checkpoint=checkpoint,
        metrics=selected_metrics,
        per_step=per_step,
        anchor_lag_metrics=anchor_lag_result,
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("Region-Stratified Metrics")
    logger.info("=" * 70)
    selected_region_metrics = compute_per_region_metrics(
        selected_results.predictions,
        selected_results.ground_truth,
        selected_results.validity_masks
    )
    logger.info(f"\n{model_spec.label}:")
    for region_name, metrics in selected_region_metrics.items():
        logger.info(f"  {region_name:40s} | RMSE: {metrics.rmse:7.4f} m | "
                   f"MAE: {metrics.mae:7.4f} m | n={metrics.n_samples:5d}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Maneuver-Intensity Stratified Metrics")
    logger.info("=" * 70)
    _log_maneuver_stratified(selected_results, cfg)

    logger.info("")
    logger.info("Evaluation complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main_benchmark()
