# =============================================================================
# config.py  —  Robust PINN Configuration Center (Production-Grade)
# =============================================================================
"""
Hierarchical configuration system for AUV trajectory prediction PINN.

Design Principles:
    1. Cross-platform compatibility via pathlib.Path
    2. Zero import side-effects (no module-level directory creation)
    3. Type-safe dataclass encapsulation
    4. Lazy initialization with explicit setup_environment() method
    5. Dynamic device detection with fallback logic

Usage:
    >>> from config import Config
    >>> cfg = Config()
    >>> cfg.setup_environment()  # Explicit directory creation
    >>> print(cfg.model.d_model)  # Access nested config
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Tuple

import torch


logger = logging.getLogger(__name__)


# =============================================================================
# Path Configuration
# =============================================================================

@dataclass(frozen=True)
class PathConfig:
    """File system paths with cross-platform compatibility.

    All paths are resolved relative to the project root directory,
    ensuring portability across Windows/Linux/macOS environments.

    Attributes:
        project_root: Absolute path to PINN module directory.
        data_root: Root directory containing AUV datasets.
        gt_path: Ground truth trajectory CSV path.
        cor_path: Corrupted observations CSV path.
        checkpoint_dir: Model checkpoint storage directory.
        figure_dir: Visualization output directory.
    """
    project_root: Path
    data_root: Path
    gt_path: Path
    cor_path: Path
    checkpoint_dir: Path
    figure_dir: Path

    @classmethod
    def from_project_root(cls, project_root: Path) -> PathConfig:
        """Factory method to construct paths from project root.

        Args:
            project_root: Absolute path to PINN module directory.

        Returns:
            PathConfig instance with all paths resolved.
        """
        data_root = project_root.parent / "output"

        return cls(
            project_root=project_root,
            data_root=data_root,
            gt_path=data_root / "ground_truth.csv",
            cor_path=data_root / "corrupted_data.csv",
            checkpoint_dir=project_root / "checkpoints",
            figure_dir=project_root / "figures"
        )


# =============================================================================
# Data Configuration
# =============================================================================

@dataclass(frozen=True)
class DataConfig:
    """Dataset and preprocessing configuration.

    Attributes:
        feature_cols: Input feature column names (corrupted observations).
        target_cols: Prediction target column names (GT positions).
        vel_cols: Velocity column names for physics constraints.
        seq_len: Input sequence length (historical timesteps).
        pred_len: Prediction horizon length (future timesteps).
        train_ratio: Fraction of data for training split.
        val_ratio: Fraction of data for validation split.
        gap_multiplier: Train/val gap multiplier to prevent window overlap.
    """
    feature_cols: Tuple[str, ...] = (
        'x', 'y', 'z', 'vn', 've', 'vu',
        'roll', 'pitch', 'yaw', 'ax', 'ay', 'az'
    )
    target_cols: Tuple[str, ...] = ('x', 'y', 'z')
    # NOTE: x/y/z are North/East/Down.  The legacy column name "vu" is the
    # Down-axis NED velocity, not an Up velocity.
    vel_cols: Tuple[str, ...] = ('vn', 've', 'vu')
    # Anchor ablation:
    #   last_valid = most recent valid degraded position in the input window
    #   observed   = filled degraded boundary position
    #   gt         = exact GT boundary position (ideal upper bound)
    #   zero       = no absolute anchor
    anchor_pos_source: Literal['last_valid', 'observed', 'gt', 'zero'] = field(
        default_factory=lambda: os.getenv('AUV_ANCHOR_POS_SOURCE', 'last_valid')
    )
    physics_mode: Literal['inference', 'supervised', 'none'] = field(
        default_factory=lambda: os.getenv('AUV_PHYSICS_MODE', 'inference')
    )
    control_mode: Literal['none', 'anchor_hold', 'future_truth'] = field(
        default_factory=lambda: os.getenv('AUV_CONTROL_MODE', 'none')
    )
    model_name: Literal[
        'cv',
        'lstm',
        'gru',
        'bilstm',
        'tcn',
        'vanilla_transformer',
        'vanilla_transformer_mask',
        'rope_transformer_nomask',
        'rope_transformer_mask_nophysics',
        'vrt_pinn_tau0',
        'vrt_pinn_controlled',
    ] = field(default_factory=lambda: os.getenv('AUV_MODEL_NAME', 'vrt_pinn_controlled'))
    use_control_as_feature: bool = field(
        default_factory=lambda: os.getenv('AUV_USE_CONTROL_AS_FEATURE', '0') == '1'
    )
    degradation_level: Literal['light', 'medium', 'heavy'] = field(
        default_factory=lambda: os.getenv('AUV_DEGRADATION_LEVEL', 'medium')
    )
    attitude_cols: Tuple[str, ...] = ('roll', 'pitch', 'yaw')
    body_vel_cols: Tuple[str, ...] = ('u_body', 'v_body', 'w_body')
    body_angvel_cols: Tuple[str, ...] = ('wx', 'wy', 'wz')
    thrust_cols: Tuple[str, ...] = ('thrust_net_N', 'rudder_rad', 'stern_rad')

    seq_len: int = 20
    pred_len: int = field(default_factory=lambda: int(os.getenv('AUV_PRED_LEN', '10')))
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    gap_multiplier: float = 1.0
    stride: int = field(default_factory=lambda: int(os.getenv('AUV_WINDOW_STRIDE', '1')))

    @property
    def n_features(self) -> int:
        """Number of input features."""
        return len(self.input_feature_cols)

    @property
    def input_feature_cols(self) -> Tuple[str, ...]:
        """Feature columns, optionally augmented with historical controls."""
        if not self.use_control_as_feature:
            return self.feature_cols
        return self.feature_cols + tuple(
            c for c in self.thrust_cols if c not in self.feature_cols
        )

    @property
    def n_targets(self) -> int:
        """Number of prediction targets."""
        return len(self.target_cols)

    def __post_init__(self) -> None:
        """Validate configuration consistency."""
        if not 0 < self.train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {self.train_ratio}")
        if not 0 <= self.val_ratio < 1:
            raise ValueError(f"val_ratio must be in [0, 1), got {self.val_ratio}")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be < 1")
        if self.anchor_pos_source not in ('last_valid', 'observed', 'gt', 'zero'):
            raise ValueError(
                "anchor_pos_source must be one of "
                "{'last_valid', 'observed', 'gt', 'zero'}, "
                f"got {self.anchor_pos_source!r}"
            )
        if self.physics_mode not in ('inference', 'supervised', 'none'):
            raise ValueError(
                "physics_mode must be one of {'inference', 'supervised', 'none'}, "
                f"got {self.physics_mode!r}"
            )
        if self.control_mode not in ('none', 'anchor_hold', 'future_truth'):
            raise ValueError(
                "control_mode must be one of {'none', 'anchor_hold', 'future_truth'}, "
                f"got {self.control_mode!r}"
            )
        valid_models = {
            'cv', 'lstm', 'gru', 'bilstm', 'tcn',
            'vanilla_transformer', 'vanilla_transformer_mask',
            'rope_transformer_nomask', 'rope_transformer_mask_nophysics',
            'vrt_pinn_tau0', 'vrt_pinn_controlled',
        }
        if self.model_name not in valid_models:
            raise ValueError(
                f"model_name must be one of {sorted(valid_models)}, got {self.model_name!r}"
            )
        if self.degradation_level not in ('light', 'medium', 'heavy'):
            raise ValueError(
                "degradation_level must be one of {'light', 'medium', 'heavy'}, "
                f"got {self.degradation_level!r}"
            )
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """Neural network architecture hyperparameters.

    Attributes:
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads (must divide d_model).
        num_encoder_layers: Number of Transformer encoder layers.
        dim_feedforward: Feedforward network dimension.
        dropout: Dropout probability.
        init_log_var_data: Initial log-variance for data loss.
        init_log_var_phy: Initial log-variance for physics loss.
    """
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.2
    n_dof: int = 6
    init_log_var_data: float = 0.0
    init_log_var_phy: float = 0.0
    init_log_var_dyn: float = 0.0

    def __post_init__(self) -> None:
        """Validate architecture constraints."""
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )
        head_dim = self.d_model // self.nhead
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim = d_model/nhead = {head_dim} must be even for RoPE. "
                f"Adjust d_model or nhead."
            )
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.n_dof not in (3, 6):
            raise ValueError(f"n_dof must be 3 or 6, got {self.n_dof}")


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training loop hyperparameters and optimization settings.

    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size.
        lr: Base learning rate for model parameters.
        dynamics_lr_multiplier: LR multiplier for Fossen dynamics parameters.
        loss_lr_multiplier: LR multiplier for uncertainty loss parameters.
        weight_decay: L2 regularization coefficient.
        grad_clip: Maximum gradient norm for clipping.
        patience: Early stopping patience (epochs without validation-MSE improvement).
        min_delta: Minimum absolute validation-MSE improvement.
        min_epochs: Minimum epochs before early stopping can trigger.
        log_interval: Epoch interval for progress logging.
        lr_scheduler: Learning rate scheduler type.
        lr_t0: CosineAnnealingWarmRestarts T_0 parameter.
        lr_tmult: CosineAnnealingWarmRestarts T_mult parameter.
        device: Computation device (auto-detected if not specified).
    """
    epochs: int = field(default_factory=lambda: int(os.getenv('AUV_EPOCHS', '80')))
    batch_size: int = field(default_factory=lambda: int(os.getenv('AUV_BATCH_SIZE', '64')))
    lr: float = field(default_factory=lambda: float(os.getenv('AUV_LR', '1e-3')))
    dynamics_lr_multiplier: float = 0.1
    loss_lr_multiplier: float = 0.1
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    patience: int = field(default_factory=lambda: int(os.getenv('AUV_PATIENCE', '12')))
    min_delta: float = field(default_factory=lambda: float(os.getenv('AUV_MIN_DELTA', '1e-5')))
    min_epochs: int = field(default_factory=lambda: int(os.getenv('AUV_MIN_EPOCHS', '5')))
    log_interval: int = field(default_factory=lambda: int(os.getenv('AUV_LOG_INTERVAL', '5')))

    lr_scheduler: Literal['plateau', 'cosine', 'step'] = 'plateau'
    lr_t0: int = 20
    lr_tmult: int = 2
    lr_plateau_patience: int = 4
    lr_plateau_factor: float = 0.5
    lr_min: float = 1e-5

    device: str = field(default_factory=lambda: _detect_device())

    def __post_init__(self) -> None:
        """Validate training hyperparameters."""
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}")
        if self.min_epochs <= 0:
            raise ValueError(f"min_epochs must be positive, got {self.min_epochs}")
        if self.log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {self.log_interval}")


def _detect_device() -> str:
    """Detect optimal computation device with fallback logic.

    Returns:
        Device string: 'cuda', 'mps' (Apple Silicon), or 'cpu'.
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


# =============================================================================
# Main Configuration
# =============================================================================

@dataclass
class Config:
    """Unified configuration center for Robust PINN system.

    This class aggregates all configuration subsystems and provides
    explicit environment setup to avoid import side-effects.

    Attributes:
        paths: File system path configuration.
        data: Dataset and preprocessing configuration.
        model: Neural network architecture configuration.
        train: Training loop configuration.

    Usage:
        >>> cfg = Config()
        >>> cfg.setup_environment()
        >>> model = create_model(cfg.model)
    """
    paths: PathConfig = field(default_factory=lambda: PathConfig.from_project_root(
        Path(__file__).resolve().parent
    ))
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def setup_environment(self, verbose: bool = True) -> None:
        """Create necessary directories for checkpoints and figures.

        This method must be called explicitly in the main entry point
        to avoid import side-effects and race conditions.

        Args:
            verbose: Whether to log directory creation.
        """
        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.paths.figure_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            logger.info(f"Checkpoint directory: {self.paths.checkpoint_dir}")
            logger.info(f"Figure directory: {self.paths.figure_dir}")

    @property
    def GT_PATH(self) -> Path:
        """Legacy compatibility: Ground truth CSV path."""
        return self.paths.gt_path

    @property
    def COR_PATH(self) -> Path:
        """Legacy compatibility: Corrupted data CSV path."""
        return self.paths.cor_path

    @property
    def SAVE_DIR(self) -> Path:
        """Legacy compatibility: Checkpoint directory."""
        return self.paths.checkpoint_dir

    @property
    def FIG_DIR(self) -> Path:
        """Legacy compatibility: Figure directory."""
        return self.paths.figure_dir

    @property
    def FEATURE_COLS(self) -> Tuple[str, ...]:
        """Legacy compatibility: Feature column names."""
        return self.data.input_feature_cols

    @property
    def TARGET_COLS(self) -> Tuple[str, ...]:
        """Legacy compatibility: Target column names."""
        return self.data.target_cols

    @property
    def VEL_COLS(self) -> Tuple[str, ...]:
        """Legacy compatibility: Velocity column names."""
        return self.data.vel_cols

    @property
    def ANCHOR_POS_SOURCE(self) -> str:
        """Legacy compatibility: Boundary anchor source for ablations."""
        return self.data.anchor_pos_source

    @property
    def PHYSICS_MODE(self) -> str:
        """Legacy compatibility: Physics supervision mode."""
        return self.data.physics_mode

    @property
    def CONTROL_MODE(self) -> str:
        """Control forcing mode for Fossen dynamics residual."""
        return self.data.control_mode

    @property
    def MODEL_NAME(self) -> str:
        """Benchmark model identifier selected by AUV_MODEL_NAME."""
        return self.data.model_name

    @property
    def USE_CONTROL_AS_FEATURE(self) -> bool:
        """Whether historical controls are appended to encoder features."""
        return self.data.use_control_as_feature

    @property
    def DEGRADATION_LEVEL(self) -> str:
        """Legacy compatibility: Degradation level label."""
        return self.data.degradation_level

    @property
    def WINDOW_STRIDE(self) -> int:
        """Legacy compatibility: Sliding-window stride."""
        return self.data.stride

    @property
    def N_FEATURES(self) -> int:
        """Legacy compatibility: Number of features."""
        return self.data.n_features

    @property
    def N_TARGETS(self) -> int:
        """Legacy compatibility: Number of targets."""
        return self.data.n_targets

    @property
    def SEQ_LEN(self) -> int:
        """Legacy compatibility: Sequence length."""
        return self.data.seq_len

    @property
    def PRED_LEN(self) -> int:
        """Legacy compatibility: Prediction length."""
        return self.data.pred_len

    @property
    def TRAIN_RATIO(self) -> float:
        """Legacy compatibility: Training ratio."""
        return self.data.train_ratio

    @property
    def D_MODEL(self) -> int:
        """Legacy compatibility: Model dimension."""
        return self.model.d_model

    @property
    def NHEAD(self) -> int:
        """Legacy compatibility: Number of attention heads."""
        return self.model.nhead

    @property
    def NUM_LAYERS(self) -> int:
        """Legacy compatibility: Number of encoder layers."""
        return self.model.num_encoder_layers

    @property
    def DROPOUT(self) -> float:
        """Legacy compatibility: Dropout probability."""
        return self.model.dropout

    @property
    def EPOCHS(self) -> int:
        """Legacy compatibility: Number of epochs."""
        return self.train.epochs

    @property
    def BATCH_SIZE(self) -> int:
        """Legacy compatibility: Batch size."""
        return self.train.batch_size

    @property
    def LR(self) -> float:
        """Legacy compatibility: Learning rate."""
        return self.train.lr

    @property
    def PATIENCE(self) -> int:
        """Legacy compatibility: Early stopping patience."""
        return self.train.patience

    @property
    def MIN_DELTA(self) -> float:
        """Legacy compatibility: Minimum validation-MSE improvement."""
        return self.train.min_delta

    @property
    def MIN_EPOCHS(self) -> int:
        """Legacy compatibility: Minimum epochs before early stopping."""
        return self.train.min_epochs

    @property
    def LOG_INTERVAL(self) -> int:
        """Legacy compatibility: Epoch logging interval."""
        return self.train.log_interval

    @property
    def DEVICE(self) -> str:
        """Legacy compatibility: Computation device."""
        return self.train.device

    @property
    def HIDDEN_SIZE(self) -> int:
        """Legacy compatibility: Hidden size (maps to d_model)."""
        return self.model.d_model

    @property
    def W_PHYSICS_MAX(self) -> float:
        """Legacy compatibility: Deprecated (now learned via uncertainty)."""
        return 3.0

    @property
    def W_PHYSICS_MIN(self) -> float:
        """Legacy compatibility: Deprecated (now learned via uncertainty)."""
        return 0.1

    @property
    def LR_STEP(self) -> int:
        """Legacy compatibility: LR scheduler step size."""
        return 80

    @property
    def LR_GAMMA(self) -> float:
        """Legacy compatibility: LR scheduler decay rate."""
        return 0.5

    @property
    def WEIGHT_DECAY(self) -> float:
        """Legacy compatibility: Weight decay coefficient."""
        return self.train.weight_decay

    @property
    def LOSS_LR_MULTIPLIER(self) -> float:
        """Legacy compatibility: Loss module LR multiplier."""
        return self.train.loss_lr_multiplier

    @property
    def DYNAMICS_LR_MULTIPLIER(self) -> float:
        """Legacy compatibility: Dynamics module LR multiplier."""
        return self.train.dynamics_lr_multiplier

    @property
    def GRAD_CLIP(self) -> float:
        """Legacy compatibility: Gradient clipping norm."""
        return self.train.grad_clip

    @property
    def LR_T0(self) -> int:
        """Legacy compatibility: Cosine scheduler T_0."""
        return self.train.lr_t0

    @property
    def LR_TMULT(self) -> int:
        """Legacy compatibility: Cosine scheduler T_mult."""
        return self.train.lr_tmult

    @property
    def LR_SCHEDULER(self) -> str:
        """Legacy compatibility: LR scheduler name."""
        return self.train.lr_scheduler

    @property
    def LR_PLATEAU_PATIENCE(self) -> int:
        """Legacy compatibility: ReduceLROnPlateau patience."""
        return self.train.lr_plateau_patience

    @property
    def LR_PLATEAU_FACTOR(self) -> float:
        """Legacy compatibility: ReduceLROnPlateau factor."""
        return self.train.lr_plateau_factor

    @property
    def LR_MIN(self) -> float:
        """Legacy compatibility: Minimum learning rate."""
        return self.train.lr_min

    @property
    def VAL_RATIO(self) -> float:
        """Legacy compatibility: Validation ratio."""
        return self.data.val_ratio

    @property
    def DIM_FEEDFORWARD(self) -> int:
        """Legacy compatibility: Feedforward dimension."""
        return self.model.dim_feedforward


# =============================================================================
# Configuration Validation
# =============================================================================

def validate_config(cfg: Config) -> None:
    """Validate configuration consistency across subsystems.

    Args:
        cfg: Configuration instance to validate.

    Raises:
        FileNotFoundError: If required data files are missing.
        ValueError: If configuration parameters are inconsistent.
    """
    if not cfg.paths.gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {cfg.paths.gt_path}")

    if not cfg.paths.cor_path.exists():
        raise FileNotFoundError(f"Corrupted data file not found: {cfg.paths.cor_path}")

    if cfg.data.seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {cfg.data.seq_len}")

    if cfg.data.pred_len <= 0:
        raise ValueError(f"pred_len must be positive, got {cfg.data.pred_len}")

    logger.info("Configuration validation passed")


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'Config',
    'PathConfig',
    'DataConfig',
    'ModelConfig',
    'TrainConfig',
    'validate_config',
]
