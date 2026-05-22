# =============================================================================
# dataset.py  —  AUV 滑窗数据集 (Data-Leakage-Free Implementation)
# =============================================================================
"""
Production-grade time-series dataset for AUV trajectory prediction.

Key Design Principles:
    1. Strict temporal causality: Scaler fitted ONLY on training split
    2. Train/Val/Test isolation with configurable gap to prevent window overlap
    3. Robust NaN handling with high-precision validity masks
    4. Full type hints and Google-style docstrings

Mathematical Guarantee:
    Let $T_{split}$ be the train/val boundary timestamp.
    - $\mu_{train}, \sigma_{train}$ computed from $t < T_{split}$ only
    - Validation/Test transforms use frozen $(\mu_{train}, \sigma_{train})$
    - Gap of (SEQ_LEN + PRED_LEN) samples inserted to prevent input window
      from the validation set overlapping with training targets.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass(frozen=True)
class DatasetConfig:
    """Immutable configuration for AUV dataset construction.

    Attributes:
        seq_len: Input sequence length (number of historical timesteps).
        pred_len: Prediction horizon length.
        train_ratio: Fraction of data for training (0 < ratio < 1).
        val_ratio: Fraction of data for validation. Test = 1 - train - val.
        batch_size: Mini-batch size for DataLoader.
        feature_cols: Column names for input features.
        target_cols: Column names for position targets (x, y, z).
        vel_cols: NED velocity columns (vn, ve, vu). The legacy name "vu"
            stores Down velocity and the order must match target_cols.
        anchor_pos_source: Source for the boundary anchor returned as
            last_pos. Options: 'last_valid' uses the most recent valid
            degraded position in the history window, 'gt' uses exact ground
            truth, 'observed' uses filled corrupted observations, and 'zero'
            removes the absolute anchor.
        stride: Sliding-window start stride.
        gap_multiplier: Multiplier for train/val gap. Gap = multiplier * (seq_len + pred_len).
    """
    seq_len: int = 20
    pred_len: int = 10
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    batch_size: int = 64
    feature_cols: Tuple[str, ...] = (
        'x', 'y', 'z', 'vn', 've', 'vu',
        'roll', 'pitch', 'yaw', 'ax', 'ay', 'az'
    )
    target_cols: Tuple[str, ...] = ('x', 'y', 'z')
    # x/y/z are North/East/Down.  "vu" is a legacy name for Down velocity.
    vel_cols: Tuple[str, ...] = ('vn', 've', 'vu')
    attitude_cols: Tuple[str, ...] = ('roll', 'pitch', 'yaw')
    body_vel_cols: Tuple[str, ...] = ('u_body', 'v_body', 'w_body')
    body_angvel_cols: Tuple[str, ...] = ('wx', 'wy', 'wz')
    thrust_cols: Tuple[str, ...] = ('thrust_net_N', 'rudder_rad', 'stern_rad')
    anchor_pos_source: str = 'last_valid'
    gap_multiplier: float = 1.0
    stride: int = 1

    def __post_init__(self) -> None:
        if not 0 < self.train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0, 1), got {self.train_ratio}")
        if not 0 <= self.val_ratio < 1:
            raise ValueError(f"val_ratio must be in [0, 1), got {self.val_ratio}")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be < 1")
        if self.anchor_pos_source not in ('last_valid', 'gt', 'observed', 'zero'):
            raise ValueError(
                "anchor_pos_source must be one of "
                "{'last_valid', 'gt', 'observed', 'zero'}, "
                f"got {self.anchor_pos_source!r}"
            )
        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")


class ValidityMaskBuilder:
    """Constructs high-precision validity masks from corrupted observations.

    Validity is determined by:
        1. NaN presence in position columns → invalid
        2. Impulse noise flag (if available) → invalid
        3. Feature-wise NaN ratio exceeding threshold → degraded confidence

    Output mask values:
        - 1.0: Fully valid observation
        - 0.0: Missing or corrupted observation
        - (0, 1): Partial validity (future extension for soft masks)
    """

    def __init__(
        self,
        target_cols: Tuple[str, ...],
        feature_cols: Tuple[str, ...],
        impulse_col: str = 'is_impulse'
    ) -> None:
        self._target_cols = target_cols
        self._feature_cols = feature_cols
        self._impulse_col = impulse_col

    def build(self, cor_df: pd.DataFrame) -> NDArray[np.float32]:
        """Build validity mask array.

        Args:
            cor_df: Corrupted observation DataFrame.

        Returns:
            validity: Shape [N], values in [0, 1].
        """
        n_samples = len(cor_df)

        pos_invalid = cor_df[list(self._target_cols)].isna().any(axis=1).values

        if self._impulse_col in cor_df.columns:
            impulse_invalid = cor_df[self._impulse_col].values.astype(bool)
        else:
            impulse_invalid = np.zeros(n_samples, dtype=bool)

        invalid_mask = pos_invalid | impulse_invalid
        validity = (~invalid_mask).astype(np.float32)

        return validity


class TemporalScalerWrapper:
    """Wrapper ensuring scaler is fitted only on designated training data.

    This class enforces the temporal causality constraint by:
        1. Accepting pre-split training data for fitting
        2. Freezing statistics after fit
        3. Providing transform-only interface for val/test data

    Mathematical Formulation:
        Given training features $X_{train} \in \mathbb{R}^{N_{train} \times F}$:
        $$\mu_f = \frac{1}{N_{train}} \sum_{i=1}^{N_{train}} X_{train}[i, f]$$
        $$\sigma_f = \sqrt{\frac{1}{N_{train}} \sum_{i=1}^{N_{train}} (X_{train}[i, f] - \mu_f)^2}$$

        Transform: $\tilde{X}[i, f] = \frac{X[i, f] - \mu_f}{\sigma_f + \epsilon}$
    """

    def __init__(self) -> None:
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted: bool = False
        self._n_features: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def mean(self) -> Optional[NDArray[np.float64]]:
        return self._scaler.mean_ if self._is_fitted else None

    @property
    def std(self) -> Optional[NDArray[np.float64]]:
        return self._scaler.scale_ if self._is_fitted else None

    def fit(self, X_train: NDArray[np.float32]) -> TemporalScalerWrapper:
        """Fit scaler on training data only.

        Args:
            X_train: Training features, shape [N_train, F].

        Returns:
            self for method chaining.

        Raises:
            ValueError: If X_train contains all-NaN columns.
        """
        if np.all(np.isnan(X_train), axis=0).any():
            raise ValueError("Training data contains all-NaN columns, cannot fit scaler")

        self._scaler = StandardScaler()
        self._scaler.fit(X_train)
        self._is_fitted = True
        self._n_features = X_train.shape[1]

        return self

    def transform(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        """Transform features using frozen training statistics.

        Args:
            X: Features to transform, shape [N, F].

        Returns:
            Standardized features, shape [N, F].

        Raises:
            RuntimeError: If scaler not fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")

        return self._scaler.transform(X).astype(np.float32)

    def get_sklearn_scaler(self) -> StandardScaler:
        """Return underlying sklearn scaler for serialization."""
        if not self._is_fitted:
            raise RuntimeError("Scaler not fitted")
        return self._scaler


class AUVSlidingWindowDataset(Dataset):
    """Memory-efficient sliding window dataset for AUV trajectory prediction.

    This dataset implements lazy window extraction from pre-loaded arrays,
    avoiding redundant memory copies while maintaining O(1) access time.

    Sample Structure:
        - x_seq: [SEQ_LEN, N_FEATURES] - Standardized input sequence
        - validity: [SEQ_LEN, 1] - Per-timestep validity mask
        - target_pos: [PRED_LEN, 3] - Future positions in meters
        - last_vel: [3] - NED velocity at the selected anchor
        - last_pos: [3] - Boundary anchor selected by anchor_pos_source
          (GT, filled corrupted observation, or zero-anchor ablation)
        - anchor_thrust: [3] - Control at the selected anchor
          (thrust_net_N, rudder_rad, stern_rad)
        - target_thrust: [PRED_LEN, 3] - Future thrust/control sequence
          for step-wise Fossen forcing.
        - target_attitude: [PRED_LEN, 3] - Future roll/pitch/yaw for
          rotating predicted NED velocity into the body frame.

    Attributes:
        n_samples: Total number of valid sliding windows.
        n_features: Number of input features.
        mean_dt: Mean sampling interval in seconds.
    """

    def __init__(
        self,
        features: NDArray[np.float32],
        validity: NDArray[np.float32],
        gt_pos: NDArray[np.float32],
        gt_vel: NDArray[np.float32],
        gt_thrust: NDArray[np.float32],
        gt_attitude: NDArray[np.float32],
        anchor_pos: NDArray[np.float32],
        anchor_vel: NDArray[np.float32],
        anchor_attitude: NDArray[np.float32],
        anchor_thrust: NDArray[np.float32],
        anchor_valid: NDArray[np.float32],
        anchor_lag: NDArray[np.float32],
        mean_dt: float,
        seq_len: int,
        pred_len: int,
        valid_indices: Optional[List[int]] = None,
        gt_body_vel: Optional[NDArray[np.float32]] = None,
    ) -> None:
        """Initialize dataset from pre-processed arrays.

        Args:
            features: Standardized features, shape [N, F].
            validity: Validity mask, shape [N].
            gt_pos: Ground truth positions, shape [N, 3].
            gt_vel: Ground truth NED velocities, shape [N, 3].
            gt_thrust: Ground truth thrust/control, shape [N, 3].
            gt_attitude: Ground truth roll/pitch/yaw, shape [N, 3].
            anchor_pos: Boundary anchor positions, shape [N, 3].
            anchor_vel: Boundary anchor velocities, shape [N, 3].
            anchor_attitude: Boundary anchor roll/pitch/yaw, shape [N, 3].
            anchor_thrust: Boundary anchor controls, shape [N, 3].
            anchor_valid: Whether last-valid anchor found a real valid
                observation, shape [N].
            anchor_lag: Number of steps between boundary and anchor, shape [N].
            mean_dt: Mean sampling interval.
            seq_len: Input sequence length.
            pred_len: Prediction horizon.
            valid_indices: Optional list of valid window start indices.
            gt_body_vel: Body-frame velocities [u,v,w], shape [N, 3].
        """
        self._features = features
        self._validity = validity
        self._gt_pos = gt_pos
        self._gt_vel = gt_vel
        self._gt_thrust = gt_thrust
        self._gt_attitude = gt_attitude
        self._anchor_pos = anchor_pos
        self._anchor_vel = anchor_vel
        self._anchor_attitude = anchor_attitude
        self._anchor_thrust = anchor_thrust
        self._anchor_valid = anchor_valid
        self._anchor_lag = anchor_lag
        self._gt_body_vel = gt_body_vel if gt_body_vel is not None else gt_vel
        self._mean_dt = mean_dt
        self._seq_len = seq_len
        self._pred_len = pred_len

        n_total = len(features)
        max_start_idx = n_total - seq_len - pred_len

        if valid_indices is not None:
            self._indices = [i for i in valid_indices if 0 <= i <= max_start_idx]
        else:
            self._indices = list(range(max_start_idx + 1))

    @property
    def n_samples(self) -> int:
        return len(self._indices)

    @property
    def n_features(self) -> int:
        return self._features.shape[1]

    @property
    def mean_dt(self) -> float:
        return self._mean_dt

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Extract single sliding window sample.

        Args:
            idx: Sample index in [0, n_samples).

        Returns:
            Tuple of (x_seq, validity, target_pos, last_vel, last_pos,
                      anchor_thrust, target_thrust, target_vel, target_body_vel,
                      target_attitude).

        Changed in v2.2:
            Added target_vel [PRED_LEN, 3] with GT NED velocity over the prediction
            horizon.
            Added target_thrust [PRED_LEN, 3], target_body_vel [PRED_LEN, 6],
            and target_attitude [PRED_LEN, 3]. FossenDynamicsLoss rotates
            velocity derived from predicted NED positions into the body frame
            and uses step-wise forcing before applying the residual.
        Changed in v2.3:
            The sixth returned tensor is anchor_thrust, aligned with
            last_pos/last_vel/anchor_attitude. It replaces the old
            window-end last_thrust for controlled inference physics.
        """
        start = self._indices[idx]
        seq_end = start + self._seq_len
        pred_end = seq_end + self._pred_len

        x_seq = torch.from_numpy(self._features[start:seq_end].copy())
        validity = torch.from_numpy(
            self._validity[start:seq_end].copy()
        ).unsqueeze(-1)
        target_pos = torch.from_numpy(self._gt_pos[seq_end:pred_end].copy())
        last_vel = torch.from_numpy(self._anchor_vel[seq_end - 1].copy())
        anchor_idx = seq_end - 1
        last_pos = torch.from_numpy(self._anchor_pos[anchor_idx].copy())
        anchor_valid = torch.tensor(self._anchor_valid[anchor_idx], dtype=torch.float32)
        anchor_lag = torch.tensor(self._anchor_lag[anchor_idx], dtype=torch.float32)
        anchor_attitude = torch.from_numpy(self._anchor_attitude[anchor_idx].copy())
        anchor_thrust = torch.from_numpy(self._anchor_thrust[anchor_idx].copy())
        target_thrust = torch.from_numpy(self._gt_thrust[seq_end:pred_end].copy())
        target_vel = torch.from_numpy(self._gt_vel[seq_end:pred_end].copy())
        target_body_vel = torch.from_numpy(self._gt_body_vel[seq_end:pred_end].copy())
        target_attitude = torch.from_numpy(self._gt_attitude[seq_end:pred_end].copy())

        return (
            x_seq, validity, target_pos, last_vel, last_pos, anchor_thrust,
            target_thrust, target_vel, target_body_vel, target_attitude,
            anchor_valid, anchor_lag, anchor_attitude
        )


class AUVDataPipeline:
    """End-to-end data pipeline with strict temporal isolation.

    This pipeline guarantees:
        1. Scaler fitted exclusively on training split
        2. Configurable gap between train/val/test to prevent window overlap
        3. Consistent preprocessing across all splits

    Usage:
        >>> pipeline = AUVDataPipeline(gt_path, cor_path, config)
        >>> train_loader, val_loader, test_loader = pipeline.get_dataloaders()
        >>> dt = pipeline.mean_dt
        >>> scaler = pipeline.scaler
    """

    def __init__(
        self,
        gt_path: Union[str, Path],
        cor_path: Union[str, Path],
        config: DatasetConfig
    ) -> None:
        """Initialize pipeline and execute preprocessing.

        Args:
            gt_path: Path to ground truth CSV.
            cor_path: Path to corrupted observations CSV.
            config: Dataset configuration.
        """
        self._config = config
        self._gt_path = Path(gt_path)
        self._cor_path = Path(cor_path)

        self._scaler = TemporalScalerWrapper()
        self._mean_dt: float = 0.1
        self._vel_scale: float = 1.0  # std(velocity) over training set

        self._features: Optional[NDArray[np.float32]] = None
        self._validity: Optional[NDArray[np.float32]] = None
        self._gt_pos: Optional[NDArray[np.float32]] = None
        self._gt_vel: Optional[NDArray[np.float32]] = None
        self._anchor_pos: Optional[NDArray[np.float32]] = None
        self._anchor_vel: Optional[NDArray[np.float32]] = None
        self._anchor_attitude: Optional[NDArray[np.float32]] = None
        self._anchor_thrust: Optional[NDArray[np.float32]] = None
        self._anchor_valid: Optional[NDArray[np.float32]] = None
        self._anchor_lag: Optional[NDArray[np.float32]] = None
        self._gt_attitude: Optional[NDArray[np.float32]] = None
        self._gt_body_vel: Optional[NDArray[np.float32]] = None
        self._gt_thrust: Optional[NDArray[np.float32]] = None
        self._control_columns_present: bool = False
        self._control_fallback_zero: bool = False

        self._train_indices: List[int] = []
        self._val_indices: List[int] = []
        self._test_indices: List[int] = []

        self._load_and_preprocess()

    @property
    def mean_dt(self) -> float:
        return self._mean_dt

    @property
    def vel_scale(self) -> float:
        """Characteristic velocity scale (std over training set).

        Used to non-dimensionalize the Fossen dynamics residual so that
        L_dyn is on the same order of magnitude as L_data and L_kin.
        """
        return self._vel_scale

    @property
    def scaler(self) -> StandardScaler:
        return self._scaler.get_sklearn_scaler()

    @property
    def n_features(self) -> int:
        return len(self._config.feature_cols)

    @property
    def control_columns_present(self) -> bool:
        return self._control_columns_present

    @property
    def control_fallback_zero(self) -> bool:
        return self._control_fallback_zero

    @property
    def control_stats(self) -> Dict[str, float]:
        if self._gt_thrust is None:
            return {}
        stats: Dict[str, float] = {
            'control_columns_present': float(self._control_columns_present),
            'control_fallback_zero': float(self._control_fallback_zero),
        }
        for idx, name in enumerate(self._config.thrust_cols):
            values = self._gt_thrust[:, idx]
            stats[f'{name}_mean'] = float(np.mean(values))
            stats[f'{name}_std'] = float(np.std(values))
            stats[f'{name}_min'] = float(np.min(values))
            stats[f'{name}_max'] = float(np.max(values))
            stats[f'{name}_nonzero_rate'] = float(np.mean(np.abs(values) > 1e-12))
        return stats

    @property
    def anchor_stats(self) -> Dict[str, float]:
        if self._anchor_valid is None or self._anchor_lag is None:
            return {}
        return {
            'anchor_valid_rate': float(np.mean(self._anchor_valid)),
            'anchor_fallback_rate': float(1.0 - np.mean(self._anchor_valid)),
            'anchor_lag_mean': float(np.mean(self._anchor_lag)),
            'anchor_lag_p95': float(np.percentile(self._anchor_lag, 95)),
            'anchor_lag_max': float(np.max(self._anchor_lag)),
        }

    def _load_and_preprocess(self) -> None:
        """Load data and execute preprocessing with temporal isolation.

        Supports two modes:
            1. Multi-trajectory: if 'trajectory_id' and 'split' columns exist,
               windows are constrained to never cross trajectory boundaries.
               Split is determined by the 'split' column (train/val/test).
            2. Legacy single-trajectory: falls back to time-based splitting.
        """
        gt_df = pd.read_csv(self._gt_path)
        cor_df = pd.read_csv(self._cor_path)

        if len(gt_df) != len(cor_df):
            raise ValueError(
                f"GT and corrupted data length mismatch: {len(gt_df)} vs {len(cor_df)}"
            )

        n_total = len(gt_df)
        self._multi_trajectory = (
            'trajectory_id' in gt_df.columns and 'split' in gt_df.columns
        )

        self._mean_dt = self._compute_mean_dt(gt_df)

        mask_builder = ValidityMaskBuilder(
            self._config.target_cols,
            self._config.feature_cols
        )
        self._validity = mask_builder.build(cor_df)

        self._gt_pos = gt_df[list(self._config.target_cols)].values.astype(np.float32)
        self._gt_vel = gt_df[list(self._config.vel_cols)].values.astype(np.float32)
        attitude_cols = list(getattr(self._config, 'attitude_cols', ('roll', 'pitch', 'yaw')))
        if all(c in gt_df.columns for c in attitude_cols):
            self._gt_attitude = gt_df[attitude_cols].values.astype(np.float32)
        else:
            warnings.warn(
                f"Attitude columns {attitude_cols} not found. "
                "Using zero attitude for body-frame dynamics rotation."
            )
            self._gt_attitude = np.zeros((n_total, 3), dtype=np.float32)

        # Body-frame velocities [u, v, w] for Fossen dynamics equation.
        body_vel_cols = list(self._config.body_vel_cols)
        body_angvel_cols = list(getattr(self._config, 'body_angvel_cols', ()))
        if all(c in gt_df.columns for c in body_vel_cols):
            body_lin = gt_df[body_vel_cols].values.astype(np.float32)
        else:
            warnings.warn(
                f"Body-frame velocity columns {body_vel_cols} not found. "
                "Falling back to NED velocities (physically incorrect)."
            )
            body_lin = self._gt_vel.copy()

        # Angular rates [p, q, r] for 6-DOF Fossen dynamics.
        if body_angvel_cols and all(c in gt_df.columns for c in body_angvel_cols):
            body_ang = gt_df[body_angvel_cols].values.astype(np.float32)
            self._gt_body_vel = np.concatenate([body_lin, body_ang], axis=1)  # [N, 6]
        else:
            if body_angvel_cols:
                warnings.warn(
                    f"Angular rate columns {body_angvel_cols} not found. "
                    "Using 3-DOF body velocity only (C matrix will be zero)."
                )
            self._gt_body_vel = body_lin  # [N, 3]

        # Thrust / control inputs for Fossen dynamics (τ vector).
        thrust_cols = list(self._config.thrust_cols)
        if all(c in gt_df.columns for c in thrust_cols):
            self._gt_thrust = gt_df[thrust_cols].values.astype(np.float32)
            self._control_columns_present = True
            self._control_fallback_zero = False
        else:
            # Fallback: zero thrust (passive damping assumption).
            warnings.warn(
                f"Thrust columns {thrust_cols} not found in GT data. "
                "Falling back to zero thrust (passive damping)."
            )
            self._gt_thrust = np.zeros((n_total, len(thrust_cols)), dtype=np.float32)
            self._control_columns_present = False
            self._control_fallback_zero = True

        features_raw = self._fill_missing_features(cor_df)
        (
            self._anchor_pos,
            self._anchor_vel,
            self._anchor_attitude,
            self._anchor_thrust,
            self._anchor_valid,
            self._anchor_lag,
        ) = (
            self._build_anchor_positions(cor_df, features_raw)
        )

        if self._multi_trajectory:
            self._load_multi_trajectory(gt_df, features_raw)
        else:
            n_train_raw = int(n_total * self._config.train_ratio)
            n_val_raw = int(n_total * self._config.val_ratio)
            train_features = features_raw[:n_train_raw]
            self._scaler.fit(train_features)
            self._features = self._scaler.transform(features_raw)
            self._vel_scale = float(np.std(self._gt_body_vel[:n_train_raw]).clip(min=1e-6))
            self._compute_split_indices(n_total, n_train_raw, n_val_raw)

    def _build_anchor_positions(
        self,
        cor_df: pd.DataFrame,
        features_raw: NDArray[np.float32],
    ) -> Tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        """Build boundary anchors for GT/observed/no-anchor ablations."""
        source = self._config.anchor_pos_source
        n_total = len(features_raw)
        feature_cols = list(self._config.feature_cols)
        vel_idx = [feature_cols.index(c) for c in self._config.vel_cols]
        att_idx = [feature_cols.index(c) for c in self._config.attitude_cols]
        observed_vel = features_raw[:, vel_idx].astype(np.float32, copy=True)
        observed_att = features_raw[:, att_idx].astype(np.float32, copy=True)
        if source == 'gt':
            return (
                self._gt_pos.copy(),
                self._gt_vel.copy(),
                self._gt_attitude.copy(),
                self._gt_thrust.copy(),
                np.ones(n_total, dtype=np.float32),
                np.zeros(n_total, dtype=np.float32),
            )
        if source == 'zero':
            return (
                np.zeros((n_total, len(self._config.target_cols)), dtype=np.float32),
                np.zeros((n_total, len(self._config.vel_cols)), dtype=np.float32),
                np.zeros((n_total, len(self._config.attitude_cols)), dtype=np.float32),
                np.zeros((n_total, len(self._config.thrust_cols)), dtype=np.float32),
                np.zeros(n_total, dtype=np.float32),
                np.zeros(n_total, dtype=np.float32),
            )

        try:
            pos_idx = [feature_cols.index(c) for c in self._config.target_cols]
        except ValueError as exc:
            raise ValueError(
                "anchor_pos_source='observed' requires target_cols to be "
                "present in feature_cols."
            ) from exc
        observed_anchor = features_raw[:, pos_idx].astype(np.float32, copy=True)
        if source == 'observed':
            return (
                observed_anchor,
                observed_vel,
                observed_att,
                self._gt_thrust.copy(),
                self._validity.astype(np.float32, copy=True),
                np.zeros(n_total, dtype=np.float32),
            )

        # Last-valid anchor: for each row, use the most recent valid degraded
        # position within the same trajectory. If none exists, fall back to the
        # filled observed boundary value and mark anchor_valid=0.
        anchor = np.empty_like(observed_anchor)
        anchor_vel = np.empty_like(observed_vel)
        anchor_att = np.empty_like(observed_att)
        anchor_thrust = np.empty_like(self._gt_thrust)
        anchor_valid = np.zeros(n_total, dtype=np.float32)
        anchor_lag = np.zeros(n_total, dtype=np.float32)

        if 'trajectory_id' in cor_df.columns:
            groups = cor_df.groupby('trajectory_id', sort=False).indices.values()
        else:
            groups = [np.arange(n_total)]

        validity = self._validity.astype(bool)
        for group_indices in groups:
            last_valid_idx: Optional[int] = None
            for idx in group_indices:
                if validity[idx]:
                    last_valid_idx = int(idx)
                if last_valid_idx is None:
                    anchor[idx] = observed_anchor[idx]
                    anchor_vel[idx] = observed_vel[idx]
                    anchor_att[idx] = observed_att[idx]
                    anchor_thrust[idx] = self._gt_thrust[idx]
                    anchor_valid[idx] = 0.0
                    anchor_lag[idx] = 0.0
                else:
                    anchor[idx] = observed_anchor[last_valid_idx]
                    anchor_vel[idx] = observed_vel[last_valid_idx]
                    anchor_att[idx] = observed_att[last_valid_idx]
                    anchor_thrust[idx] = self._gt_thrust[last_valid_idx]
                    anchor_valid[idx] = 1.0
                    anchor_lag[idx] = float(int(idx) - last_valid_idx)

        return anchor, anchor_vel, anchor_att, anchor_thrust, anchor_valid, anchor_lag

    def _load_multi_trajectory(
        self,
        gt_df: pd.DataFrame,
        features_raw: NDArray[np.float32],
    ) -> None:
        """Process multi-trajectory data with trajectory-aware splitting.

        Windows never cross trajectory boundaries. The scaler is fitted
        only on training trajectories.
        """
        seq_len = self._config.seq_len
        pred_len = self._config.pred_len
        window = seq_len + pred_len

        traj_ids = gt_df['trajectory_id'].values
        splits = gt_df['split'].values

        # Identify contiguous trajectory segments.
        unique_tids = gt_df['trajectory_id'].unique()
        traj_segments: List[Tuple[int, int, str]] = []
        for tid in unique_tids:
            mask = traj_ids == tid
            indices = np.where(mask)[0]
            start, end = int(indices[0]), int(indices[-1]) + 1
            split = str(splits[start])
            traj_segments.append((start, end, split))

        # Fit scaler on training trajectories only.
        train_rows = []
        for start, end, split in traj_segments:
            if split == 'train':
                train_rows.append(features_raw[start:end])

        if train_rows:
            train_features = np.concatenate(train_rows, axis=0)
        else:
            raise ValueError(
                "No training trajectories found; refusing to fit scaler on all data."
            )

        self._scaler.fit(train_features)
        self._features = self._scaler.transform(features_raw)

        # Velocity scale from training trajectories for physics loss normalization.
        # Uses body-frame velocities since Fossen dynamics operates in body frame.
        train_vel_rows = []
        for start, end, split in traj_segments:
            if split == 'train':
                train_vel_rows.append(self._gt_body_vel[start:end])
        if train_vel_rows:
            self._vel_scale = float(np.std(np.concatenate(train_vel_rows)).clip(min=1e-6))
        else:
            self._vel_scale = float(np.std(self._gt_body_vel).clip(min=1e-6))

        # Build per-split window indices (no cross-boundary windows).
        self._train_indices = []
        self._val_indices = []
        self._test_indices = []

        for start, end, split in traj_segments:
            seg_len = end - start
            if seg_len < window:
                continue
            max_win_start = end - window
            seg_indices = list(range(start, max_win_start + 1, self._config.stride))

            if split == 'train':
                self._train_indices.extend(seg_indices)
            elif split == 'val':
                self._val_indices.extend(seg_indices)
            else:
                self._test_indices.extend(seg_indices)

        if len(self._val_indices) == 0:
            warnings.warn("Validation set is empty after trajectory-aware splitting.")
        if len(self._test_indices) == 0:
            warnings.warn("Test set is empty after trajectory-aware splitting.")

    def _compute_mean_dt(self, gt_df: pd.DataFrame) -> float:
        """Compute robust mean sampling interval."""
        if 't' not in gt_df.columns:
            warnings.warn("No 't' column found, using default dt=0.1s")
            return 0.1

        t_arr = gt_df['t'].values
        dt_arr = np.diff(t_arr)
        positive_dt = dt_arr[dt_arr > 1e-6]

        if len(positive_dt) == 0:
            return 0.1

        return float(np.median(positive_dt))

    def _fill_missing_features(self, cor_df: pd.DataFrame) -> NDArray[np.float32]:
        """Forward-fill then zero-pad missing feature values.

        Strategy:
            1. Forward fill (ffill) to propagate last valid observation
            2. Zero-fill remaining NaNs (typically at sequence start)
            3. Log warning if >10% of data required filling

        v2.1 — Critical fix: When multi-trajectory data is concatenated,
        a global ffill() leaks the last valid value from trajectory N into
        the first NaN frames of trajectory N+1.  This is now prevented by
        grouping by trajectory_id before ffilling, so the fill operation
        never crosses trajectory boundaries.
        """
        feature_df = cor_df[list(self._config.feature_cols)].copy()

        nan_ratio = feature_df.isna().sum().sum() / feature_df.size
        if nan_ratio > 0.1:
            warnings.warn(
                f"High NaN ratio in features: {nan_ratio:.1%}. "
                "Consider investigating data quality."
            )

        if 'trajectory_id' in cor_df.columns:
            # Per-trajectory ffill prevents cross-boundary leakage.
            traj_ids = cor_df['trajectory_id']
            filled = feature_df.groupby(traj_ids, sort=False).ffill().fillna(0.0)
        else:
            filled = feature_df.ffill().fillna(0.0)

        return filled.values.astype(np.float32)

    def _compute_split_indices(
        self,
        n_total: int,
        n_train_raw: int,
        n_val_raw: int
    ) -> None:
        """Compute window indices for each split with gap isolation.

        Gap Calculation:
            gap = gap_multiplier * (seq_len + pred_len)

        This ensures no validation input window can overlap with
        training target windows, and vice versa.
        """
        seq_len = self._config.seq_len
        pred_len = self._config.pred_len
        gap = int(self._config.gap_multiplier * (seq_len + pred_len))

        max_window_start = n_total - seq_len - pred_len

        train_end_idx = n_train_raw - seq_len - pred_len
        self._train_indices = list(
            range(0, max(0, train_end_idx + 1), self._config.stride)
        )

        val_start_raw = n_train_raw
        val_end_raw = n_train_raw + n_val_raw

        val_start_idx = val_start_raw - seq_len + gap
        val_end_idx = val_end_raw - seq_len - pred_len

        self._val_indices = [
            i for i in range(val_start_idx, val_end_idx + 1)
            if 0 <= i <= max_window_start
            and (i - val_start_idx) % self._config.stride == 0
        ]

        test_start_raw = val_end_raw
        test_start_idx = test_start_raw - seq_len + gap

        self._test_indices = [
            i for i in range(test_start_idx, max_window_start + 1)
            if 0 <= i <= max_window_start
            and (i - test_start_idx) % self._config.stride == 0
        ]

        if len(self._val_indices) == 0:
            warnings.warn(
                "Validation set is empty after gap isolation. "
                "Consider reducing gap_multiplier or increasing data size."
            )
        if len(self._test_indices) == 0:
            warnings.warn(
                "Test set is empty after gap isolation. "
                "Consider reducing gap_multiplier or increasing data size."
            )

    def _create_dataset(self, indices: List[int]) -> AUVSlidingWindowDataset:
        """Create dataset for given window indices."""
        return AUVSlidingWindowDataset(
            features=self._features,
            validity=self._validity,
            gt_pos=self._gt_pos,
            gt_vel=self._gt_vel,
            gt_thrust=self._gt_thrust,
            gt_attitude=self._gt_attitude,
            anchor_pos=self._anchor_pos,
            anchor_vel=self._anchor_vel,
            anchor_attitude=self._anchor_attitude,
            anchor_thrust=self._anchor_thrust,
            anchor_valid=self._anchor_valid,
            anchor_lag=self._anchor_lag,
            mean_dt=self._mean_dt,
            seq_len=self._config.seq_len,
            pred_len=self._config.pred_len,
            valid_indices=indices,
            gt_body_vel=self._gt_body_vel,
        )

    def get_dataloaders(
        self,
        num_workers: int = 0,
        pin_memory: bool = False
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test DataLoaders.

        Args:
            num_workers: Number of worker processes for data loading.
            pin_memory: Whether to pin memory for GPU transfer.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        train_ds = self._create_dataset(self._train_indices)
        val_ds = self._create_dataset(self._val_indices)
        test_ds = self._create_dataset(self._test_indices)

        train_loader = DataLoader(
            train_ds,
            batch_size=self._config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self._config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=self._config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        print(f"[AUVDataPipeline] Split statistics:")
        print(f"  Train windows: {len(train_ds)}")
        print(f"  Val windows:   {len(val_ds)}")
        print(f"  Test windows:  {len(test_ds)}")
        print(f"  Mean dt:       {self._mean_dt:.4f}s")
        print(f"  Scaler μ:      {self._scaler.mean[:3]}... (first 3 features)")
        print(f"  Scaler σ:      {self._scaler.std[:3]}... (first 3 features)")

        return train_loader, val_loader, test_loader


def get_dataloaders(
    gt_path: Union[str, Path],
    cor_path: Union[str, Path],
    cfg: DatasetConfig
) -> Tuple[DataLoader, DataLoader, float, StandardScaler]:
    """Legacy-compatible interface for creating dataloaders.

    This function provides backward compatibility with existing training code
    while using the new leakage-free pipeline internally.

    Args:
        gt_path: Path to ground truth CSV.
        cor_path: Path to corrupted observations CSV.
        cfg: Dataset configuration object.

    Returns:
        Tuple of (train_loader, val_loader, mean_dt, scaler).

    Note:
        For new code, prefer using AUVDataPipeline directly for access
        to test_loader and more detailed statistics.
    """
    pipeline = AUVDataPipeline(gt_path, cor_path, cfg)
    train_loader, val_loader, _ = pipeline.get_dataloaders()

    return train_loader, val_loader, pipeline.mean_dt, pipeline.scaler
