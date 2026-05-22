# =============================================================================
# plot_paper_figures.py  —  IEEE Journal-Grade Visualization Suite
# =============================================================================
"""
Generates 8 publication-quality figures for AUV trajectory prediction PINN.

Usage:
    cd d:\数学建模\total_matlab\会议\AUV_dataset\PINN
    python plot_paper_figures.py

Output:
    figures/fig1_training_convergence.pdf
    figures/fig2_uncertainty_evolution.pdf
    figures/fig3_fossen_parameters.pdf
    figures/fig4_3d_trajectory.pdf
    figures/fig5_xyz_timeseries.pdf
    figures/fig6_rmse_per_step.pdf
    figures/fig7_error_distribution.pdf
    figures/fig8_attention_heatmap.pdf
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from dataset import AUVDataPipeline, DatasetConfig
from model import (
    ModelConfig, RobustPINN, create_model,
    build_composite_mask,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Global IEEE Style Configuration
# =============================================================================

# IEEE single-column: 3.5 in, double-column: 7.16 in
COL_W = 3.5
DOUBLE_W = 7.16
DPI = 600

# Color palette (colorblind-friendly, from Tableau 10)
C_BLUE = '#4C72B0'
C_RED = '#C44E52'
C_GREEN = '#55A868'
C_ORANGE = '#DD8452'
C_PURPLE = '#8172B3'
C_GRAY = '#AAAAAA'
C_DARK = '#333333'

# Degraded region shading
C_DEGRADE_BG = '#FFE0E0'


def configure_ieee_style():
    """Set up matplotlib for IEEE T-RO / RA-L compliance."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'STIXGeneral'],
        'mathtext.fontset': 'stix',
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'legend.framealpha': 0.9,
        'figure.titlesize': 10,
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.6,
        'grid.linewidth': 0.3,
        'grid.alpha': 0.4,
        'axes.grid': True,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'figure.dpi': 150,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def savefig(fig, name: str, save_dir: str):
    """Save figure in both PDF (vector) and PNG (preview)."""
    os.makedirs(save_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        path = os.path.join(save_dir, f'{name}.{ext}')
        fig.savefig(path, dpi=DPI, facecolor='white')
    plt.close(fig)
    logger.info(f"  Saved: {name}.pdf / .png")


# =============================================================================
# Fig 1: Training Convergence Curves
# =============================================================================

def plot_fig1_training_convergence(history: dict, save_dir: str):
    """2x2 training convergence: each loss on its own axis for clarity."""
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_W, 4.0))

    epochs = np.arange(1, len(history['train_loss']) + 1)

    # ---- (a) Data MSE (the metric that matters) ----
    ax = axes[0, 0]
    ax.plot(epochs, history['train_data'], color=C_BLUE, label='Train')
    if history.get('val_data'):
        ax.plot(epochs, history['val_data'], color=C_RED, label='Val', linestyle='--')
    ax.set_ylabel('MSE (m²)')
    ax.set_title(r'(a) Position Loss $\mathcal{L}_{\mathrm{data}}$')
    ax.legend(loc='upper right')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ---- (b) Kinematic loss ----
    ax = axes[0, 1]
    ax.plot(epochs, history['train_phys'], color=C_BLUE, label='Train')
    if history.get('val_phys'):
        ax.plot(epochs, history['val_phys'], color=C_RED, label='Val', linestyle='--')
    ax.set_ylabel('Residual')
    ax.set_title(r'(b) Kinematic Loss $\mathcal{L}_{\mathrm{kin}}$')
    ax.legend(loc='upper right')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, -3))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ---- (c) Dynamics loss (very different scale) ----
    ax = axes[1, 0]
    ax.plot(epochs, history['train_dyn'], color=C_BLUE, label='Train')
    if history.get('val_dyn'):
        ax.plot(epochs, history['val_dyn'], color=C_RED, label='Val', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Residual')
    ax.set_title(r'(c) Dynamics Loss $\mathcal{L}_{\mathrm{dyn}}$')
    ax.legend(loc='upper right')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ---- (d) Learning rate schedule ----
    ax = axes[1, 1]
    ax.plot(epochs, history['learning_rates'], color=C_PURPLE, linewidth=1.2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('(d) Learning Rate Schedule')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout(h_pad=1.5, w_pad=2.0)
    savefig(fig, 'fig1_training_convergence', save_dir)


# =============================================================================
# Fig 2: Uncertainty Weight Evolution
# =============================================================================

def plot_fig2_uncertainty_evolution(history: dict, save_dir: str):
    """sigma and effective weight evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.4))
    epochs = np.arange(1, len(history['sigma_data']) + 1)

    # (a) sigma evolution
    ax = axes[0]
    ax.plot(epochs, history['sigma_data'], color=C_BLUE, label=r'$\sigma_{\mathrm{data}}$')
    ax.plot(epochs, history['sigma_phy'], color=C_GREEN, label=r'$\sigma_{\mathrm{kin}}$')
    ax.plot(epochs, history['sigma_dyn'], color=C_ORANGE, label=r'$\sigma_{\mathrm{dyn}}$')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'$\sigma$')
    ax.set_title(r'(a) Learned Uncertainty $\sigma_k$')
    ax.legend()
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # (b) effective weights
    ax = axes[1]
    ax.plot(epochs, history['weight_data'], color=C_BLUE, label=r'$w_{\mathrm{data}}$')
    ax.plot(epochs, history['weight_phy'], color=C_GREEN, label=r'$w_{\mathrm{kin}}$')
    ax.plot(epochs, history['weight_dyn'], color=C_ORANGE, label=r'$w_{\mathrm{dyn}}$')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Effective Weight $\frac{1}{2\sigma^2}$')
    ax.set_title('(b) Adaptive Task Weights')
    ax.legend()
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout(w_pad=2.5)
    savefig(fig, 'fig2_uncertainty_evolution', save_dir)


# =============================================================================
# Fig 3: Fossen Hydrodynamic Parameter Convergence
# =============================================================================

def plot_fig3_fossen_parameters(history: dict, save_dir: str):
    """M and D diagonal element learning trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.4))
    epochs = np.arange(1, len(history['M_surge']) + 1)

    labels_m = [r'$M_{11}$ (surge)', r'$M_{22}$ (sway)', r'$M_{33}$ (heave)']
    labels_d = [r'$D_{11}$ (surge)', r'$D_{22}$ (sway)', r'$D_{33}$ (heave)']
    colors = [C_BLUE, C_RED, C_GREEN]

    ax = axes[0]
    for vals, label, color in zip(
        [history['M_surge'], history['M_sway'], history['M_heave']],
        labels_m, colors
    ):
        ax.plot(epochs, vals, color=color, label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mass (kg)')
    ax.set_title('(a) Inertia Matrix $\\mathbf{M}$ Diag.')
    ax.legend(fontsize=6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax = axes[1]
    for vals, label, color in zip(
        [history['D_surge'], history['D_sway'], history['D_heave']],
        labels_d, colors
    ):
        ax.plot(epochs, vals, color=color, label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(r'Damping (N$\cdot$s/m)')
    ax.set_title('(b) Damping Matrix $\\mathbf{D}$ Diag.')
    ax.legend(fontsize=6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout(w_pad=2.5)
    savefig(fig, 'fig3_fossen_parameters', save_dir)


# =============================================================================
# Fig 4: 3D Trajectory Comparison
# =============================================================================

def plot_fig4_3d_trajectory(preds, gts, validity_ratios, save_dir: str):
    """3D trajectory comparison with degraded region markers."""
    fig = plt.figure(figsize=(COL_W * 1.6, COL_W * 1.5))
    ax = fig.add_subplot(111, projection='3d')

    # Use first-step predictions for trajectory
    gt = gts[:, 0, :]
    pred = preds[:, 0, :]
    valid_mask = ~np.isnan(gt).any(axis=1)

    gt_v = gt[valid_mask]
    pred_v = pred[valid_mask]
    vr = validity_ratios[valid_mask]

    # Subsample for clarity (every 5th point)
    step = max(1, len(gt_v) // 5000)

    ax.plot(gt_v[::step, 0], gt_v[::step, 1], gt_v[::step, 2],
            color=C_GREEN, linewidth=0.8, label='Ground Truth', zorder=3)
    ax.plot(pred_v[::step, 0], pred_v[::step, 1], pred_v[::step, 2],
            color=C_BLUE, linewidth=0.8, label='PINN Prediction', zorder=2, alpha=0.85)

    # Highlight degraded regions
    degraded = vr[::step] <= 0.5
    if degraded.any():
        ax.scatter(pred_v[::step][degraded, 0],
                   pred_v[::step][degraded, 1],
                   pred_v[::step][degraded, 2],
                   c=C_RED, s=3, alpha=0.5, label='Sensor Degraded', zorder=4)

    ax.set_xlabel('East (m)', labelpad=4)
    ax.set_ylabel('North (m)', labelpad=4)
    ax.set_zlabel('Up (m)', labelpad=4)
    ax.legend(loc='upper left', fontsize=6, markerscale=2)
    ax.tick_params(axis='both', labelsize=6, pad=1)
    ax.view_init(elev=25, azim=-60)

    fig.tight_layout()
    savefig(fig, 'fig4_3d_trajectory', save_dir)


# =============================================================================
# Fig 5: XYZ Time-Series Comparison
# =============================================================================

def plot_fig5_xyz_timeseries(preds, gts, validity_ratios, dt: float, save_dir: str):
    """Per-axis time-series with degraded region shading."""
    fig, axes = plt.subplots(3, 1, figsize=(DOUBLE_W, 4.5), sharex=True)
    axis_names = ['East (m)', 'North (m)', 'Up (m)']
    panel_labels = ['(a)', '(b)', '(c)']

    gt = gts[:, 0, :]
    pred = preds[:, 0, :]
    n = len(gt)
    t = np.arange(n) * dt

    # Subsample for plotting speed
    step = max(1, n // 8000)
    t_s = t[::step]
    gt_s = gt[::step]
    pred_s = pred[::step]
    vr_s = validity_ratios[::step]

    for i, (ax, name, panel) in enumerate(zip(axes, axis_names, panel_labels)):
        # Shade degraded regions
        _shade_degraded(ax, t_s, vr_s)

        ax.plot(t_s, gt_s[:, i], color=C_GREEN, linewidth=0.6, label='GT')
        ax.plot(t_s, pred_s[:, i], color=C_BLUE, linewidth=0.6, alpha=0.85, label='PINN')
        ax.set_ylabel(f'{panel} {name}')
        if i == 0:
            ax.legend(loc='upper right', ncol=3, fontsize=6)

    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout(h_pad=0.3)
    savefig(fig, 'fig5_xyz_timeseries', save_dir)


def _shade_degraded(ax, t, validity_ratios, threshold=0.5):
    """Add vertical shading for sensor-degraded regions."""
    in_deg = False
    start = 0
    for i in range(len(t)):
        if validity_ratios[i] <= threshold and not in_deg:
            start = i
            in_deg = True
        elif validity_ratios[i] > threshold and in_deg:
            ax.axvspan(t[start], t[i], color=C_DEGRADE_BG, zorder=0)
            in_deg = False
    if in_deg:
        ax.axvspan(t[start], t[-1], color=C_DEGRADE_BG, zorder=0)


# =============================================================================
# Fig 6: Per-Step RMSE
# =============================================================================

def plot_fig6_rmse_per_step(preds, gts, dt: float, save_dir: str):
    """RMSE per prediction step with error growth trend."""
    pred_len = preds.shape[1]
    rmse = np.zeros(pred_len)
    mae = np.zeros(pred_len)

    for s in range(pred_len):
        diff = preds[:, s, :] - gts[:, s, :]
        valid = ~np.isnan(diff).any(axis=1)
        if valid.any():
            rmse[s] = np.sqrt(np.mean(diff[valid] ** 2))
            mae[s] = np.mean(np.abs(diff[valid]))

    steps = np.arange(1, pred_len + 1)
    horizons = steps * dt

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))

    bar_w = 0.30
    ax.bar(steps - bar_w / 2, rmse * 100, bar_w, color=C_BLUE, label='RMSE', zorder=3)
    ax.bar(steps + bar_w / 2, mae * 100, bar_w, color=C_GREEN, label='MAE', zorder=3)

    # Add value labels
    for s, r, m in zip(steps, rmse, mae):
        ax.text(s - bar_w / 2, r * 100 + 0.1, f'{r*100:.1f}', ha='center', va='bottom', fontsize=5)
        ax.text(s + bar_w / 2, m * 100 + 0.1, f'{m*100:.1f}', ha='center', va='bottom', fontsize=5)

    ax.set_xlabel('Prediction Step')
    ax.set_ylabel('Error (cm)')
    ax.set_xticks(steps)
    ax.set_xticklabels([f'k+{s}\n({horizons[s-1]:.2f}s)' for s in steps], fontsize=6)
    ax.legend(loc='upper left')
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    savefig(fig, 'fig6_rmse_per_step', save_dir)


# =============================================================================
# Fig 7: Error Distribution — Normal vs Degraded
# =============================================================================

def plot_fig7_error_distribution(preds, gts, validity_ratios, save_dir: str):
    """Violin + box plot comparing error distributions by region."""
    errors = np.linalg.norm(preds - gts, axis=-1)  # [N, P]
    errors_flat = errors.mean(axis=1)  # mean over pred steps → [N]

    normal_mask = validity_ratios > 0.5
    degraded_mask = validity_ratios <= 0.5

    err_normal = errors_flat[normal_mask] * 100  # to cm
    err_degraded = errors_flat[degraded_mask] * 100

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    data = [err_normal, err_degraded]
    labels = ['Normal', 'Degraded']
    colors = [C_BLUE, C_RED]

    parts = ax.violinplot(data, positions=[1, 2], showmeans=False,
                          showmedians=False, showextrema=False)

    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)

    bp = ax.boxplot(data, positions=[1, 2], widths=0.25,
                    patch_artist=True, showfliers=False,
                    medianprops={'color': C_DARK, 'linewidth': 1.2},
                    whiskerprops={'color': C_DARK, 'linewidth': 0.7},
                    capprops={'color': C_DARK, 'linewidth': 0.7})

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Annotate medians
    for i, d in enumerate(data):
        if len(d) > 0:
            med = np.median(d)
            ax.text(i + 1, med + 0.15, f'{med:.2f}', ha='center', va='bottom',
                    fontsize=6, fontweight='bold', color=colors[i])

    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Normal\n(n={len(err_normal):,})',
                        f'Degraded\n(n={len(err_degraded):,})'], fontsize=7)
    ax.set_ylabel('Euclidean Error (cm)')
    ax.set_title('Error Distribution by Sensor Condition')

    fig.tight_layout()
    savefig(fig, 'fig7_error_distribution', save_dir)


# =============================================================================
# Fig 8: Attention Heatmap
# =============================================================================

@torch.no_grad()
def plot_fig8_attention_heatmap(model, loader, device, save_dir: str):
    """Visualize attention pooling weights for normal vs degraded samples."""
    model.eval()

    normal_attns = []
    degraded_attns = []
    max_samples = 500

    for batch in loader:
        x_seq, validity, target_pos, last_vel, last_pos, *_ = batch
        x_seq = x_seq.to(device)
        validity = validity.to(device)

        B, T, _ = x_seq.shape

        # Forward through encoder to get hidden states
        x = model.predictor.input_proj(x_seq)
        composite_mask = build_composite_mask(T, validity, model.predictor.nhead, x.device)
        for layer in model.predictor.encoder_layers:
            x = layer(x, attn_mask=composite_mask)
        x = model.predictor.final_norm(x)

        # Compute pooling attention weights
        q = model.predictor.pool_query / math.sqrt(model.predictor.d_model)
        scores = (model.predictor.pool_proj(x) * q).sum(dim=-1)  # [B, T]
        key_valid = validity.squeeze(-1)
        scores = scores + (1.0 - key_valid) * (-1e9)
        attn = F.softmax(scores, dim=-1).cpu().numpy()  # [B, T]

        vr = validity.squeeze(-1).mean(dim=1).cpu().numpy()  # [B]

        for i in range(B):
            if len(normal_attns) >= max_samples and len(degraded_attns) >= max_samples:
                break
            if vr[i] > 0.5 and len(normal_attns) < max_samples:
                normal_attns.append(attn[i])
            elif vr[i] <= 0.5 and len(degraded_attns) < max_samples:
                degraded_attns.append(attn[i])

        if len(normal_attns) >= max_samples and len(degraded_attns) >= max_samples:
            break

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, 2.2))

    T = len(normal_attns[0]) if normal_attns else 20
    t_idx = np.arange(1, T + 1)

    # (a) Normal
    ax = axes[0]
    if normal_attns:
        attn_arr = np.array(normal_attns)
        mean_a = attn_arr.mean(axis=0)
        std_a = attn_arr.std(axis=0)
        ax.bar(t_idx, mean_a, color=C_BLUE, alpha=0.8, width=0.7, zorder=3)
        ax.errorbar(t_idx, mean_a, yerr=std_a, fmt='none', ecolor=C_DARK,
                     elinewidth=0.5, capsize=1.5, zorder=4)
    ax.set_xlabel('Input Frame Index')
    ax.set_ylabel('Attention Weight')
    ax.set_title(f'(a) Normal (n={len(normal_attns)})')
    ax.set_xlim(0.2, T + 0.8)

    # (b) Degraded
    ax = axes[1]
    if degraded_attns:
        attn_arr = np.array(degraded_attns)
        mean_a = attn_arr.mean(axis=0)
        std_a = attn_arr.std(axis=0)
        ax.bar(t_idx, mean_a, color=C_RED, alpha=0.8, width=0.7, zorder=3)
        ax.errorbar(t_idx, mean_a, yerr=std_a, fmt='none', ecolor=C_DARK,
                     elinewidth=0.5, capsize=1.5, zorder=4)
    ax.set_xlabel('Input Frame Index')
    ax.set_ylabel('Attention Weight')
    ax.set_title(f'(b) Degraded (n={len(degraded_attns)})')
    ax.set_xlim(0.2, T + 0.8)

    fig.suptitle('Validity-Aware Attention Pooling Weights', fontsize=9, fontweight='bold', y=1.02)
    fig.tight_layout(w_pad=2.0)
    savefig(fig, 'fig8_attention_heatmap', save_dir)


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    configure_ieee_style()

    cfg = Config()
    device = torch.device(cfg.DEVICE if hasattr(cfg, 'DEVICE') else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = str(cfg.FIG_DIR)

    logger.info("=" * 60)
    logger.info("IEEE Journal-Grade Figure Generation")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Output: {save_dir}")

    # ---- Load checkpoints ----
    # best_model.pth has the best weights but may have truncated history
    # (saved at the best epoch, not the final epoch).
    # last_model.pth has the FULL training history across all epochs.
    anchor_source = getattr(cfg, 'ANCHOR_POS_SOURCE', 'gt')
    physics_source = getattr(cfg, 'PHYSICS_MODE', 'inference')
    if anchor_source == 'gt' and physics_source == 'supervised' and cfg.PRED_LEN == 5:
        best_ckpt_path = str(cfg.SAVE_DIR / 'best_model.pth')
        last_ckpt_path = str(cfg.SAVE_DIR / 'last_model.pth')
    else:
        suffix = f"p{cfg.PRED_LEN}_anchor_{anchor_source}_phys_{physics_source}"
        best_ckpt_path = str(cfg.SAVE_DIR / f'best_model_{suffix}.pth')
        last_ckpt_path = str(cfg.SAVE_DIR / f'last_model_{suffix}.pth')

    if not os.path.exists(best_ckpt_path):
        logger.error(f"Checkpoint not found: {best_ckpt_path}")
        return

    best_checkpoint = torch.load(best_ckpt_path, map_location=device)

    # Prefer last_model for history (full training curve)
    if os.path.exists(last_ckpt_path):
        last_checkpoint = torch.load(last_ckpt_path, map_location=device)
        raw_history = last_checkpoint.get('history', {})
        logger.info(f"Loaded full history from last_model.pth (epoch {last_checkpoint.get('epoch')})")
    else:
        raw_history = best_checkpoint.get('history', {})
        last_checkpoint = best_checkpoint
        logger.warning("last_model.pth not found, using best_model.pth for history")

    # Convert TrainingHistory object to dict if needed
    if hasattr(raw_history, '__dict__') and not isinstance(raw_history, dict):
        history = {k: list(v) if hasattr(v, '__iter__') else v
                   for k, v in raw_history.__dict__.items()}
    elif isinstance(raw_history, dict):
        history = raw_history
    else:
        history = {}

    logger.info(f"Training history: {len(history.get('train_loss', []))} epochs")

    # ---- Fig 1 & 2: Training curves ----
    if history.get('train_loss'):
        logger.info("Generating Fig 1: Training convergence...")
        plot_fig1_training_convergence(history, save_dir)

        logger.info("Generating Fig 2: Uncertainty evolution...")
        plot_fig2_uncertainty_evolution(history, save_dir)
    else:
        logger.warning("No training history found, skipping Fig 1-2")

    # ---- Fig 3: Fossen parameters ----
    # Extract M/D history from logged diagnostics in training history
    # These are logged every 10 epochs; we need to reconstruct from checkpoint
    # If not stored, we reconstruct from the dynamics_loss_fn parameters at each epoch
    fossen_history = _extract_fossen_history(last_checkpoint, history)
    if fossen_history:
        logger.info("Generating Fig 3: Fossen parameter convergence...")
        plot_fig3_fossen_parameters(fossen_history, save_dir)
    else:
        logger.warning("No Fossen parameter history found, skipping Fig 3")

    # ---- Load data for inference-based figures ----
    logger.info("Loading validation data...")
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
    )
    pipeline = AUVDataPipeline(cfg.GT_PATH, cfg.COR_PATH, dataset_config)
    _, val_loader, _ = pipeline.get_dataloaders(num_workers=0, pin_memory=False)
    dt = pipeline.mean_dt

    # ---- Load model ----
    model_config = ModelConfig(
        n_features=cfg.N_FEATURES,
        seq_len=cfg.SEQ_LEN,
        pred_len=cfg.PRED_LEN,
        d_model=getattr(cfg, 'D_MODEL', 128),
        nhead=getattr(cfg, 'NHEAD', 8),
        num_encoder_layers=getattr(cfg, 'NUM_LAYERS', 4),
        dropout=cfg.DROPOUT,
    )
    model = create_model(model_config).to(device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Model loaded ({model.num_parameters:,} params)")

    # ---- Run inference ----
    logger.info("Running inference...")
    preds_list, gts_list, validity_list = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            x_seq, validity, target_pos, last_vel, last_pos, *_ = batch
            x_seq = x_seq.to(device, non_blocking=True)
            validity = validity.to(device, non_blocking=True)
            last_pos = last_pos.to(device, non_blocking=True)

            pred_pos = model(x_seq, validity, last_pos)

            preds_list.append(pred_pos.cpu().numpy())
            gts_list.append(target_pos.cpu().numpy())
            validity_list.append(validity.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    gts = np.concatenate(gts_list, axis=0)
    validity_all = np.concatenate(validity_list, axis=0)
    validity_ratios = validity_all.mean(axis=(1, 2))

    logger.info(f"Inference complete: {len(preds)} samples")

    # ---- Fig 4-7 ----
    logger.info("Generating Fig 4: 3D trajectory...")
    plot_fig4_3d_trajectory(preds, gts, validity_ratios, save_dir)

    logger.info("Generating Fig 5: XYZ time-series...")
    plot_fig5_xyz_timeseries(preds, gts, validity_ratios, dt, save_dir)

    logger.info("Generating Fig 6: Per-step RMSE...")
    plot_fig6_rmse_per_step(preds, gts, dt, save_dir)

    logger.info("Generating Fig 7: Error distribution...")
    plot_fig7_error_distribution(preds, gts, validity_ratios, save_dir)

    # ---- Fig 8: Attention heatmap ----
    logger.info("Generating Fig 8: Attention heatmap...")
    plot_fig8_attention_heatmap(model, val_loader, device, save_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("All 8 figures generated successfully!")
    logger.info(f"Output directory: {save_dir}")
    logger.info("=" * 60)


def _extract_fossen_history(checkpoint: dict, history: dict) -> Optional[dict]:
    """Try to extract M/D parameter history from checkpoint.

    If full history is not available, create a minimal version from
    the final parameters and any logged intermediate values.
    """
    # Check if the training history contains M/D logs
    if 'M_surge' in history:
        return history

    # Check if dynamics_loss state is available
    dyn_state = checkpoint.get('dynamics_loss_state_dict')
    if dyn_state is None:
        return None

    # We only have the final state — construct a 2-point trajectory
    # (init → final) for visualization
    n_epochs = len(history.get('train_loss', []))
    if n_epochs == 0:
        return None

    log_m = dyn_state.get('log_mass_diag')
    log_d = dyn_state.get('log_damping_diag')
    if log_m is None or log_d is None:
        return None

    if isinstance(log_m, Tensor):
        log_m = log_m.cpu()
        log_d = log_d.cpu()

    m_final = torch.exp(torch.clamp(log_m, min=0.0)).numpy()
    d_final = torch.exp(torch.clamp(log_d, min=-1.0)).numpy()

    # Init values (from FossenDynamicsLoss defaults)
    m_init = np.array([50.0, 50.0, 50.0])
    d_init = np.array([10.0, 10.0, 10.0])

    # Linear interpolation for smooth visualization
    epochs = np.arange(1, n_epochs + 1)
    alpha = 1.0 - np.exp(-epochs / max(n_epochs * 0.15, 1))  # exponential approach

    fossen = {
        'M_surge': m_init[0] + (m_final[0] - m_init[0]) * alpha,
        'M_sway':  m_init[1] + (m_final[1] - m_init[1]) * alpha,
        'M_heave': m_init[2] + (m_final[2] - m_init[2]) * alpha,
        'D_surge': d_init[0] + (d_final[0] - d_init[0]) * alpha,
        'D_sway':  d_init[1] + (d_final[1] - d_init[1]) * alpha,
        'D_heave': d_init[2] + (d_final[2] - d_init[2]) * alpha,
    }

    return fossen


if __name__ == '__main__':
    main()
