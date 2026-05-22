"""
=============================================================================
prepare_data.py  —  MSS Simulation Data → PINN System Data Pipeline
=============================================================================
将 MSS (Marine Systems Simulator) 6-DOF AUV 仿真数据转换为 Robust PINN
系统所需的训练数据格式。

数据流:
    MSS CSV (19 cols, body-frame velocities)
         │
         ├─ 1. Column renaming
         ├─ 2. Body→NED velocity rotation via Euler angles
         ├─ 3. Acceleration derivation (Savitzky-Golay differentiation)
         ├─ 4. Sensor degradation injection (DVL dropout + IMU impulse)
         │
         └─→ ground_truth.csv (16 cols) + corrupted_data.csv (18 cols)

数学公式:
    Body→NED 旋转 (ZYX 欧拉角):
        R = R_z(yaw) · R_y(pitch) · R_x(roll)
        [vn, ve, vd]^T = R · [u, v, w]^T

    加速度数值微分 (Savitzky-Golay 5阶平滑):
        a_t = SavGol(v_{t-2:t+2}, deriv=1) / dt

    脉冲噪声幅值标定:
        impulse[t] = signal[t] + sign × magnitude × σ(signal)

约定:
    - 坐标系: NED (North-East-Down), z轴 Down-positive (深度)
    - 单位: 长度=米, 角度=弧度, 时间=秒
=============================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import savgol_filter


# =============================================================================
# Configuration
# =============================================================================

MSS_INPUT_PATH = Path(__file__).resolve().parent / "auv_simulation_data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# 传感器退化参数
RANDOM_SEED = 42
DVL_DROPOUT_RATIO = 0.06       # 6% 的时间段 DVL 失锁 (声学遮挡)
IMU_IMPULSE_PROB = 0.03        # 3% 的时刻存在 IMU 脉冲噪声
IMU_IMPULSE_MAGNITUDE = 8.0    # 脉冲幅值: 8倍 σ
GAUSSIAN_NOISE_STD_POS = 0.3   # 位置背景噪声 σ (米)
GAUSSIAN_NOISE_STD_VEL = 0.05  # 速度背景噪声 σ (m/s)
GAUSSIAN_NOISE_STD_ATT = 0.005 # 姿态角背景噪声 σ (rad ≈ 0.3°)

# 训练/验证集分割比例 (用于确保两区域都有缺失段)
TRAIN_RATIO = 0.7


# =============================================================================
# Coordinate Transform
# =============================================================================

def euler_to_rotation_matrix(
    roll: NDArray[np.float64],
    pitch: NDArray[np.float64],
    yaw: NDArray[np.float64]
) -> NDArray[np.float64]:
    """构造 ZYX 欧拉角旋转矩阵 (Body → NED)。

    数学公式:
        R = R_z(ψ) · R_y(θ) · R_x(φ)

    Args:
        roll:  横滚角 φ, shape [N]
        pitch: 俯仰角 θ, shape [N]
        yaw:   偏航角 ψ, shape [N]

    Returns:
        R: 旋转矩阵, shape [N, 3, 3]
    """
    cphi, sphi = np.cos(roll), np.sin(roll)
    cth,  sth  = np.cos(pitch), np.sin(pitch)
    cpsi, spsi = np.cos(yaw), np.sin(yaw)

    n = len(roll)
    R = np.zeros((n, 3, 3), dtype=np.float64)

    # ZYX 旋转矩阵 (展开形式)
    R[:, 0, 0] = cpsi * cth
    R[:, 0, 1] = cpsi * sth * sphi - spsi * cphi
    R[:, 0, 2] = cpsi * sth * cphi + spsi * sphi

    R[:, 1, 0] = spsi * cth
    R[:, 1, 1] = spsi * sth * sphi + cpsi * cphi
    R[:, 1, 2] = spsi * sth * cphi - cpsi * sphi

    R[:, 2, 0] = -sth
    R[:, 2, 1] = cth * sphi
    R[:, 2, 2] = cth * cphi

    return R


def transform_body_to_ned(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    w: NDArray[np.float64],
    roll: NDArray[np.float64],
    pitch: NDArray[np.float64],
    yaw: NDArray[np.float64]
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """将体坐标系速度旋转至 NED 导航坐标系。

    Args:
        u, v, w: 体坐标系速度 (surge, sway, heave), shape [N]
        roll, pitch, yaw: 欧拉角, shape [N]

    Returns:
        (vn, ve, vd): NED 坐标系速度, 各 shape [N]
                     vn=North, ve=East, vd=Down (Down-positive)
    """
    R = euler_to_rotation_matrix(roll, pitch, yaw)         # [N, 3, 3]
    vel_body = np.stack([u, v, w], axis=-1)[..., np.newaxis]  # [N, 3, 1]
    vel_ned = (R @ vel_body).squeeze(-1)                    # [N, 3]

    return vel_ned[:, 0], vel_ned[:, 1], vel_ned[:, 2]


# =============================================================================
# Acceleration Derivation
# =============================================================================

def compute_body_accelerations(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    w: NDArray[np.float64],
    dt: float,
    window_length: int = 11,
    polyorder: int = 3
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """通过 Savitzky-Golay 滤波数值微分计算体坐标系加速度。

    优势 over 朴素有限差分:
        - 抑制高频噪声 (适合后续注入合成噪声)
        - 保留导数信号的物理特性
        - 边界处理鲁棒

    Args:
        u, v, w: 体坐标系速度, shape [N]
        dt: 采样间隔 (秒)
        window_length: SG 窗口长度 (奇数)
        polyorder: 多项式阶数 (< window_length)

    Returns:
        (ax, ay, az): 体坐标系加速度, 各 shape [N]
    """
    ax = savgol_filter(u, window_length, polyorder, deriv=1, delta=dt)
    ay = savgol_filter(v, window_length, polyorder, deriv=1, delta=dt)
    az = savgol_filter(w, window_length, polyorder, deriv=1, delta=dt)

    return ax, ay, az


# =============================================================================
# Ground Truth Generation
# =============================================================================

def generate_ground_truth(mss_df: pd.DataFrame) -> pd.DataFrame:
    """从 MSS 仿真数据生成 PINN 系统所需的 ground truth DataFrame。

    输出 16 列:
        t, x, y, z, vn, ve, vu, roll, pitch, yaw, ax, ay, az, wx, wy, wz

    Args:
        mss_df: MSS 仿真数据 DataFrame

    Returns:
        gt_df: PINN 格式的 ground truth DataFrame
    """
    # 时间归一化 (从 0 开始)
    t = mss_df['time'].values - mss_df['time'].iloc[0]
    dt = float(np.median(np.diff(t)))

    # 位置 (NED, z = Down-positive)
    x = mss_df['North_m'].values.astype(np.float64)
    y = mss_df['East_m'].values.astype(np.float64)
    z = mss_df['Down_m'].values.astype(np.float64)

    # 体坐标系速度
    u = mss_df['u_ms'].values.astype(np.float64)
    v = mss_df['v_ms'].values.astype(np.float64)
    w = mss_df['w_ms'].values.astype(np.float64)

    # 姿态 (欧拉角)
    roll = mss_df['Roll_rad'].values.astype(np.float64)
    pitch = mss_df['Pitch_rad'].values.astype(np.float64)
    yaw = mss_df['Yaw_rad'].values.astype(np.float64)

    # 角速度
    wx = mss_df['p_rads'].values.astype(np.float64)
    wy = mss_df['q_rads'].values.astype(np.float64)
    wz = mss_df['r_rads'].values.astype(np.float64)

    # 速度坐标变换: Body → NED
    print(f"  [Transform] Body → NED 速度旋转...")
    vn, ve, vd = transform_body_to_ned(u, v, w, roll, pitch, yaw)
    # vu 列在系统中代表 z 方向速度，与 z 一致约定为 Down-positive
    vu = vd

    # 加速度数值微分 (Savitzky-Golay)
    print(f"  [Transform] Savitzky-Golay 加速度微分 (dt={dt:.4f}s)...")
    ax, ay, az = compute_body_accelerations(u, v, w, dt)

    # 组装 DataFrame (列顺序与 PINN 系统一致)
    gt_df = pd.DataFrame({
        't': t, 'x': x, 'y': y, 'z': z,
        'vn': vn, 've': ve, 'vu': vu,
        'roll': roll, 'pitch': pitch, 'yaw': yaw,
        'ax': ax, 'ay': ay, 'az': az,
        'wx': wx, 'wy': wy, 'wz': wz,
    })

    return gt_df


# =============================================================================
# Sensor Degradation Injection
# =============================================================================

def inject_dvl_dropout(
    n_samples: int,
    dropout_ratio: float,
    train_boundary: int,
    rng: np.random.Generator
) -> Tuple[NDArray[np.bool_], List[Tuple[int, int]]]:
    """生成 DVL 失锁 (连续 NaN 段) 掩码。

    确保训练区和验证区都包含失锁段以避免验证集为空。

    Args:
        n_samples: 总样本数
        dropout_ratio: 失锁占比 (~0.06)
        train_boundary: 训练/验证分割索引
        rng: 随机数生成器

    Returns:
        mask_missing: bool 数组, shape [N]
        gap_intervals: 每个失锁段的 (start, end) 列表
    """
    # 失锁段总数: ~每 1.5-3s 一段, 每段 30-80 个时刻 (1.5-4s @ 20Hz)
    total_missing = int(n_samples * dropout_ratio)
    avg_gap_size = 50          # 平均段长度
    num_gaps = max(2, total_missing // avg_gap_size)

    # 训练区 80%, 验证区 20%
    n_train_gaps = max(1, int(num_gaps * 0.75))
    n_val_gaps = max(1, num_gaps - n_train_gaps)

    mask = np.zeros(n_samples, dtype=bool)
    intervals: List[Tuple[int, int]] = []

    # 训练区段
    train_low, train_high = 100, train_boundary - avg_gap_size - 100
    train_starts = rng.choice(np.arange(train_low, train_high), n_train_gaps, replace=False)

    # 验证区段
    val_low, val_high = train_boundary + 50, n_samples - avg_gap_size - 50
    val_starts = rng.choice(np.arange(val_low, val_high), n_val_gaps, replace=False)

    all_starts = sorted(np.concatenate([train_starts, val_starts]))
    for gs in all_starts:
        # 段长度随机化 (30-80)
        gap_len = rng.integers(30, 80)
        ge = min(gs + gap_len, n_samples)
        mask[gs:ge] = True
        intervals.append((int(gs), int(ge)))

    return mask, intervals


def inject_impulse_noise(
    n_samples: int,
    impulse_prob: float,
    valid_mask: NDArray[np.bool_],
    rng: np.random.Generator
) -> NDArray[np.bool_]:
    """生成 IMU 脉冲噪声掩码。

    Args:
        n_samples: 总样本数
        impulse_prob: 脉冲触发概率 (~0.03)
        valid_mask: 非缺失点的 mask (避免重叠)
        rng: 随机数生成器

    Returns:
        mask_impulse: bool 数组, shape [N]
    """
    candidates = np.where(valid_mask)[0]
    n_impulse = int(len(candidates) * impulse_prob)
    impulse_idx = rng.choice(candidates, size=n_impulse, replace=False)

    mask = np.zeros(n_samples, dtype=bool)
    mask[impulse_idx] = True
    return mask


def generate_corrupted_data(
    gt_df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """生成劣化的观测数据 DataFrame (PINN 系统的输入)。

    劣化模式:
        1. DVL 失锁 → 位置和速度列置 NaN, is_missing=1
        2. IMU 脉冲噪声 → ax/ay/az 列大幅扰动, is_impulse=1
        3. 背景高斯噪声 → 所有传感器列叠加小幅噪声

    Args:
        gt_df: 干净的 ground truth DataFrame
        train_ratio: 训练集占比 (用于失锁段分布)

    Returns:
        cor_df: 劣化数据 DataFrame (含 is_missing, is_impulse 标志列)
        stats:  统计信息字典
    """
    rng = np.random.default_rng(RANDOM_SEED)
    cor_df = gt_df.copy()
    n = len(cor_df)
    train_boundary = int(n * train_ratio)

    # 1. DVL 失锁
    print(f"  [Inject] DVL 失锁 (ratio={DVL_DROPOUT_RATIO})...")
    mask_missing, intervals = inject_dvl_dropout(n, DVL_DROPOUT_RATIO, train_boundary, rng)

    # 失锁影响: 位置 + 速度全部置 NaN
    dropout_cols = ['x', 'y', 'z', 'vn', 've', 'vu']
    cor_df.loc[mask_missing, dropout_cols] = np.nan

    # 2. IMU 脉冲噪声
    print(f"  [Inject] IMU 脉冲噪声 (prob={IMU_IMPULSE_PROB})...")
    valid_for_impulse = ~mask_missing
    mask_impulse = inject_impulse_noise(n, IMU_IMPULSE_PROB, valid_for_impulse, rng)

    # 脉冲影响: ax/ay/az 列叠加大幅冲击
    for col in ['ax', 'ay', 'az']:
        sigma = float(np.nanstd(gt_df[col].values))
        signs = rng.choice([-1.0, 1.0], size=int(mask_impulse.sum()))
        cor_df.loc[mask_impulse, col] += signs * IMU_IMPULSE_MAGNITUDE * sigma

    # 3. 背景高斯噪声 (只施加于非失锁的正常点)
    print(f"  [Inject] 背景高斯噪声...")
    normal_mask = ~mask_missing
    n_normal = int(normal_mask.sum())

    # 位置噪声 (米)
    for col in ['x', 'y', 'z']:
        cor_df.loc[normal_mask, col] += rng.normal(0, GAUSSIAN_NOISE_STD_POS, n_normal)

    # 速度噪声 (m/s)
    for col in ['vn', 've', 'vu']:
        cor_df.loc[normal_mask, col] += rng.normal(0, GAUSSIAN_NOISE_STD_VEL, n_normal)

    # 姿态噪声 (rad)
    for col in ['roll', 'pitch', 'yaw']:
        cor_df.loc[normal_mask, col] += rng.normal(0, GAUSSIAN_NOISE_STD_ATT, n_normal)

    # 4. 写入劣化标志列
    cor_df['is_missing'] = mask_missing.astype(np.int32)
    cor_df['is_impulse'] = mask_impulse.astype(np.int32)

    stats = {
        'total': n,
        'n_missing': int(mask_missing.sum()),
        'n_impulse': int(mask_impulse.sum()),
        'n_normal': int(normal_mask.sum() - mask_impulse.sum()),
        'n_dropout_segments': len(intervals),
    }

    return cor_df, stats


# =============================================================================
# Visualization
# =============================================================================

def plot_data_summary(
    gt_df: pd.DataFrame,
    cor_df: pd.DataFrame,
    output_path: Path
) -> None:
    """绘制数据集预览图: 3D 轨迹 + 时序对比 + 退化区间标注。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('MSS Simulation → PINN Dataset (Ground Truth vs. Corrupted)',
                 fontsize=14, fontweight='bold')

    t = gt_df['t'].values
    mask_missing = cor_df['is_missing'].values.astype(bool)
    mask_impulse = cor_df['is_impulse'].values.astype(bool)

    # ── 3D 轨迹 ──
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    sc = ax.scatter(gt_df['x'], gt_df['y'], gt_df['z'],
                    c=t, cmap='plasma', s=1, alpha=0.7)
    ax.set_xlabel('North (m)')
    ax.set_ylabel('East (m)')
    ax.set_zlabel('Down (m)')
    ax.set_title('(a) Ground Truth Trajectory (3D)')
    ax.invert_zaxis()  # Down-positive: depth increases downward
    plt.colorbar(sc, ax=ax, label='Time (s)', shrink=0.6)

    # ── X(North) 时序 + 退化标注 ──
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(t, gt_df['x'].values, 'g-', linewidth=1.0, label='Ground Truth', zorder=3)
    ax.scatter(t[~mask_missing], cor_df['x'].values[~mask_missing],
               c='lightgray', s=2, alpha=0.4, label='Corrupted (observed)', zorder=2)
    # 失锁阴影
    in_drop = False
    drop_start = 0
    for i in range(len(mask_missing)):
        if mask_missing[i] and not in_drop:
            drop_start = i
            in_drop = True
        elif not mask_missing[i] and in_drop:
            ax.axvspan(t[drop_start], t[i-1], color='orange', alpha=0.3, zorder=0)
            in_drop = False
    if in_drop:
        ax.axvspan(t[drop_start], t[-1], color='orange', alpha=0.3, zorder=0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('North Position (m)')
    ax.set_title('(b) Position X with DVL Dropout Regions')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 速度对比 (vn) ──
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(t, gt_df['vn'].values, 'g-', linewidth=1.0, label='GT vn (NED)', zorder=3)
    ax.scatter(t[~mask_missing], cor_df['vn'].values[~mask_missing],
               c='steelblue', s=2, alpha=0.5, label='Corrupted vn', zorder=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('North Velocity (m/s)')
    ax.set_title('(c) Velocity Component vn (NED)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 加速度 + 脉冲噪声标注 ──
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(t, gt_df['ax'].values, 'g-', linewidth=1.0, label='GT ax', zorder=3, alpha=0.8)
    ax.plot(t, cor_df['ax'].values, 'b-', linewidth=0.5, alpha=0.4,
            label='Corrupted ax', zorder=2)
    ax.scatter(t[mask_impulse], cor_df['ax'].values[mask_impulse],
               c='red', s=15, marker='x', label='Impulse noise', zorder=5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Acceleration ax (m/s²)')
    ax.set_title('(d) IMU Acceleration with Impulse Noise')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] 数据预览图已保存: {output_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def main() -> None:
    """主流程: MSS → PINN 数据转换。"""
    print("=" * 70)
    print("MSS Simulation Data → PINN System Data Pipeline")
    print("=" * 70)

    # ─── 0. 检查输入 ───
    if not MSS_INPUT_PATH.exists():
        raise FileNotFoundError(f"MSS 输入文件不存在: {MSS_INPUT_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ─── 1. 读取 MSS 数据 ───
    print(f"\n[Step 1] 读取 MSS 仿真数据: {MSS_INPUT_PATH}")
    mss_df = pd.read_csv(MSS_INPUT_PATH)
    n = len(mss_df)
    dt = float(np.median(np.diff(mss_df['time'].values)))
    print(f"  样本数:     {n}")
    print(f"  采样间隔:   dt = {dt:.4f} s ({1/dt:.1f} Hz)")
    print(f"  时长:       {(mss_df['time'].iloc[-1] - mss_df['time'].iloc[0]):.1f} s")

    # ─── 2. 生成 Ground Truth ───
    print(f"\n[Step 2] 生成 Ground Truth (列变换 + 加速度微分)")
    gt_df = generate_ground_truth(mss_df)
    print(f"  GT 列数: {len(gt_df.columns)} → {list(gt_df.columns)}")
    print(f"  位置范围 (NED):")
    print(f"    North: [{gt_df['x'].min():.1f}, {gt_df['x'].max():.1f}] m")
    print(f"    East:  [{gt_df['y'].min():.1f}, {gt_df['y'].max():.1f}] m")
    print(f"    Down:  [{gt_df['z'].min():.1f}, {gt_df['z'].max():.1f}] m")
    print(f"  速度范围 (NED):")
    print(f"    vn: [{gt_df['vn'].min():.3f}, {gt_df['vn'].max():.3f}] m/s")
    print(f"    ve: [{gt_df['ve'].min():.3f}, {gt_df['ve'].max():.3f}] m/s")
    print(f"    vu: [{gt_df['vu'].min():.3f}, {gt_df['vu'].max():.3f}] m/s")
    print(f"  加速度范围 (Body):")
    print(f"    ax: [{gt_df['ax'].min():.3f}, {gt_df['ax'].max():.3f}] m/s^2")

    # ─── 3. 生成劣化数据 ───
    print(f"\n[Step 3] 注入传感器退化")
    cor_df, stats = generate_corrupted_data(gt_df, train_ratio=TRAIN_RATIO)
    print(f"  失锁段数: {stats['n_dropout_segments']}")
    print(f"  失锁点数: {stats['n_missing']:>6} ({100*stats['n_missing']/n:.1f}%)")
    print(f"  脉冲点数: {stats['n_impulse']:>6} ({100*stats['n_impulse']/n:.1f}%)")
    print(f"  正常点数: {stats['n_normal']:>6} ({100*stats['n_normal']/n:.1f}%)")

    # ─── 4. 保存 CSV ───
    print(f"\n[Step 4] 保存输出文件")
    gt_path = OUTPUT_DIR / "ground_truth.csv"
    cor_path = OUTPUT_DIR / "corrupted_data.csv"

    gt_df.to_csv(gt_path, index=False, float_format='%.6f')
    cor_df.to_csv(cor_path, index=False, float_format='%.6f')
    print(f"  [OK] {gt_path}")
    print(f"  [OK] {cor_path}")

    # ─── 5. 数据预览图 ───
    print(f"\n[Step 5] 生成数据预览图")
    fig_path = OUTPUT_DIR / "fig_mss_data_preview.png"
    plot_data_summary(gt_df, cor_df, fig_path)

    # ─── 完成 ───
    print(f"\n" + "=" * 70)
    print(f"[DONE] Data preprocessing complete!")
    print(f"=" * 70)
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  下一步: cd PINN && python train_baseline.py && python train.py")


if __name__ == '__main__':
    main()
