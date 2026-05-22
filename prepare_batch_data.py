"""
=============================================================================
prepare_batch_data.py  —  批量 MSS 轨迹 → PINN 训练数据管线
=============================================================================
将 generate_100_trajectories.m 生成的 100 条 AUV 仿真轨迹批量转换为
Robust PINN v2.0 系统所需的训练数据格式。

与单轨迹版 prepare_data.py 的核心区别:
    1. 读取 trajectory_dataset/ 目录下所有 traj_*.csv
    2. 为每条轨迹独立注入不同随机种子的传感器退化
    3. 在 CSV 中追加 trajectory_id 列，用于跨轨迹分割
    4. 按轨迹级别 (而非时间步级别) 进行 train/val/test 分割
       避免同一轨迹的窗口同时出现在训练集和验证集

数据流:
    trajectory_dataset/traj_001.csv ... traj_100.csv
         │
         ├─ 1. 逐轨迹: 列重命名 + Body→NED 旋转 + SavGol 加速度
         ├─ 2. 逐轨迹: 独立随机种子注入传感器退化
         ├─ 3. 按轨迹 ID 划分: 70条训练 / 15条验证 / 15条测试
         ├─ 4. 拼接并导出
         │
         └─→ output/ground_truth.csv + corrupted_data.csv
              (含 trajectory_id 列，用于 dataset.py 中的轨迹感知分割)

使用方法:
    python prepare_batch_data.py
"""

from __future__ import annotations

import os
import warnings
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

TRAJ_INPUT_DIR = Path(__file__).resolve().parent / "trajectory_dataset"
OUTPUT_DIR     = Path(__file__).resolve().parent / "output"

DEGRADATION_LEVEL = os.getenv("AUV_DEGRADATION_LEVEL", "medium")
DEGRADATION_PRESETS = {
    "light": {
        "dvl_dropout_ratio": 0.06,
        "imu_impulse_prob": 0.03,
        "outage_min_len": 30,
        "outage_max_len": 80,
    },
    "medium": {
        "dvl_dropout_ratio": 0.15,
        "imu_impulse_prob": 0.06,
        "outage_min_len": 10,
        "outage_max_len": 30,
    },
    "heavy": {
        "dvl_dropout_ratio": 0.25,
        "imu_impulse_prob": 0.10,
        "outage_min_len": 20,
        "outage_max_len": 60,
    },
}
if DEGRADATION_LEVEL not in DEGRADATION_PRESETS:
    raise ValueError(
        f"Unsupported AUV_DEGRADATION_LEVEL={DEGRADATION_LEVEL!r}; "
        f"choose one of {sorted(DEGRADATION_PRESETS)}"
    )
DEG = DEGRADATION_PRESETS[DEGRADATION_LEVEL]

# 传感器退化参数
DVL_DROPOUT_RATIO     = DEG["dvl_dropout_ratio"]
IMU_IMPULSE_PROB      = DEG["imu_impulse_prob"]
DVL_OUTAGE_MIN_LEN    = DEG["outage_min_len"]
DVL_OUTAGE_MAX_LEN    = DEG["outage_max_len"]
IMU_IMPULSE_MAGNITUDE = 8.0
GAUSSIAN_NOISE_STD_POS = 0.3
GAUSSIAN_NOISE_STD_VEL = 0.05
GAUSSIAN_NOISE_STD_ATT = 0.005

# 轨迹级别分割比例
TRAIN_RATIO = 0.70   # 70 条训练
VAL_RATIO   = 0.15   # 15 条验证
# TEST = 1 - TRAIN - VAL = 0.15  → 15 条测试


# =============================================================================
# Coordinate Transform (与 prepare_data.py 完全一致)
# =============================================================================

def euler_to_rotation_matrix(
    roll: NDArray[np.float64],
    pitch: NDArray[np.float64],
    yaw: NDArray[np.float64],
) -> NDArray[np.float64]:
    """ZYX 欧拉角旋转矩阵 (Body → NED)。"""
    cphi, sphi = np.cos(roll), np.sin(roll)
    cth,  sth  = np.cos(pitch), np.sin(pitch)
    cpsi, spsi = np.cos(yaw), np.sin(yaw)

    n = len(roll)
    R = np.zeros((n, 3, 3), dtype=np.float64)

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
    u: NDArray, v: NDArray, w: NDArray,
    roll: NDArray, pitch: NDArray, yaw: NDArray,
) -> Tuple[NDArray, NDArray, NDArray]:
    """将体坐标系速度旋转至 NED 导航坐标系。"""
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    vel_body = np.stack([u, v, w], axis=-1)[..., np.newaxis]
    vel_ned = (R @ vel_body).squeeze(-1)
    return vel_ned[:, 0], vel_ned[:, 1], vel_ned[:, 2]


def compute_body_accelerations(
    u: NDArray, v: NDArray, w: NDArray,
    dt: float, window_length: int = 11, polyorder: int = 3,
) -> Tuple[NDArray, NDArray, NDArray]:
    """Savitzky-Golay 数值微分计算体坐标系加速度。"""
    wl = min(window_length, len(u))
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, polyorder + 2)
    if wl % 2 == 0:
        wl += 1

    ax = savgol_filter(u, wl, polyorder, deriv=1, delta=dt)
    ay = savgol_filter(v, wl, polyorder, deriv=1, delta=dt)
    az = savgol_filter(w, wl, polyorder, deriv=1, delta=dt)
    return ax, ay, az


# =============================================================================
# Per-trajectory Processing
# =============================================================================

def process_single_trajectory(
    mss_df: pd.DataFrame,
    traj_id: int,
) -> pd.DataFrame:
    """将单条 MSS 仿真 CSV 转换为 PINN ground truth 格式。

    Args:
        mss_df: MSS 仿真数据 (23 列).
        traj_id: 轨迹编号.

    Returns:
        gt_df: 17 列 DataFrame (t,x,y,z,vn,ve,vu,roll,pitch,yaw,
               ax,ay,az,wx,wy,wz,trajectory_id).
    """
    t = mss_df['time'].values - mss_df['time'].iloc[0]
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.05

    x = mss_df['North_m'].values.astype(np.float64)
    y = mss_df['East_m'].values.astype(np.float64)
    z = mss_df['Down_m'].values.astype(np.float64)

    u = mss_df['u_ms'].values.astype(np.float64)
    v = mss_df['v_ms'].values.astype(np.float64)
    w = mss_df['w_ms'].values.astype(np.float64)

    roll  = mss_df['Roll_rad'].values.astype(np.float64)
    pitch = mss_df['Pitch_rad'].values.astype(np.float64)
    yaw   = mss_df['Yaw_rad'].values.astype(np.float64)

    wx = mss_df['p_rads'].values.astype(np.float64)
    wy = mss_df['q_rads'].values.astype(np.float64)
    wz = mss_df['r_rads'].values.astype(np.float64)

    vn, ve, vd = transform_body_to_ned(u, v, w, roll, pitch, yaw)

    ax, ay, az = compute_body_accelerations(u, v, w, dt)

    # --- Thrust / control inputs (Fossen dynamics τ) ---
    thrust_net = mss_df['Thrust_net_N'].values.astype(np.float64)
    rudder     = mss_df['Rudder_rad'].values.astype(np.float64)
    stern      = mss_df['SternPlane_rad'].values.astype(np.float64)
    prop_rpm   = mss_df['Propeller_RPM'].values.astype(np.float64)

    gt_df = pd.DataFrame({
        't': t, 'x': x, 'y': y, 'z': z,
        # "vu" is kept for backward compatibility; it is NED Down velocity.
        'vn': vn, 've': ve, 'vu': vd,
        'u_body': u, 'v_body': v, 'w_body': w,
        'roll': roll, 'pitch': pitch, 'yaw': yaw,
        'ax': ax, 'ay': ay, 'az': az,
        'wx': wx, 'wy': wy, 'wz': wz,
        'thrust_net_N': thrust_net,
        'rudder_rad': rudder,
        'stern_rad': stern,
        'prop_rpm': prop_rpm,
        'trajectory_id': traj_id,
    })

    return gt_df


# =============================================================================
# Sensor Degradation Injection
# =============================================================================

def inject_degradation_single_trajectory(
    gt_df: pd.DataFrame,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """为单条轨迹注入传感器退化。

    每条轨迹使用独立的随机种子，确保退化模式多样化。

    Args:
        gt_df: 单条轨迹的 ground truth.
        rng: 该轨迹专属的随机数生成器.

    Returns:
        cor_df: 含 is_missing/is_impulse 标志列的劣化 DataFrame.
        stats: 统计信息.
    """
    cor_df = gt_df.copy()
    n = len(cor_df)

    # --- DVL 失锁 (连续 NaN 段) ---
    total_missing = int(n * DVL_DROPOUT_RATIO)
    avg_gap = max(1, (DVL_OUTAGE_MIN_LEN + DVL_OUTAGE_MAX_LEN) // 2)
    num_gaps = max(2, total_missing // avg_gap)

    mask_missing = np.zeros(n, dtype=bool)
    safe_margin = 100

    if n > 2 * safe_margin + avg_gap:
        starts = rng.choice(
            np.arange(safe_margin, n - avg_gap - safe_margin),
            size=min(num_gaps, n // avg_gap),
            replace=False,
        )
        for gs in starts:
            gap_len = rng.integers(DVL_OUTAGE_MIN_LEN, DVL_OUTAGE_MAX_LEN + 1)
            ge = min(gs + gap_len, n)
            mask_missing[gs:ge] = True

    dropout_cols = ['x', 'y', 'z', 'vn', 've', 'vu']
    cor_df.loc[mask_missing, dropout_cols] = np.nan

    # --- IMU 脉冲噪声 ---
    valid_for_impulse = ~mask_missing
    candidates = np.where(valid_for_impulse)[0]
    n_impulse = int(len(candidates) * IMU_IMPULSE_PROB)
    mask_impulse = np.zeros(n, dtype=bool)
    if n_impulse > 0 and len(candidates) > n_impulse:
        impulse_idx = rng.choice(candidates, size=n_impulse, replace=False)
        mask_impulse[impulse_idx] = True

    for col in ['ax', 'ay', 'az']:
        sigma = float(np.nanstd(gt_df[col].values))
        if sigma < 1e-10:
            sigma = 0.1
        n_imp = int(mask_impulse.sum())
        if n_imp > 0:
            signs = rng.choice([-1.0, 1.0], size=n_imp)
            cor_df.loc[mask_impulse, col] += signs * IMU_IMPULSE_MAGNITUDE * sigma

    # --- 背景高斯噪声 ---
    normal_mask = ~mask_missing
    n_normal = int(normal_mask.sum())

    for col in ['x', 'y', 'z']:
        cor_df.loc[normal_mask, col] += rng.normal(0, GAUSSIAN_NOISE_STD_POS, n_normal)
    for col in ['vn', 've', 'vu']:
        cor_df.loc[normal_mask, col] += rng.normal(0, GAUSSIAN_NOISE_STD_VEL, n_normal)
    for col in ['roll', 'pitch', 'yaw']:
        cor_df.loc[normal_mask, col] += rng.normal(0, GAUSSIAN_NOISE_STD_ATT, n_normal)

    # --- 标志列 ---
    cor_df['is_missing'] = mask_missing.astype(np.int32)
    cor_df['is_impulse'] = mask_impulse.astype(np.int32)

    stats = {
        'total': n,
        'n_missing': int(mask_missing.sum()),
        'n_impulse': int(mask_impulse.sum()),
    }

    return cor_df, stats


# =============================================================================
# Visualization
# =============================================================================

def plot_multi_trajectory_summary(
    gt_df: pd.DataFrame,
    cor_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """绘制多轨迹数据集预览图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    traj_ids = gt_df['trajectory_id'].unique()
    n_traj = len(traj_ids)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f'Multi-Trajectory AUV Dataset ({n_traj} trajectories)',
        fontsize=14, fontweight='bold',
    )

    # (a) 3D 轨迹全局视图
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    cmap = plt.cm.tab20(np.linspace(0, 1, min(n_traj, 20)))
    for i, tid in enumerate(traj_ids[:20]):
        mask = gt_df['trajectory_id'] == tid
        ax.plot(
            gt_df.loc[mask, 'x'].values,
            gt_df.loc[mask, 'y'].values,
            gt_df.loc[mask, 'z'].values,
            color=cmap[i % 20], linewidth=0.8, alpha=0.7,
        )
    ax.set_xlabel('North (m)')
    ax.set_ylabel('East (m)')
    ax.set_zlabel('Down (m)')
    ax.set_title(f'(a) 3D Trajectories (showing {min(n_traj, 20)}/{n_traj})')
    ax.invert_zaxis()

    # (b) 速度分布直方图
    ax = fig.add_subplot(2, 2, 2)
    speed = np.sqrt(
        gt_df['vn'].values**2 + gt_df['ve'].values**2 + gt_df['vu'].values**2
    )
    ax.hist(speed, bins=100, color='steelblue', alpha=0.7, edgecolor='none')
    ax.set_xlabel('Speed (m/s)')
    ax.set_ylabel('Count')
    ax.set_title('(b) Speed Distribution (all trajectories)')
    ax.grid(True, alpha=0.3)

    # (c) 深度分布
    ax = fig.add_subplot(2, 2, 3)
    ax.hist(gt_df['z'].values, bins=100, color='coral', alpha=0.7, edgecolor='none')
    ax.set_xlabel('Depth / Down (m)')
    ax.set_ylabel('Count')
    ax.set_title('(c) Depth Distribution')
    ax.grid(True, alpha=0.3)

    # (d) 每条轨迹的退化统计
    ax = fig.add_subplot(2, 2, 4)
    missing_ratios = []
    impulse_ratios = []
    for tid in traj_ids:
        mask = cor_df['trajectory_id'] == tid
        sub = cor_df.loc[mask]
        n_sub = len(sub)
        missing_ratios.append(sub['is_missing'].sum() / n_sub * 100)
        impulse_ratios.append(sub['is_impulse'].sum() / n_sub * 100)
    ax.bar(range(len(missing_ratios)), missing_ratios, alpha=0.7,
           color='orange', label='DVL Dropout %')
    ax.bar(range(len(impulse_ratios)), impulse_ratios, alpha=0.7,
           color='red', label='IMU Impulse %', bottom=missing_ratios)
    ax.set_xlabel('Trajectory Index')
    ax.set_ylabel('Degradation Ratio (%)')
    ax.set_title('(d) Per-Trajectory Sensor Degradation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] 数据集预览图: {output_path}")


# =============================================================================
# Main Pipeline
# =============================================================================

def main() -> None:
    """主流程: 批量 MSS → PINN 数据转换。"""
    print("=" * 70)
    print("批量 MSS 轨迹 → PINN 训练数据管线")
    print("=" * 70)

    # ─── 0. 发现轨迹文件 ───
    if not TRAJ_INPUT_DIR.exists():
        raise FileNotFoundError(
            f"轨迹目录不存在: {TRAJ_INPUT_DIR}\n"
            f"请先在 MATLAB 中运行 generate_100_trajectories.m"
        )

    traj_files = sorted(TRAJ_INPUT_DIR.glob("traj_*.csv"))
    n_traj = len(traj_files)
    if n_traj == 0:
        raise FileNotFoundError(
            f"未找到轨迹文件 (traj_*.csv) 在 {TRAJ_INPUT_DIR}"
        )

    print(f"\n[Step 0] 发现 {n_traj} 条轨迹文件")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta_path = TRAJ_INPUT_DIR / "trajectory_metadata.csv"
    traj_meta = None
    if meta_path.exists():
        traj_meta = pd.read_csv(meta_path)
        print(f"  [OK] 读取激励元数据: {meta_path.name}")
    else:
        print("  [WARN] 未找到 trajectory_metadata.csv，将跳过持续激励评分检查")

    # ─── 1. 逐轨迹处理 ───
    print(f"\n[Step 1] 逐轨迹预处理 (坐标变换 + 加速度微分)")
    all_gt: List[pd.DataFrame] = []
    all_cor: List[pd.DataFrame] = []
    total_stats = {'total': 0, 'n_missing': 0, 'n_impulse': 0}

    for i, fpath in enumerate(traj_files):
        traj_id = i + 1

        mss_df = pd.read_csv(fpath)
        n_rows = len(mss_df)

        # Ground truth
        gt_df = process_single_trajectory(mss_df, traj_id)

        # 传感器退化 (每条轨迹独立种子)
        rng = np.random.default_rng(seed=42 + traj_id * 137)
        cor_df, stats = inject_degradation_single_trajectory(gt_df, rng)

        all_gt.append(gt_df)
        all_cor.append(cor_df)

        total_stats['total'] += stats['total']
        total_stats['n_missing'] += stats['n_missing']
        total_stats['n_impulse'] += stats['n_impulse']

        if (i + 1) % 10 == 0 or i == 0 or i == n_traj - 1:
            print(f"  [{i+1:3d}/{n_traj}] {fpath.name}: "
                  f"{n_rows} steps, "
                  f"missing={stats['n_missing']}, "
                  f"impulse={stats['n_impulse']}")

    # ─── 2. 拼接 ───
    print(f"\n[Step 2] 拼接全部轨迹")
    gt_all = pd.concat(all_gt, ignore_index=True)
    cor_all = pd.concat(all_cor, ignore_index=True)

    n_total = len(gt_all)
    print(f"  总样本数: {n_total:,}")
    print(f"  总轨迹数: {n_traj}")
    print(
        f"  Degradation level: {DEGRADATION_LEVEL} "
        f"(DVL={DVL_DROPOUT_RATIO:.0%}, IMU={IMU_IMPULSE_PROB:.0%}, "
        f"outage={DVL_OUTAGE_MIN_LEN}-{DVL_OUTAGE_MAX_LEN} frames)"
    )
    print(f"  DVL 失锁: {total_stats['n_missing']:,} "
          f"({100*total_stats['n_missing']/n_total:.1f}%)")
    print(f"  IMU 脉冲: {total_stats['n_impulse']:,} "
          f"({100*total_stats['n_impulse']/n_total:.1f}%)")

    if traj_meta is not None and not traj_meta.empty:
        print(f"\n[Step 2b] 持续激励检查")
        meta_sorted = traj_meta.sort_values('pe_score', ascending=False)
        top_rows = meta_sorted.head(3)[['trajectory_id', 'type', 'pe_score']]
        low_rows = meta_sorted.tail(3)[['trajectory_id', 'type', 'pe_score']]
        print("  PE 评分最高的 3 条轨迹:")
        for _, row in top_rows.iterrows():
            print(f"    traj_{int(row['trajectory_id']):03d} | {row['type']} | score={row['pe_score']:.3f}")
        print("  PE 评分最低的 3 条轨迹:")
        for _, row in low_rows.iterrows():
            print(f"    traj_{int(row['trajectory_id']):03d} | {row['type']} | score={row['pe_score']:.3f}")

        pe_median = float(traj_meta['pe_score'].median())
        pe_p10 = float(traj_meta['pe_score'].quantile(0.10))
        print(f"  PE 评分中位数: {pe_median:.3f} | 10%分位: {pe_p10:.3f}")

        weak_mask = traj_meta['pe_score'] < pe_p10
        if weak_mask.any():
            weak_ids = traj_meta.loc[weak_mask, 'trajectory_id'].astype(int).tolist()
            print(f"  [WARN] 低激励轨迹（低于10%分位）: {weak_ids}")

    # ─── 3. 轨迹级别分割标注 ───
    print(f"\n[Step 3] 轨迹级别 train/val/test 划分")
    traj_ids = np.arange(1, n_traj + 1)
    rng_split = np.random.default_rng(seed=2024)
    rng_split.shuffle(traj_ids)

    n_train = int(n_traj * TRAIN_RATIO)
    n_val   = int(n_traj * VAL_RATIO)

    train_ids = set(traj_ids[:n_train].tolist())
    val_ids   = set(traj_ids[n_train:n_train + n_val].tolist())
    test_ids  = set(traj_ids[n_train + n_val:].tolist())

    # 添加 split 列
    def assign_split(tid: int) -> str:
        if tid in train_ids:
            return 'train'
        elif tid in val_ids:
            return 'val'
        else:
            return 'test'

    gt_all['split'] = gt_all['trajectory_id'].apply(assign_split)
    cor_all['split'] = cor_all['trajectory_id'].apply(assign_split)

    n_train_samples = (gt_all['split'] == 'train').sum()
    n_val_samples   = (gt_all['split'] == 'val').sum()
    n_test_samples  = (gt_all['split'] == 'test').sum()

    print(f"  训练集: {n_train} 轨迹, {n_train_samples:,} 样本")
    print(f"  验证集: {n_val} 轨迹, {n_val_samples:,} 样本")
    print(f"  测试集: {n_traj - n_train - n_val} 轨迹, {n_test_samples:,} 样本")
    print(f"  训练轨迹 IDs: {sorted(train_ids)[:10]}... ")
    print(f"  验证轨迹 IDs: {sorted(val_ids)}")
    print(f"  测试轨迹 IDs: {sorted(test_ids)}")

    # ─── 4. 导出 CSV ───
    print(f"\n[Step 4] 导出 CSV")

    gt_path = OUTPUT_DIR / "ground_truth.csv"
    cor_path = OUTPUT_DIR / "corrupted_data.csv"

    gt_all.to_csv(gt_path, index=False, float_format='%.6f')
    cor_all.to_csv(cor_path, index=False, float_format='%.6f')

    gt_size_mb = gt_path.stat().st_size / (1024 * 1024)
    cor_size_mb = cor_path.stat().st_size / (1024 * 1024)

    print(f"  [OK] {gt_path} ({gt_size_mb:.1f} MB)")
    print(f"  [OK] {cor_path} ({cor_size_mb:.1f} MB)")

    # 导出分割索引文件 (方便 dataset.py 使用)
    split_info = pd.DataFrame({
        'trajectory_id': list(range(1, n_traj + 1)),
        'split': [assign_split(tid) for tid in range(1, n_traj + 1)],
    })
    split_path = OUTPUT_DIR / "trajectory_splits.csv"
    split_info.to_csv(split_path, index=False)
    print(f"  [OK] {split_path}")

    # ─── 5. 数据统计摘要 ───
    print(f"\n[Step 5] 数据统计")
    print(f"  位置范围 (NED):")
    print(f"    North: [{gt_all['x'].min():.1f}, {gt_all['x'].max():.1f}] m")
    print(f"    East:  [{gt_all['y'].min():.1f}, {gt_all['y'].max():.1f}] m")
    print(f"    Down:  [{gt_all['z'].min():.1f}, {gt_all['z'].max():.1f}] m")
    print(f"  速度范围 (NED):")
    print(f"    vn: [{gt_all['vn'].min():.3f}, {gt_all['vn'].max():.3f}] m/s")
    print(f"    ve: [{gt_all['ve'].min():.3f}, {gt_all['ve'].max():.3f}] m/s")
    print(f"    vu/Down: [{gt_all['vu'].min():.3f}, {gt_all['vu'].max():.3f}] m/s")

    # ─── 6. 可视化 ───
    print(f"\n[Step 6] 生成可视化")
    fig_path = OUTPUT_DIR / "fig_multi_trajectory_dataset.png"
    plot_multi_trajectory_summary(gt_all, cor_all, fig_path)

    # ─── 7. PINN 训练数据量估算 ───
    print(f"\n[Step 7] PINN 训练数据量估算")
    seq_len = 20
    pred_len = 5
    window = seq_len + pred_len

    # 按轨迹计算可用窗口数
    train_windows = 0
    val_windows = 0
    test_windows = 0
    for tid in range(1, n_traj + 1):
        n_traj_samples = (gt_all['trajectory_id'] == tid).sum()
        n_windows = max(0, n_traj_samples - window + 1)
        split = assign_split(tid)
        if split == 'train':
            train_windows += n_windows
        elif split == 'val':
            val_windows += n_windows
        else:
            test_windows += n_windows

    print(f"  seq_len={seq_len}, pred_len={pred_len}, window={window}")
    print(f"  训练窗口: {train_windows:,}")
    print(f"  验证窗口: {val_windows:,}")
    print(f"  测试窗口: {test_windows:,}")
    print(f"  总窗口数: {train_windows + val_windows + test_windows:,}")

    n_model_params = 200_000  # 模型参数近似
    effective_independent = train_windows // window
    ratio = n_model_params / max(effective_independent, 1)
    print(f"  有效独立样本 (去重叠): ~{effective_independent:,}")
    print(f"  参数-样本比: {ratio:.1f}:1 "
          f"({'OK' if ratio < 10 else 'WARNING: too high'})")

    # ─── Done ───
    print(f"\n{'=' * 70}")
    print(f"[DONE] 批量数据预处理完成!")
    print(f"{'=' * 70}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  下一步:")
    print(f"    1. 检查 trajectory_splits.csv 中的分割是否合理")
    print(f"    2. cd PINN && python train.py")


if __name__ == '__main__':
    main()
