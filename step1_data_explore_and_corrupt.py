"""
=============================================================================
AUV 轨迹预测 - 第一步：数据探索与"钻探干扰"注入
=============================================================================
背景：模拟水下石油钻探场景中，强声学噪声和信号遮挡导致的数据劣化
数据集：AUV_navigation_dataset (真实湖泊/海洋 AUV 航行数据)

数据列说明：
  lat/lon/alt  - 纬度/经度/高度 (GPS/GNSS 位置)
  roll/pitch/yaw - 横滚/俯仰/偏航角 (IMU 姿态)
  vn/ve        - 北向/东向速度 m/s (DVL)
  vf/vl/vu     - 前向/左向/上向速度 m/s
  ax/ay/az     - X/Y/Z 轴加速度 m/s² (IMU)
  wx/wy/wz     - X/Y/Z 轴角速度 rad/s (IMU)
  time         - 时间戳 (秒)
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # 非 GUI 环境必须，否则 plt.show() 会阻塞导致数据不保存
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import os

# ======================== 配置项 ========================
DATA_FILE = r"d:\数学建模\total_matlab\会议\AUV_dataset\20220712_0_1.csv"
OUTPUT_DIR = r"d:\数学建模\total_matlab\会议\AUV_dataset\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 钻探干扰参数
MISSING_GAP_RATIO  = 0.08   # 8% 的时间段数据完全丢失 (USBL 遮挡)
IMPULSE_NOISE_PROB = 0.05   # 5% 的数据点存在脉冲噪声 (钻机声学干扰)
IMPULSE_MAGNITUDE  = 15.0   # 脉冲噪声幅值 (正常值的倍数)
GAUSSIAN_STD       = 0.5    # 背景高斯白噪声标准差 (m)

# ======================== 1. 加载数据 ========================
print("=" * 60)
print("Step 1: 加载原始 AUV 导航数据")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
print(f"数据总行数: {len(df)} 行")
print(f"数据总列数: {len(df.columns)} 列")
print(f"时间跨度: {df['time'].min():.2f}s → {df['time'].max():.2f}s")
print(f"总时长: {df['time'].max() - df['time'].min():.2f} 秒")
print(f"\n列名: {list(df.columns)}")
print(f"\n数据统计:")
print(df[['lat', 'lon', 'alt', 'vn', 've', 'roll', 'pitch', 'yaw']].describe().round(4))

# ======================== 2. 坐标转换：经纬度 → 局部 XYZ (米) ========================
print("\nStep 2: 经纬度 → 局部 ENU 直角坐标系 (米)")

# 取第一个点作为原点
lat0 = df['lat'].iloc[0]
lon0 = df['lon'].iloc[0]
alt0 = df['alt'].iloc[0]

R_EARTH = 6371000.0  # 地球半径 (米)

# 简化的平面投影 (适合小范围场景 <10km)
df['x'] = (df['lon'] - lon0) * np.cos(np.radians(lat0)) * np.radians(1) * R_EARTH
df['y'] = (df['lat'] - lat0) * np.radians(1) * R_EARTH
df['z'] = df['alt'] - alt0
df['t'] = df['time'] - df['time'].min()  # 归一化时间

print(f"X 范围: [{df['x'].min():.2f}, {df['x'].max():.2f}] 米")
print(f"Y 范围: [{df['y'].min():.2f}, {df['y'].max():.2f}] 米")
print(f"Z 范围: [{df['z'].min():.2f}, {df['z'].max():.2f}] 米")
print(f"总轨迹长度: {df['x'].max() - df['x'].min():.1f} × {df['y'].max() - df['y'].min():.1f} 米")

# ======================== 3. 人工注入"钻探干扰" ========================
print("\nStep 3: 模拟水下钻探环境干扰 (数据劣化)")

x_clean = df[['x', 'y', 'z']].copy()
x_corrupted = df[['x', 'y', 'z']].copy()

n = len(df)
mask_missing  = np.zeros(n, dtype=bool)  # True = 数据缺失
mask_impulse  = np.zeros(n, dtype=bool)  # True = 脉冲噪声

# --- 干扰类型1：连续数据缺失段 (USBL 定位信号被遮挡) ---
num_gaps = max(1, int(n * MISSING_GAP_RATIO / 20))  # 缺失段数量
gap_size  = int(n * MISSING_GAP_RATIO / num_gaps)
np.random.seed(42)

# 确保训练区间和验证区间都有缺失段（防止验证集缺失样本数=0）
val_boundary = int(n * 0.8)
train_candidates = np.arange(50, val_boundary - gap_size)
val_candidates   = np.arange(val_boundary + 10, n - gap_size - 10)

# 前 80% 放 num_gaps-1 个，后 20% 至少放 1 个
n_train_gaps = max(1, num_gaps - 1)
n_val_gaps   = max(1, num_gaps - n_train_gaps)
gap_starts = list(np.random.choice(train_candidates, n_train_gaps, replace=False))
gap_starts += list(np.random.choice(val_candidates, n_val_gaps, replace=False))
gap_starts = np.array(sorted(gap_starts))

for gs in gap_starts:
    mask_missing[gs : gs + gap_size] = True

x_corrupted.loc[mask_missing, ['x', 'y', 'z']] = np.nan  # 缺失用 NaN 表示
print(f"  [OK] 数据缺失: {num_gaps} 个连续缺失段, 每段约 {gap_size} 个点 "
      f"({mask_missing.sum()} 点, 占 {100*mask_missing.mean():.1f}%)")

# --- 干扰类型2：随机脉冲噪声 (钻机破岩声学脉冲干扰) ---
impulse_idx = np.where(~mask_missing)[0]
impulse_sel = np.random.choice(impulse_idx,
                                size=int(len(impulse_idx) * IMPULSE_NOISE_PROB),
                                replace=False)
mask_impulse[impulse_sel] = True

std_x = df['x'].std()
std_y = df['y'].std()
x_corrupted.loc[mask_impulse, 'x'] += np.random.choice([-1, 1], size=mask_impulse.sum()) \
                                       * IMPULSE_MAGNITUDE * std_x
x_corrupted.loc[mask_impulse, 'y'] += np.random.choice([-1, 1], size=mask_impulse.sum()) \
                                       * IMPULSE_MAGNITUDE * std_y
print(f"  [OK] 脉冲噪声: {mask_impulse.sum()} 个点 "
      f"(幅值≈±{IMPULSE_MAGNITUDE*std_x:.1f} 米)")

# --- 干扰类型3：背景高斯白噪声 (仪器自身测量误差) ---
valid_idx = ~mask_missing & ~mask_impulse
x_corrupted.loc[valid_idx, 'x'] += np.random.normal(0, GAUSSIAN_STD, valid_idx.sum())
x_corrupted.loc[valid_idx, 'y'] += np.random.normal(0, GAUSSIAN_STD, valid_idx.sum())
x_corrupted.loc[valid_idx, 'z'] += np.random.normal(0, GAUSSIAN_STD * 0.3, valid_idx.sum())
print(f"  [OK] 高斯白噪声: σ = {GAUSSIAN_STD} 米 (施加于 {valid_idx.sum()} 个正常点)")

# ======================== 4. 可视化 ========================
print("\nStep 4: 生成可视化图表")

# ---- 图1: 原始 vs 劣化 3D 轨迹对比 ----
fig = plt.figure(figsize=(18, 7))
fig.suptitle('AUV Trajectory: Clean vs. Corrupted (Simulated Drilling Interference)',
             fontsize=14, fontweight='bold', y=0.98)

# 子图1: 原始轨迹
ax1 = fig.add_subplot(121, projection='3d')
sc1 = ax1.scatter(df['x'], df['y'], df['z'],
                  c=df['t'], cmap='plasma', s=2, alpha=0.8)
ax1.set_xlabel('X (East, m)', fontsize=9)
ax1.set_ylabel('Y (North, m)', fontsize=9)
ax1.set_zlabel('Z (Up, m)', fontsize=9)
ax1.set_title('(a) Ground Truth (Clean Data)', fontsize=11, fontweight='bold')
plt.colorbar(sc1, ax=ax1, label='Time (s)', shrink=0.5)

# 子图2: 劣化轨迹
ax2 = fig.add_subplot(122, projection='3d')
# 正常点（浅蓝色）
valid_mask = ~mask_missing & ~mask_impulse
ax2.scatter(x_corrupted.loc[valid_mask, 'x'],
            x_corrupted.loc[valid_mask, 'y'],
            x_corrupted.loc[valid_mask, 'z'],
            c='steelblue', s=2, alpha=0.6, label='Normal (w/ Gaussian noise)')
# 脉冲噪声点（红色）
ax2.scatter(x_corrupted.loc[mask_impulse, 'x'],
            x_corrupted.loc[mask_impulse, 'y'],
            x_corrupted.loc[mask_impulse, 'z'],
            c='red', s=20, alpha=1.0, marker='x', label='Impulse noise (drilling)')
# 缺失区域参考线（橙色）
for gs in gap_starts:
    ge = gs + gap_size
    ax2.plot([df['x'].iloc[gs], df['x'].iloc[min(ge, n-1)]],
             [df['y'].iloc[gs], df['y'].iloc[min(ge, n-1)]],
             [df['z'].iloc[gs], df['z'].iloc[min(ge, n-1)]],
             'orange', linewidth=3, alpha=0.7)

ax2.set_xlabel('X (East, m)', fontsize=9)
ax2.set_ylabel('Y (North, m)', fontsize=9)
ax2.set_zlabel('Z (Up, m)', fontsize=9)
ax2.set_title('(b) Corrupted Data (Drilling Environment)', fontsize=11, fontweight='bold')
ax2.legend(loc='upper left', fontsize=7)

plt.tight_layout()
save_path_1 = os.path.join(OUTPUT_DIR, 'fig1_trajectory_3d_comparison.png')
plt.savefig(save_path_1, dpi=150, bbox_inches='tight')
print(f"  [OK] 保存: {save_path_1}")


# ---- 图2: 时间序列对比（X轴，含缺失段标注）----
fig2, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig2.suptitle('AUV Position Time Series: Clean vs. Corrupted\n'
              '(Orange shading = USBL signal blocked; Red dots = drilling impulse noise)',
              fontsize=12, fontweight='bold')

labels = ['X (East, m)', 'Y (North, m)', 'Z (Depth, m)']
coords  = ['x', 'y', 'z']
colors_clean = ['royalblue', 'seagreen', 'darkorange']

for i, (ax, coord, label, color) in enumerate(zip(axes, coords, labels, colors_clean)):
    t = df['t'].values
    clean_vals = x_clean[coord].values
    corr_vals  = x_corrupted[coord].values

    # 绘制真实轨迹
    ax.plot(t, clean_vals, color=color, linewidth=1.5, label='Ground Truth', zorder=3)

    # 绘制劣化后的点
    plot_mask = ~mask_missing
    ax.scatter(t[plot_mask], corr_vals[plot_mask],
               c='lightgray', s=3, alpha=0.7, label='Corrupted (observed)', zorder=2)

    # 标注脉冲噪声点
    ax.scatter(t[mask_impulse], corr_vals[mask_impulse],
               c='red', s=25, marker='x', zorder=5, label='Impulse noise')

    # 标注缺失段 (橙色阴影)
    for j, gs in enumerate(gap_starts):
        ge = min(gs + gap_size, n - 1)
        ax.axvspan(t[gs], t[ge], alpha=0.25, color='orange',
                   label='Missing segment' if j == 0 else '')

    ax.set_ylabel(label, fontsize=10)
    ax.legend(loc='upper right', fontsize=7, ncol=4)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (s)', fontsize=10)
plt.tight_layout()
save_path_2 = os.path.join(OUTPUT_DIR, 'fig2_timeseries_comparison.png')
plt.savefig(save_path_2, dpi=150, bbox_inches='tight')
print(f"  [OK] 保存: {save_path_2}")
plt.show()

# ======================== 5. 保存处理后的数据 ========================
print("\nStep 5: 保存处理后的数据集")

# 保存 Ground Truth
gt_save = pd.DataFrame({
    't': df['t'], 'x': df['x'], 'y': df['y'], 'z': df['z'],
    'vn': df['vn'], 've': df['ve'], 'vu': df['vu'],
    'roll': df['roll'], 'pitch': df['pitch'], 'yaw': df['yaw'],
    'ax': df['ax'], 'ay': df['ay'], 'az': df['az'],
    'wx': df['wx'], 'wy': df['wy'], 'wz': df['wz'],
})
gt_path = os.path.join(OUTPUT_DIR, 'ground_truth.csv')
gt_save.to_csv(gt_path, index=False)
print(f"  [OK] Ground Truth 已保存: {gt_path}")

# 保存劣化数据（含缺失段 NaN 和噪声）
corrupted_save = gt_save.copy()
corrupted_save['x'] = x_corrupted['x'].values
corrupted_save['y'] = x_corrupted['y'].values
corrupted_save['z'] = x_corrupted['z'].values
corrupted_save['is_missing'] = mask_missing.astype(int)
corrupted_save['is_impulse'] = mask_impulse.astype(int)
corrupted_path = os.path.join(OUTPUT_DIR, 'corrupted_data.csv')
corrupted_save.to_csv(corrupted_path, index=False)
print(f"  [OK] 劣化数据 已保存: {corrupted_path}")

# ======================== 总结 ========================
print("\n" + "=" * 60)
print("[DONE] 数据探索与劣化注入完成！摘要：")
print("=" * 60)
print(f"  原始数据点数:   {n}")
print(f"  数据缺失点数:   {mask_missing.sum()} ({100*mask_missing.mean():.1f}%)")
print(f"  脉冲噪声点数:   {mask_impulse.sum()} ({100*mask_impulse.mean():.1f}%)")
print(f"  正常(含高斯噪声): {valid_idx.sum()} ({100*valid_idx.mean():.1f}%)")
print(f"\n  输出文件目录: {OUTPUT_DIR}")
print(f"  → ground_truth.csv   (用于模型训练的参考真实值)")
print(f"  → corrupted_data.csv (用于模型输入的劣化观测数据)")
print(f"\n⏭️  下一步：使用这两个文件训练 Robust PINN 模型！")
