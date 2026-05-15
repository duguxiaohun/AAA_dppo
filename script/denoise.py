import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch

# ====== 1. 全局风格 ======
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'

# ====== 2. 配置 ======
h5_path = "/home/codon/github/AAA_dppo/script/offline_data/offline_dataset_20260107_161451.hdf5"
ep_id = 16
ep_name = f"{ep_id:05d}"

TRIM_START = 20
TARGET_STEP = 90
REAL_INDEX = TRIM_START + TARGET_STEP

# ====== 3. 读取数据 ======
chain_trajectory = None
print(f"Reading chains from Episode {ep_name}, Index {REAL_INDEX}...")

with h5py.File(h5_path, "r") as f:
    if ep_name in f["episodes"]:
        grp = f["episodes"][ep_name]
        if "chains" in grp:
            chains_ds = grp["chains"]
            if REAL_INDEX < len(chains_ds):
                raw_data = np.asarray(chains_ds[REAL_INDEX]).reshape(-1)
                if len(raw_data) == 21 * 2:
                    chain_trajectory = raw_data.reshape(21, 2)
                else:
                    print(f"Error: Data shape mismatch.")
            else:
                print(f"Error: Index out of bounds.")
        else:
            print("Error: 'chains' dataset not found.")

if chain_trajectory is None:
    exit()

# ====== 4. 绘图 ======
fig, ax = plt.subplots(figsize=(10, 8), dpi=120)

x = chain_trajectory[:, 0]
y = chain_trajectory[:, 1]

# --- 颜色映射 ---
# 紫色(开始/20) -> 黄色(结束/0)
cmap = plt.get_cmap('plasma')
norm = mcolors.Normalize(vmin=0, vmax=20)

# --- A. 绘制轨迹箭头 ---
for i in range(len(chain_trajectory) - 1):
    x_start, y_start = x[i], y[i]
    x_end, y_end = x[i+1], y[i+1]
    
    if np.hypot(x_end - x_start, y_end - y_start) > 1e-4:
        arrow = FancyArrowPatch(
            (x_start, y_start), (x_end, y_end),
            arrowstyle='-|>',
            mutation_scale=20,
            color=cmap(norm(i)), 
            linewidth=2.5,
            alpha=0.7, # 稍微透明，避免盖住圆点
            zorder=2
        )
        ax.add_patch(arrow)

# --- B. 绘制关键点 (核心修改：把点显示出来) ---

# 1. 起点 (Initial Noise, t=20)
ax.scatter(x[0], y[0], marker='x', s=150, color='gray', 
           linewidth=3.0, label='Initial Noise', zorder=3)

# 2. 中间点 (现在显示出来了！)
# s=60: 点的大小
# zorder=2: 和箭头同层级
sc = ax.scatter(x[1:-1], y[1:-1], c=np.arange(1, 20), cmap='plasma', norm=norm, 
                s=60, edgecolors='white', linewidth=0.5, zorder=2)

# 3. 终点 (Final Action, t=0)
ax.scatter(x[-1], y[-1], marker='*', s=450, color='#D62728', edgecolors='black', 
           linewidth=1.2, label='Final Action', zorder=4)

# --- C. 布局美化 ---

# 1. Colorbar (倒序)
cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Denoise Step', fontsize=20, rotation=90, labelpad=15)

# 刻度位置: 0, 5, 10, 15, 20
ticks_loc = np.linspace(0, 20, 5)
cbar.set_ticks(ticks_loc)

# 标签倒序: 20 -> 0
tick_labels = np.linspace(20, 0, 5).astype(int) 
cbar.set_ticklabels(tick_labels)
cbar.ax.tick_params(labelsize=16)

# 2. 坐标轴
ax.set_xlabel("Action Dim 1", fontsize=20, labelpad=10)
ax.set_ylabel("Action Dim 2", fontsize=20, labelpad=10)

# 3. Time Step (右上角)
ax.text(0.96, 0.96, f'Time Step: {TARGET_STEP}', transform=ax.transAxes,
        fontsize=20, fontweight='bold', color='#333333',
        ha='right', va='top',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'),
        zorder=10)

# 4. 图例 (左上角)
ax.legend(loc='upper left', fontsize=16, frameon=True, fancybox=False, framealpha=0.95)

# 5. 辅助线
ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax.axvline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
ax.axis('equal')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()

# 保存
save_name = f"Single_Chain_Ep{ep_name}_Step{TARGET_STEP}_Colored.png"
plt.savefig(save_name, dpi=300, bbox_inches='tight')
plt.savefig(save_name.replace(".png", ".pdf"), bbox_inches='tight')

print(f"Plot saved to {save_name}")
plt.show()