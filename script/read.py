import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import os

# ====== 1. 配置参数 ======
h5_path = "./diffusion_chains_log1.hdf5"
TARGET_POINT = np.array([0.831, 0.069])

# ====== 2. 绘图风格 ======
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'

# --- 标签定义 ---
GUIDANCE_COLOR = '#00BFFF'
LABEL_INIT = 'Initial Noise'
LABEL_GUIDE = 'Q-value Guidance'
LABEL_GT = 'Ground Truth'

# ====== 3. 读取函数 ======
def load_and_filter_chains(filepath, target_point, top_k=5):
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return None, None
    
    with h5py.File(filepath, "r") as f:
        all_chains = f["chains"][:]
    
    final_actions = all_chains[:, -1, :]
    dists = np.linalg.norm(final_actions - target_point, axis=1)
    st = 10
    top_indices = np.argsort(dists)[st:st+top_k]
    
    return all_chains[top_indices], dists[top_indices]

# ====== 4. 绘图函数 (最终定稿) ======
def plot_optimized_chains(chains, distances, target):
    if chains is None: return

    fig, ax = plt.subplots(figsize=(11, 9), dpi=120)
    
    # --- 颜色映射 ---
    cmap = plt.get_cmap('plasma')
    n_total_points = chains.shape[1]
    n_diff_steps = n_total_points - 2 
    norm = mcolors.Normalize(vmin=0, vmax=n_diff_steps)

    # --- 遍历轨迹 ---
    for idx, traj in enumerate(chains):
        x = traj[:, 0]
        y = traj[:, 1]
        
        # A. Q-value Guidance (Step 0->1)
        if np.hypot(x[1]-x[0], y[1]-y[0]) > 1e-4:
            arrow_guide = FancyArrowPatch(
                (x[0], y[0]), (x[1], y[1]),
                arrowstyle='-|>', mutation_scale=20, 
                color=GUIDANCE_COLOR, linewidth=2.8, alpha=1.0, zorder=5 
            )
            ax.add_patch(arrow_guide)

        # B. Diffusion Denoising (Step 1->End)
        for i in range(1, len(traj) - 1):
            if np.hypot(x[i+1]-x[i], y[i+1]-y[i]) > 1e-4:
                current_diff_step = i - 1 
                arrow_diff = FancyArrowPatch(
                    (x[i], y[i]), (x[i+1], y[i+1]),
                    arrowstyle='-|>', mutation_scale=16, 
                    color=cmap(norm(current_diff_step)), 
                    alpha=0.8, linewidth=1.8, zorder=3
                )
                ax.add_patch(arrow_diff)
        
        # C. 标记起点
        ax.scatter(x[0], y[0], marker='x', s=150, color='gray', linewidth=2.5, zorder=6)
        
        # D. 标记编号
        ax.text(x[0] + 0.08, y[0] - 0.08, 
                f"Traj {idx+1}", 
                fontsize=14, fontweight='bold', color='black', 
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.0),
                zorder=7)

        # E. 标记终点
        ax.scatter(x[-1], y[-1], s=60, color=cmap(0.99), edgecolor='white', zorder=4)

    # --- 绘制 Ground Truth ---
    ax.scatter(target[0], target[1], marker='*', s=550, 
               color='#2CA02C', edgecolors='black', linewidth=1.5, zorder=10)

    # --- 图例 (位置调整：往中间靠) ---
    legend_elements = [
        Line2D([0], [0], marker='x', color='w', markeredgecolor='gray', markersize=12, markeredgewidth=2.5, label=LABEL_INIT),
        Line2D([0], [0], color=GUIDANCE_COLOR, lw=3, label=LABEL_GUIDE),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#2CA02C', markeredgecolor='k', markersize=18, label=LABEL_GT)
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.32, 0.98), 
              fontsize=16, frameon=True, fancybox=False, framealpha=0.95)

    # --- L2 距离信息框 (位置调整：左侧中部偏上) ---
    info_text = "Final L2 Distances:\n"
    for i, d in enumerate(distances):
        info_text += f"Traj {i+1}: {d:.4f}\n"
    
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, edgecolor='gray')
    ax.text(0.06, 0.62, info_text, transform=ax.transAxes, fontsize=20,
            verticalalignment='center', horizontalalignment='left', bbox=props, zorder=12)

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Denoise Step', fontsize=22, rotation=90, labelpad=15)
    
    ticks_loc = np.linspace(0, n_diff_steps, 5)
    cbar.set_ticks(ticks_loc)
    tick_labels = np.linspace(9, 0, 5).astype(int) 
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=16)

    # --- 坐标轴 ---
    ax.set_xlabel("Action Dim 1", fontsize=20, labelpad=10)
    ax.set_ylabel("Action Dim 2", fontsize=20, labelpad=10)
    
    ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    
    # 保存图片
    png_name = "Final_Optimized_Chains_v3.png"
    pdf_name = "Final_Optimized_Chains_v3.pdf"
    
    plt.savefig(png_name, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_name, bbox_inches='tight')
    
    print(f"Plot saved to: {os.path.abspath(png_name)}")
    plt.show()

# ====== 5. 执行 ======
if __name__ == "__main__":
    chains, dists = load_and_filter_chains(h5_path, TARGET_POINT)
    if chains is not None:
        plot_optimized_chains(chains, dists, TARGET_POINT)