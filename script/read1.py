import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import os

# ====== 1. 核心配置参数 (Parameterization) ======
h5_path = "./diffusion_chains_log2.hdf5"
TARGET_POINT = np.array([0.831, 0.069])

# --- 参数化设置 ---
TOP_K = 15         # <---【修改这里】你想画几条轨迹？
OFFSET = 0       # <---【修改这里】跳过前多少个？(0表示不跳过，10表示从第11个开始取)

# ====== 2. 绘图风格 ======
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'

# --- 标签与颜色定义 ---
GUIDANCE_COLOR = '#00BFFF'
LABEL_INIT = 'Initial Noise'
LABEL_GUIDE = 'Q-value Guidance'
LABEL_GT = 'Ground Truth'

# ====== 3. 读取函数 (支持参数化) ======
def load_and_filter_chains(filepath, target_point, top_k, offset=0):
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return None, None
    
    with h5py.File(filepath, "r") as f:
        all_chains = f["chains"][:]
    
    # 1. 计算所有轨迹终点到目标的距离
    final_actions = all_chains[:, -1, :]
    dists = np.linalg.norm(final_actions - target_point, axis=1)
    
    # 2. 排序并根据 offset 和 top_k 切片
    # argsort 返回的是从小到大的索引
    sorted_indices = np.argsort(dists)
    
    # 检查数据够不够
    start_idx = offset
    end_idx = offset + top_k
    
    if len(all_chains) < end_idx:
        print(f"[Warning] Not enough data. Requested index {end_idx}, but only have {len(all_chains)}.")
        end_idx = len(all_chains)
        if start_idx >= end_idx:
            print("[Error] Offset is larger than total data size.")
            return None, None

    target_indices = sorted_indices[start_idx:end_idx]
    
    print(f"Selecting Rank {start_idx+1} to {end_idx} (Top-{top_k} with Offset {offset})")
    
    return all_chains[target_indices], dists[target_indices]

# ====== 4. 绘图函数 (动态适应 K) ======
def plot_optimized_chains(chains, distances, target):
    if chains is None: return

    # 如果 K 很大，可能需要把画布拉高一点
    fig_height = 9 + (len(chains) * 0.2) 
    fig, ax = plt.subplots(figsize=(11, fig_height), dpi=120)
    
    # --- 颜色映射 ---
    cmap = plt.get_cmap('plasma')
    n_total_points = chains.shape[1]
    n_diff_steps = n_total_points - 2 
    norm = mcolors.Normalize(vmin=0, vmax=n_diff_steps)

    # --- 遍历 K 条轨迹 ---
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
        
        # D. 标记编号 (动态编号)
        # 这里的 idx+1 对应的是我们筛选出来的顺序 (1, 2, ... K)
        if idx == 0:
            ax.text(x[0] - 0.2, y[0] -0.05, 
                f"Traj {idx+1}", 
                fontsize=14, fontweight='bold', color='black', 
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5),
                zorder=7)
        elif idx == 4:
            ax.text(x[0] - 0.04, y[0] +0.3, 
                f"Traj {idx+1}", 
                fontsize=14, fontweight='bold', color='black', 
                ha='left', va='top',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5),
                zorder=7)
        else:
            ax.text(x[0] - 0.2, y[0] , 
                    f"Traj {idx+1}", 
                    fontsize=14, fontweight='bold', color='black', 
                    ha='left', va='top',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5),
                    zorder=7)

        # E. 标记终点
        ax.scatter(x[-1], y[-1], s=60, color=cmap(0.99), edgecolor='white', zorder=4)

    # --- 绘制 Ground Truth ---
    ax.scatter(target[0], target[1], marker='*', s=550, 
               color='#2CA02C', edgecolors='black', linewidth=1.5, zorder=10)

    # --- 图例 (位置：左上角) ---
    legend_elements = [
        Line2D([0], [0], marker='x', color='w', markeredgecolor='gray', markersize=12, markeredgewidth=2.5, label=LABEL_INIT),
        Line2D([0], [0], color=GUIDANCE_COLOR, lw=3, label=LABEL_GUIDE),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#2CA02C', markeredgecolor='k', markersize=18, label=LABEL_GT)
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98),
              fontsize=16, frameon=True, fancybox=False, framealpha=0.95)

    # --- L2 距离信息框 (动态生成内容) ---
    info_text = "Final L2 Distances:\n"
    # 动态循环生成 K 行文字
    for i, d in enumerate(distances):
        info_text += f"Traj {i+1}: {d:.4f}\n"
    
    # 放在右下角
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, edgecolor='gray')
    ax.text(0.97, 0.1, info_text, transform=ax.transAxes, fontsize=18,
            verticalalignment='bottom', horizontalalignment='right', bbox=props, zorder=12)

    # --- Colorbar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Denoise Step', fontsize=22, rotation=90, labelpad=15)
    
    # 刻度设置 (0, 9)
    ticks_loc = np.linspace(0, n_diff_steps, 5)
    cbar.set_ticks(ticks_loc)
    tick_labels = np.linspace(9, 0, 5).astype(int) 
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=16)

    # --- 坐标轴 ---
    ax.set_xlabel("Action Dim 1", fontsize=20, labelpad=10)
    ax.set_ylabel("Action Dim 2", fontsize=20, labelpad=10)
    
    ax.margins(0.1) 
    ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    
    # 文件名自动带上 K 值
    png_name = f"Final_Chains_Top{len(chains)}.png"
    pdf_name = f"Final_Chains_Top{len(chains)}.pdf"
    
    plt.savefig(png_name, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_name, bbox_inches='tight')
    
    print(f"Plot saved to: {os.path.abspath(png_name)}")
    plt.show()

# ====== 5. 执行 ======
if __name__ == "__main__":
    # 调用时传入配置的参数
    chains, dists = load_and_filter_chains(h5_path, TARGET_POINT, top_k=TOP_K, offset=OFFSET)
    
    if chains is not None:
        plot_optimized_chains(chains, dists, TARGET_POINT)