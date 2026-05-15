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

# --- 参数化设置 (修改此处) ---
# 输入你想要的排名（1代表距离最近的第1条，以此类推）
SELECTED_RANKS = [2, 4, 5, 9, 10]  # <---【修改这里】你想画哪几条？

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

# ====== 3. 读取函数 (支持特定排名提取) ======
def load_specific_ranks(filepath, target_point, selected_ranks):
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return None, None, None
    
    with h5py.File(filepath, "r") as f:
        all_chains = f["chains"][:]
    
    # 1. 计算所有轨迹终点到目标的距离
    final_actions = all_chains[:, -1, :]
    dists = np.linalg.norm(final_actions - target_point, axis=1)
    
    # 2. 对所有轨迹按距离进行排序 (从小到大)
    # argsort 返回的是排序后的索引
    sorted_indices = np.argsort(dists)
    
    # 3. 提取特定排名的轨迹
    valid_ranks = []
    target_indices = []
    
    total_data = len(all_chains)
    
    for r in selected_ranks:
        # 用户输入是 1-based (第1名)，Python索引是 0-based
        idx_in_sorted = r - 1
        
        if 0 <= idx_in_sorted < total_data:
            # 找到由于排序后位于第 r 位的原始数据索引
            original_idx = sorted_indices[idx_in_sorted]
            target_indices.append(original_idx)
            valid_ranks.append(r)
        else:
            print(f"[Warning] Rank {r} is out of bounds (Total data: {total_data}). Skipped.")
            
    if not target_indices:
        print("[Error] No valid ranks selected.")
        return None, None, None

    print(f"Selecting Ranks: {valid_ranks}")
    
    # 返回：筛选出的轨迹，对应的距离，以及对应的排名编号
    return all_chains[target_indices], dists[target_indices], valid_ranks

# ====== 4. 绘图函数 (适配特定排名列表) ======
def plot_selected_chains(chains, distances, ranks, target):
    if chains is None: return

    # 动态调整画布高度
    fig_height = 9 + (len(chains) * 0.2) 
    fig, ax = plt.subplots(figsize=(11, fig_height), dpi=120)
    
    # --- 颜色映射 ---
    cmap = plt.get_cmap('plasma')
    n_total_points = chains.shape[1]
    n_diff_steps = n_total_points - 2 
    norm = mcolors.Normalize(vmin=0, vmax=n_diff_steps)

    # --- 遍历选中的轨迹 ---
    # 使用 zip 同时遍历 轨迹数据(traj) 和 它的排名(rank)
    for i, (traj, rank) in enumerate(zip(chains, ranks)):
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
        for j in range(1, len(traj) - 1):
            if np.hypot(x[j+1]-x[j], y[j+1]-y[j]) > 1e-4:
                current_diff_step = j - 1 
                arrow_diff = FancyArrowPatch(
                    (x[j], y[j]), (x[j+1], y[j+1]),
                    arrowstyle='-|>', mutation_scale=16, 
                    color=cmap(norm(current_diff_step)), 
                    alpha=0.8, linewidth=1.8, zorder=3
                )
                ax.add_patch(arrow_diff)
        
        # C. 标记起点
        ax.scatter(x[0], y[0], marker='x', s=150, color='gray', linewidth=2.5, zorder=6)
        
        # D. 标记编号 (使用真实的 rank)
        # 简单的错位逻辑，防止文字重叠，也可以根据需要固定位置
        y_offset = -0.05 if i % 2 == 0 else 0.05
        
        ax.text(x[0] - 0.2, y[0] + y_offset, 
            f"Traj {rank}",  # 这里显示真实的排名
            fontsize=14, fontweight='bold', color='black', 
            ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5),
            zorder=7)

        # E. 标记终点
        ax.scatter(x[-1], y[-1], s=60, color=cmap(0.99), edgecolor='white', zorder=4)

    # --- 绘制 Ground Truth ---
    ax.scatter(target[0], target[1], marker='*', s=550, 
               color='#2CA02C', edgecolors='black', linewidth=1.5, zorder=10)

    # --- 图例 ---
    legend_elements = [
        Line2D([0], [0], marker='x', color='w', markeredgecolor='gray', markersize=12, markeredgewidth=2.5, label=LABEL_INIT),
        Line2D([0], [0], color=GUIDANCE_COLOR, lw=3, label=LABEL_GUIDE),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#2CA02C', markeredgecolor='k', markersize=18, label=LABEL_GT)
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98),
              fontsize=16, frameon=True, fancybox=False, framealpha=0.95)

    # --- L2 距离信息框 (显示真实排名) ---
    info_text = "Final L2 Distances:\n"
    for d, r in zip(distances, ranks):
        info_text += f"Rank {r}: {d:.4f}\n"
    
    props = dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.95, edgecolor='gray')
    ax.text(0.97, 0.1, info_text, transform=ax.transAxes, fontsize=18,
            verticalalignment='bottom', horizontalalignment='right', bbox=props, zorder=12)

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
    
    ax.margins(0.1) 
    ax.axis('equal')
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    
    # 文件名生成
    ranks_str = "_".join(map(str, ranks))
    if len(ranks_str) > 20: ranks_str = "Selected" # 防止文件名过长
    png_name = f"Final_Chains_Ranks_{ranks_str}.png"
    
    plt.savefig(png_name, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {os.path.abspath(png_name)}")
    plt.show()

# ====== 5. 执行 ======
if __name__ == "__main__":
    # 1. 提取数据
    chains, dists, ranks = load_specific_ranks(h5_path, TARGET_POINT, SELECTED_RANKS)
    
    # 2. 绘图
    if chains is not None:
        plot_selected_chains(chains, dists, ranks, TARGET_POINT)