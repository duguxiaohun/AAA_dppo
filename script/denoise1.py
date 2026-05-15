import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch

# ====== 1. 全局绘图风格 ======
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'

# ====== 2. 数据录入 (直接使用你提供的 Tensor 数据) ======

# Tensor 1 数据
tensor_data_1 = np.array([
    [0.9803, 0.2675], [1.0000, 0.7675], [1.0000, 0.7675], [1.0000, 0.7675],
    [1.0000, 0.7675], [1.0000, 0.7675], [1.0000, 0.7675], [1.0000, 0.7675],
    [1.0000, 0.7675], [1.0000, 0.7675], [0.3850, 0.7022], [0.5888, 0.8280],
    [0.4947, 1.2445], [0.6163, 1.4545], [0.9925, 1.7313], [0.9839, 1.9298],
    [1.1985, 1.4844], [1.0893, 1.2242], [1.1515, 1.0677], [1.0882, 0.9736],
    [1.0000, 0.9671]
], dtype=np.float32)

# Tensor 2 数据
tensor_data_2 = np.array([
    [0.5082, 1.7541], [1.0000, 1.0000], [1.0000, 1.0000], [1.0000, 1.0000],
    [1.0000, 1.0000], [1.0000, 1.0000], [1.0000, 1.0000], [1.0000, 1.0000],
    [1.0000, 1.0000], [1.0000, 1.0000], [1.2980, 1.2949], [0.9702, 1.7321],
    [1.1672, 1.0615], [1.0499, 1.0452], [0.9287, 0.8726], [0.8202, 0.8153],
    [0.8698, 0.7635], [0.5282, 0.4839], [0.7729, 0.2850], [0.7727, 0.2805],
    [0.8137, 0.2712]
], dtype=np.float32)

# ====== 3. 绘图函数 ======
def plot_diffusion_trajectory(data, title, filename):
    """
    绘制单条去噪轨迹
    data: shape (21, 2)
    """
    # 确保数据形状正确
    data = data.reshape(21, 2)
    
    x = data[:, 0]
    y = data[:, 1]
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    
    # --- 颜色映射 ---
    # 使用 'viridis' 或 'plasma' 来表示时间进度
    cmap = plt.get_cmap('plasma')
    norm = mcolors.Normalize(vmin=0, vmax=20)
    
    # --- 绘制箭头轨迹 ---
    for i in range(len(data) - 1):
        x_start, y_start = x[i], y[i]
        x_end, y_end = x[i+1], y[i+1]
        
        # 如果起点和终点重合（比如模型在某些步数静止），则跳过箭头绘制
        if np.hypot(x_end - x_start, y_end - y_start) < 1e-4:
            continue
            
        color = cmap(norm(i))
        
        arrow = FancyArrowPatch((x_start, y_start), (x_end, y_end),
                                arrowstyle='-|>',
                                mutation_scale=20,  # 箭头大小
                                color=color,
                                linewidth=2.5,
                                alpha=0.9,
                                zorder=2)
        ax.add_patch(arrow)

    # --- 绘制关键点 ---
    # 1. 起点 (Initial Noise)
    ax.scatter(x[0], y[0], marker='x', s=150, color='gray', linewidth=3, 
               label='Initial Noise (t=0)', zorder=3)
    
    # 2. 中间点 (Denoising Steps)
    sc = ax.scatter(x[1:-1], y[1:-1], c=np.arange(1, 20), cmap='plasma', norm=norm, 
                    s=60, zorder=2, alpha=0.8)
    
    # 3. 终点 (Final Action)
    ax.scatter(x[-1], y[-1], marker='*', s=350, color='#D62728', edgecolors='k', 
               label='Final Action (t=20)', zorder=4)

    # --- 布局美化 ---
    # 添加颜色条
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Denoising Step', fontsize=18)
    cbar.set_ticks(np.arange(0, 21, 5))
    
    # 坐标轴
    ax.set_xlabel("Action Dim 1", fontsize=20, labelpad=10)
    ax.set_ylabel("Action Dim 2", fontsize=20, labelpad=10)
    ax.set_title(title, fontsize=20, pad=15)
    
    # 十字参考线 (0,0)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
    
    ax.legend(loc='best', fontsize=14, frameon=True, fancybox=False, framealpha=0.9)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.axis('equal') # 保持比例一致
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.show()

# ====== 4. 执行绘图 ======

# 画第一个 Tensor
plot_diffusion_trajectory(tensor_data_1, "Diffusion Trajectory (Sample 1)", "Diff_Traj_Sample1.png")

# 画第二个 Tensor
plot_diffusion_trajectory(tensor_data_2, "Diffusion Trajectory (Sample 2)", "Diff_Traj_Sample2.png")