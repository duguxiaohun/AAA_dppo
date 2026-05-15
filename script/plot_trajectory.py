import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

# --- 配置参数 ---

# H5_PATH = "/home/codon/github/AAA_dppo/script/offline_data/offline_dataset_20260107_152917.hdf5"
H5_PATH = "/home/codon/github/AAA_dppo/script/offline_data/offline_dataset_20260107_161451.hdf5"
EP_ID = 19
EP_NAME = f"{EP_ID:05d}"

ORIGIN_X, ORIGIN_Y = -45, -45 
CLEAN_MODE = True 

# --- 【修改 1】参数更新 ---
STEP = 10                 # 步长改为 10
X_MIN, X_MAX = -60, 50    # X轴范围 -60 到 50
Y_MIN, Y_MAX = -29, 11    # Y轴范围 -30 到 10

def plot_gradient_traj(grp, name, ax, step=10, offset_x=0, offset_y=0):
    ds = grp[name]
    
    # 1. 读取数据
    if ds.dtype.kind == "O":
        mat = np.vstack([np.asarray(ds[i], dtype=np.float32).reshape(-1) for i in range(len(ds))])
    else:
        mat = np.asarray(ds[...], dtype=np.float32)
        if mat.ndim == 1:
            mat = mat[:, None]

    # 坐标平移
    x = mat[:, 0] - offset_x
    y = mat[:, 1] - offset_y
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    
    # --- 颜色定义 ---
#     styles = {
#     "ego": {"color": "#D62728", "marker": "o", "label": "Ego Speed", "lw": 2.5, "ms": 7, "zorder": 10},
#     "v_2": {"color": "#1F77B4", "marker": "s", "label": "Vehicle 2", "lw": 1.8, "ms": 6, "zorder": 5},
#     "v_3": {"color": "#9467BD", "marker": "^", "label": "Vehicle 3", "lw": 1.8, "ms": 6, "zorder": 5}
# }

    # if name == "ego":
    #     base_color_hex = "#D93025"  # 红
    #     glow_color_hex = "#FF4500"  # 橙红光晕
    #     core_color_hex = "#FFD700"  # 金色核心
    #     lw = 3.0
    #     z = 10
    # else:
    #     base_color_hex = "#1E88E5"  # 蓝
    #     glow_color_hex = "#81D4FA"  # 浅蓝光晕
    #     core_color_hex = "#1E88E5"  # 蓝色核心
    #     lw = 2.0
    #     z = 5
    if name == "ego":
        # === Ego: 红色主调 + 橙红光晕 + 金色核心 (王者感) ===
        base_color_hex = "#D62728"  # 鲜红 (原配置)
        glow_color_hex = "#FF6347"  # 番茄红 (光晕更亮)
        core_color_hex = "#FFD700"  # 金色 (核心高亮)
        marker = "o"
        lw = 3.0
        z = 10
        
    elif name == "v_2":
        # === v_2: 蓝色主调 + 青蓝光晕 + 白蓝核心 (科技感) ===
        base_color_hex = "#1F77B4"  # 经典蓝 (原配置)
        glow_color_hex = "#4FC3F7"  # 亮天蓝 (光晕)
        core_color_hex = "#E1F5FE"  # 近乎白的淡蓝 (核心)
        marker = "s"                # 方块
        lw = 2.0
        z = 5

    elif name == "v_3":
        # === v_3: 紫色主调 + 粉紫光晕 + 白紫核心 (神秘感) ===
        base_color_hex = "#9467BD"  # 优雅紫 (原配置)
        glow_color_hex = "#E0B0FF"  # 锦葵紫 (光晕)
        core_color_hex = "#F3E5F5"  # 极淡紫 (核心)
        marker = "^"                # 三角
        lw = 2.0
        z = 5

    elif name == "v_7":
        # === v_7: 绿色主调 + 嫩绿光晕 + 白绿核心 (清新感) ===
        base_color_hex = "#2CA02C"  # 墨绿 (原配置)
        glow_color_hex = "#90EE90"  # 淡绿 (光晕)
        core_color_hex = "#F1F8E9"  # 极淡绿 (核心)
        marker = "D"                # 菱形
        lw = 2.0
        z = 5
        
    else:
        # === 其他车辆: 灰色兜底 ===
        base_color_hex = "#7F7F7F"
        glow_color_hex = "#D3D3D3"
        core_color_hex = "#FFFFFF"
        marker = "x"
        lw = 1.5
        z = 3
    num_points = len(x)

    # ==========================================
    # Part A: 画渐变轨迹线 (LineCollection)
    # ==========================================
    # 将点两两连接成线段
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    num_segments = len(segments)

    # 生成透明度数组：从 0.1 (开始) -> 1.0 (结束)
    # 这样起点不会完全不可见，终点最深
    line_alphas = np.linspace(0.1, 1.0, num_segments)
    
    # 构造 RGBA 颜色数组
    base_rgb = to_rgba(base_color_hex)[:3] # 取 RGB
    colors = np.zeros((num_segments, 4))
    colors[:, :3] = base_rgb
    colors[:, 3] = line_alphas # 设置 Alpha

    # 创建线集合
    lc = LineCollection(segments, colors=colors, linewidth=lw, zorder=z)
    ax.add_collection(lc)

    # ==========================================
    # Part B: 画渐变轨迹点 (Scatter with Step)
    # ==========================================
    idx_points = np.arange(0, num_points, step)
    x_pts = x[idx_points]
    y_pts = y[idx_points]
    n_dots = len(idx_points)

    # 计算每个点在整个轨迹中的进度 (0.0 到 1.0)
    progress = idx_points / (num_points - 1)
    # 映射透明度：起点比较透(0.2)，终点不透(1.0)
    dot_alphas = 0.2 + 0.8 * progress 

    # 1. 核心点颜色 (Core)
    core_rgb = to_rgba(core_color_hex)[:3]
    core_rgba = np.zeros((n_dots, 4))
    core_rgba[:, :3] = core_rgb
    core_rgba[:, 3] = dot_alphas 

    # 2. 光晕颜色 (Glow)
    # 光晕本身就要半透明，所以透明度上限设低一点 (比如最大 0.5)
    glow_rgb = to_rgba(glow_color_hex)[:3]
    glow_rgba = np.zeros((n_dots, 4))
    glow_rgba[:, :3] = glow_rgb
    glow_rgba[:, 3] = dot_alphas * 0.5 # 让光晕比核心更透

    # 绘制散点
    ax.scatter(x_pts, y_pts, s=120, c=glow_rgba, edgecolors='none', zorder=z+1)
    ax.scatter(x_pts, y_pts, s=25, c=core_rgba, edgecolors='white', linewidth=0.5, zorder=z+2)

    # ==========================================
    # Part C: 起点和终点
    # ==========================================
    # 起点 (Start) - 设置为最淡的透明度
    start_alpha = 0.3
    ax.scatter([x[0]], [y[0]], marker='o', s=150, color=base_color_hex, alpha=start_alpha,
               edgecolors='white', linewidth=1.5, zorder=z+3)
    
    # 终点 (End) - 设置为完全不透明
    ax.scatter([x[-1]], [y[-1]], marker='x', s=150, color=base_color_hex, alpha=1.0,
               linewidth=3.0, zorder=z+3)

    return x, y

# --- 主程序 ---
with h5py.File(H5_PATH, "r") as f:
    if EP_NAME not in f["episodes"]:
        print(f"Episode {EP_NAME} not found.")
    else:
        grp = f["episodes"][EP_NAME]

        # 计算画布比例
        range_x = X_MAX - X_MIN
        range_y = Y_MAX - Y_MIN
        ratio = range_x / range_y
        base_h = 6
        
        fig = plt.figure(figsize=(base_h * ratio, base_h), dpi=150)
        ax = plt.gca()

        # 背景透明
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

        names = ["ego", "v_2", "v_3", "v_7"]  # 按需增减

        for n in names:
            if n in grp:
                plot_gradient_traj(grp, n, ax, step=STEP, offset_x=ORIGIN_X, offset_y=ORIGIN_Y)

        # 锁定范围
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)

        # 隐形锚点
        corners_x = [X_MIN, X_MAX, X_MIN, X_MAX]
        corners_y = [Y_MIN, Y_MIN, Y_MAX, Y_MAX]
        ax.scatter(corners_x, corners_y, alpha=0.0) 

        # 比例与反转
        ax.set_aspect('equal')
        ax.invert_yaxis()

        if CLEAN_MODE:
            ax.axis('off')
        else:
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_title(f"Range X[{X_MIN}, {X_MAX}], Y[{Y_MIN}, {Y_MAX}]")

        plt.tight_layout()
        save_name = f"traj_gradient_{EP_NAME}.png"
        plt.savefig(save_name, transparent=True, bbox_inches='tight', pad_inches=0)
        print(f"图片已保存: {save_name}")
        plt.show()