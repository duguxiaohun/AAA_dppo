import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

# --- 1. 字体与绘图风格设置 ---
config = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.figsize": (12, 7),
}
plt.rcParams.update(config)

# --- 配置参数 ---
# h5_path = "/home/codon/github/AAA_dppo/script/offline_data/offline_dataset_20260107_152917.hdf5"
h5_path = "/home/codon/github/AAA_dppo/script/offline_data/offline_dataset_20260107_161451.hdf5"
ep_id = 19
ep_name = f"{ep_id:05d}"

# --- 3. 定义你希望变成 (0,0) 的那个原始点 ---
ORIGIN_X, ORIGIN_Y = -45, -45 

def plot_traj(grp, name, ax, x_idx=0, y_idx=1, marker_every=0, offset_x=0, offset_y=0):
    ds = grp[name]

    # 读取数据
    if ds.dtype.kind == "O":
        mat = np.vstack([np.asarray(ds[i], dtype=np.float32).reshape(-1) for i in range(len(ds))])
    else:
        mat = np.asarray(ds[...], dtype=np.float32)
        if mat.ndim == 1:
            mat = mat[:, None]

    if mat.shape[1] <= max(x_idx, y_idx):
        raise ValueError(f"{name} 维度不足，mat.shape={mat.shape}")

    # 【关键修改】在这里进行坐标平移
    x = mat[:, x_idx] - offset_x
    y = mat[:, y_idx] - offset_y

    # --- 制作渐变轨迹 ---
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    base_color = next(ax._get_lines.prop_cycler)['color']

    n_seg = len(segments)
    alphas = np.linspace(0.1, 1.0, n_seg)
    
    rgba_colors = np.zeros((n_seg, 4))
    rgba_colors[:, :3] = to_rgba(base_color)[:3]
    rgba_colors[:, 3] = alphas

    lc = LineCollection(segments, colors=rgba_colors, linewidth=2.5)
    ax.add_collection(lc)

    # 图例占位
    ax.plot([], [], color=base_color, label=name, linewidth=2.5)

    # 起点终点
    ax.scatter([x[0]], [y[0]], marker="o", color=base_color, alpha=0.5, zorder=5, s=60)
    ax.scatter([x[-1]], [y[-1]], marker="x", color=base_color, alpha=1.0, zorder=5, s=100, linewidth=2.5)

    return mat

with h5py.File(h5_path, "r") as f:
    if ep_name not in f["episodes"]:
        print(f"Error: Episode {ep_name} not found.")
    else:
        grp = f["episodes"][ep_name]

        plt.figure(dpi=120)
        ax = plt.gca()

        names = ["ego", "v_1", "v_2", "v_3", "v_4", "v_5", "v_6", "v_7", "v_8", "v_9"]  # 按需增减
        names = ["ego", "v_2", "v_3", "v_7"]  # 按需增减

        for n in names:
            if n in grp:
                # 【关键修改】把偏移量传进去
                plot_traj(grp, n, ax, offset_x=ORIGIN_X, offset_y=ORIGIN_Y)
            else:
                print(f"[WARN] {n} not found")


        
        # 现在的坐标已经是相对坐标了，标签也可以改一下
        ax.set_xlabel("Relative Position X (m)")
        ax.set_ylabel("Relative Position Y (m)")
        ax.set_title(f"Trajectory (Centered at {ORIGIN_X}, {ORIGIN_Y})")

        # --- 设置视野 ---
        # 现在的中心点是 (0, 0)，不再是 (-40, -45)
        view_center_x, view_center_y = 0, 0
        width, height = 150, 60

        ax.set_xlim(view_center_x - width / 2, view_center_x + width / 2)
        ax.set_ylim(view_center_y - height / 2, view_center_y + height / 2)
        
        # 依然保留反转 Y 轴（如果原本数据是左手坐标系）
        ax.invert_yaxis()

        # 添加网格线，方便看 0,0 点
        ax.grid(True, linestyle='--', alpha=0.3)
        # 可以在 0,0 处画个红十字标记一下中心

        ax.legend(loc='upper right', frameon=True, framealpha=0.9)
        
        plt.tight_layout()
        plt.show()