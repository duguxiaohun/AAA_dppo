import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon
import os

# ====== 1. 全局配置 ======
# 使用你指定的 HDF5 文件
# h5_path = "/home/codon/github/AAA_dppo/script/offline_data/offline_dataset_20260107_152917.hdf5"
h5_path = "/home/codon/github/AAA_dppo/script/offline_data/offline_dataset_20260107_161451.hdf5"

ep_id = 19
TRIM_START = 20  # 前移截断参数
VLINE_X = 80     # 竖线位置

ep_name = f"{ep_id:05d}"

# 【修正】严格按照你的要求提取 v_2, v_3, v_7
names = ["ego", "v_2", "v_3", "v_7"]  


# ====== 2. 绘图风格 ======
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'

# ====== 3. 辅助函数：渐变填充 ======
def add_gradient_fill(ax, x, y, color, y_min=0):
    rgb = to_rgb(color)
    z = np.empty((100, 1, 4), dtype=float)
    z[:, :, 0] = rgb[0]
    z[:, :, 1] = rgb[1]
    z[:, :, 2] = rgb[2]
    z[:, :, 3] = np.linspace(0.35, 0.0, 100)[:, None]
    
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y_min, y.max() * 1.1
    
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='upper', zorder=2)
    
    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, y_min], xy, [xmax, y_min], [xmin, y_min]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', transform=ax.transData)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

# ====== 4. 样式定义 ======
styles = {
    # Ego: 红色, 圆点
    "ego": {"color": "#D62728", "marker": "o", "label": "Ego Speed", "lw": 2.5, "ms": 7, "zorder": 10},
    # V2: 蓝色, 方块
    "v_2": {"color": "#1F77B4", "marker": "s", "label": "Vehicle 2", "lw": 1.8, "ms": 6, "zorder": 5},
    # V3: 紫色, 三角
    "v_3": {"color": "#9467BD", "marker": "^", "label": "Vehicle 3", "lw": 1.8, "ms": 6, "zorder": 5},
    # 【新增配置】V7: 墨绿色, 菱形 (与前三个明显区分)
    "v_7": {"color": "#2CA02C", "marker": "D", "label": "Vehicle 5", "lw": 1.8, "ms": 6, "zorder": 5}
}

# ====== 5. 数据读取 ======
data_dict = {}
max_step = 0
print(f"Reading Episode {ep_name}, Trimming first {TRIM_START} steps...")

with h5py.File(h5_path, "r") as f:
    if ep_name in f["episodes"]:
        grp = f["episodes"][ep_name]
        for name in names:
            if name in grp:
                ds = grp[name]
                speeds = []
                for i in range(len(ds)):
                    try:
                        row = np.asarray(ds[i]).reshape(-1)
                        if len(row) >= 4:
                            v = np.sqrt(float(row[2])**2 + float(row[3])**2)
                            speeds.append(v)
                    except: pass
                
                speeds_arr = np.array(speeds)
                if len(speeds_arr) > TRIM_START:
                    trimmed_speeds = speeds_arr[TRIM_START:]
                    data_dict[name] = trimmed_speeds
                    max_step = max(max_step, len(trimmed_speeds))
            else:
                print(f"Warning: {name} not found in this episode.")

# ====== 6. 绘图 ======
fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

for name in names:
    if name in data_dict:
        speed_data = data_dict[name]
        t = np.arange(len(speed_data))
        style = styles.get(name)
        interval = max(1, len(speed_data) // 25) 
        
        ax.plot(t, speed_data, 
                label=style["label"],
                color=style["color"],
                linewidth=style["lw"],
                marker=style["marker"],
                markersize=style["ms"],
                markevery=interval,
                markerfacecolor=style["color"],
                markeredgecolor='white',
                markeredgewidth=1.0,
                alpha=1.0,
                zorder=style["zorder"])

        if len(speed_data) > 1:
            add_gradient_fill(ax, t, speed_data, style["color"])

# ====== 7. 添加美化的竖线 (t=67) ======
if VLINE_X < max_step:
    ax.axvline(x=VLINE_X, 
               color='#333333',     
               linestyle='--',      
               linewidth=2.0,       
               alpha=0.8,           
               zorder=3)            
    
    y_limits = ax.get_ylim()
    text_y = y_limits[1] * 0.92
    
    ax.text(VLINE_X - 1.0, 
            text_y, 
            f't = {VLINE_X}', 
            color='#333333', 
            fontsize=24, 
            fontname='Times New Roman',
            fontweight='bold',
            ha='right', 
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

# ====== 8. 布局优化 ======
ax.set_xlim(left=0, right=max_step-1)
ax.margins(x=0)
ax.set_ylim(bottom=0)

ax.grid(True, which='major', linestyle=':', linewidth=1.0, color='#808080', alpha=0.5, zorder=0)

ax.set_xlabel("Time Step", fontsize=22, labelpad=10)
ax.set_ylabel("Speed (m/s)", fontsize=22, labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=18, direction='in', length=6)

ax.legend(fontsize=16, loc="upper left", frameon=True, edgecolor='black', fancybox=False, framealpha=0.95)

plt.tight_layout()

# ====== 9. 保存图片 ======
png_filename = f"Velocity_Ep{ep_name}_v7.png"
pdf_filename = f"Velocity_Ep{ep_name}_v7.pdf"

plt.savefig(png_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"Image saved to: {os.path.abspath(png_filename)}")

plt.savefig(pdf_filename, format='pdf', bbox_inches='tight', pad_inches=0.1)
print(f"PDF saved to:   {os.path.abspath(pdf_filename)}")

plt.show()