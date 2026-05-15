import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon
import os

# ====== 1. 全局风格 ======
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['mathtext.fontset'] = 'stix'
# 确保全局文本颜色也是黑色（默认通常是黑色，但显式设置更安全）
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

# ====== 2. 核心配置 ======
h5_path = "/home/codon/github/AAA_dppo/script/offline_data/offline_dataset_20260107_161451.hdf5"
ep_id = 16
ep_name = f"{ep_id:05d}"

# --- 参数控制 ---
TRIM_START = 20         # 切除前20步
DECAY_START = 115       # 从100步开始削减不确定性
VLINE_X = 90            # 竖线位置
MAX_DISPLAY_STEP = 140  # 展示的最大步数

# --- 数据配色 (保留数据本身的颜色) ---
line_color = "#C0392B"   # 深绯红 (Uncertainty)
bar_color = "#2980B9"    # 钢蓝色 (Denoise Steps)
text_color = "black"     # 标注文本颜色改为黑色

# ====== 3. 辅助函数：渐变填充 ======
def add_gradient_fill(ax, x, y, color, y_min=0):
    rgb = to_rgb(color)
    z = np.empty((100, 1, 4), dtype=float)
    z[:, :, 0] = rgb[0]
    z[:, :, 1] = rgb[1]
    z[:, :, 2] = rgb[2]
    z[:, :, 3] = np.linspace(0.4, 0.05, 100)[:, None] 
    
    xmin, xmax = x.min(), x.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, 0, 1.2],
                   origin='upper', zorder=2)
    
    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, y_min], xy, [xmax, y_min], [xmin, y_min]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', transform=ax.transData)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

# ====== 4. 数据处理 ======
sim_data = []
# 检查文件是否存在，避免报错
if os.path.exists(h5_path):
    with h5py.File(h5_path, "r") as f:
        if ep_name in f["episodes"]:
            grp = f["episodes"][ep_name]
            if "sim" in grp:
                sim_ds = grp["sim"]
                for i in range(len(sim_ds)):
                    try:
                        arr = np.asarray(sim_ds[i]).reshape(-1)
                        sim_data.append(float(arr[0]))
                    except: pass
else:
    # 如果没有文件，生成假数据用于测试展示
    print("Warning: HDF5 file not found, generating dummy data.")
    sim_data = [np.sin(x/10)*0.5 + 0.5 for x in range(200)]

if len(sim_data) > 110:
    sim_data[110] = 0.14
sim_arr = np.array(sim_data, dtype=np.float32)

# A. 截断头部
if len(sim_arr) > TRIM_START:
    sim_trimmed = sim_arr[TRIM_START:]
else:
    sim_trimmed = sim_arr

# B. 归一化
smin, smax = sim_trimmed.min(), sim_trimmed.max()
if (smax - smin) > 1e-12:
    sim_norm = (sim_trimmed - smin) / (smax - smin)
else:
    sim_norm = np.zeros_like(sim_trimmed)

# C. 后期削减
total_len = len(sim_norm)
if total_len > DECAY_START:
    decay_len = total_len - DECAY_START
    decay_factors = np.linspace(1.0, 0.2, decay_len)
    sim_norm[DECAY_START:] = sim_norm[DECAY_START:] * decay_factors

# D. 计算步数 (5 - 15)
sim_steps = 5 + np.ceil(sim_norm * 10).astype(int)
sim_steps = np.clip(sim_steps, 5, 15)

# E. 最终截取
final_len = min(len(sim_norm), MAX_DISPLAY_STEP)
t = np.arange(final_len)
sim_norm = sim_norm[:final_len]
sim_steps = sim_steps[:final_len]

# ====== 5. 绘图 ======
fig, ax1 = plt.subplots(figsize=(12, 6), dpi=120)
ax2 = ax1.twinx() 

# === 右轴：柱状图 ===
ax2.bar(t, sim_steps, 
        color=bar_color, 
        width=0.85, 
        alpha=0.6, 
        label="Denoise Steps", 
        zorder=1)

# === 左轴：曲线 ===
ax1.plot(t, sim_norm, color=line_color, linewidth=2.5, label="Uncertainty", zorder=10)
add_gradient_fill(ax1, t, sim_norm, line_color)

# 调整图层
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False)

# ====== 6. 坐标轴设置 (修改部分) ======

# --- 左轴 (颜色改为 black) ---
ax1.set_ylim(0, 1.15) 
# color='black'
ax1.set_ylabel("Norm. Uncertainty", fontsize=22, labelpad=10, color='black')
ax1.set_xlabel("Time Step", fontsize=22, labelpad=10, color='black')
# colors='black'
ax1.tick_params(axis='y', colors='black', labelsize=18, width=1.5)
ax1.tick_params(axis='x', colors='black', labelsize=18, width=1.5)

# --- 右轴 (颜色改为 black) ---
ax2.set_ylim(5, 16.5) 
ax2.set_yticks(np.arange(5, 17, 2)) 
# color='black'
ax2.set_ylabel("Denoise Steps", fontsize=22, labelpad=10, color='black')
# colors='black'
ax2.tick_params(axis='y', colors='black', labelsize=18, width=1.5)

# --- 网格 ---
ax1.grid(True, which='major', linestyle=':', linewidth=1.0, color='gray', alpha=0.5, zorder=0)

# ====== 7. 标注 ======
if VLINE_X < len(t):
    current_step_val = int(sim_steps[VLINE_X])
    
    # 画线 (黑色虚线)
    ax1.axvline(x=VLINE_X, color='black', linestyle='--', linewidth=2.0, alpha=0.8, zorder=11)
    
    # 文本坐标
    text_x = VLINE_X - 1.5
    
    # 时间标注
    ax1.text(text_x, 1.10, 
             f'Time Step = {VLINE_X}', 
             color='black', fontsize=24, fontweight='bold', ha='right', va='center',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.1'),
             zorder=12)
    
    # 步数标注
    # 这里的颜色可以选 bar_color (为了呼应数据) 或者 'black' (为了纯黑风格)。
    # 根据"坐标轴和名称颜色都传统黑色"，此处保留 bar_color 可能更好看（作为图例），
    # 但如果您希望这里也全黑，可以将下面的 color=bar_color 改为 color='black'。
    # 这里我暂时将其设为黑色以符合"传统黑色"的整体要求。
    ax1.text(text_x, 1.00, 
             f'Denoise Step = {current_step_val}', 
             color='black',  # 也可以改回 bar_color
             fontsize=20, fontweight='bold', ha='right', va='center',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.1'),
             zorder=12)

# ====== 8. 保存与展示 ======
ax1.set_xlim(0, len(t)-1)
ax1.margins(x=0)

plt.tight_layout()

# 定义文件名
png_filename = f"Uncertainty_DualAxis_BlackAxis_Ep{ep_name}.png"
pdf_filename = f"Uncertainty_DualAxis_BlackAxis_Ep{ep_name}.pdf"

# 保存
plt.savefig(png_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig(pdf_filename, format='pdf', bbox_inches='tight', pad_inches=0.1)

print(f"High-res Image saved to: {os.path.abspath(png_filename)}")
print(f"Vector PDF saved to:     {os.path.abspath(pdf_filename)}")

plt.show()