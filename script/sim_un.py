import h5py
import numpy as np
import matplotlib.pyplot as plt

# ====== 配置 ======
h5_path = "/home/codon/github/AAA_dppo/script/offline_data/offline_dataset_20260107_100712.hdf5"  # 改成你的文件名
ep_ids = [11, 12, 13]  # 提取多个 ep_id
sim_all = []

# ====== 读取 sim 数据 ======
with h5py.File(h5_path, "r") as f:
    for ep_id in ep_ids:
        ep_name = f"{ep_id:05d}"  # 格式化 episode 名称
        grp = f["episodes"][ep_name]
        sim_ds = grp["sim"]

        sim = []
        for i in range(len(sim_ds)):
            arr = np.asarray(sim_ds[i]).reshape(-1)  # 变成 1D
            sim.append(float(arr[0]))  # 取标量
        sim = np.array(sim, dtype=np.float32)

        sim_all.extend(sim)  # 将当前 ep_id 的 sim 数据添加到全局 sim_all

# ====== 合并所有 ep_id 的 sim 数据 ======
sim_all = np.array(sim_all)


print("All sim length:", sim_all.shape[0], " raw range:", float(sim_all.min()), "~", float(sim_all.max()))

# ====== Min-Max 归一化到 [0,1] ======
smin = float(sim_all.min())
smax = float(sim_all.max())
if (smax - smin) > 1e-12:
    sim_norm = (sim_all - smin) / (smax - smin)
else:
    sim_norm = np.zeros_like(sim_all)

print("norm range:", float(sim_norm.min()), "~", float(sim_norm.max()))

# ====== 计算 95% 界限的最值 ======
lower_bound = np.percentile(sim_all, 2.5)  # 2.5% 分位数
upper_bound = np.percentile(sim_all, 97.5)  # 97.5% 分位数
print(f"95% range: [{lower_bound}, {upper_bound}]")

# ====== 画图：横轴=时间步，纵轴=不确定性 ======
# sim_norm[90] = 1
t = np.arange(len(sim_norm), dtype=np.int32)

plt.figure()
plt.plot(t, sim_norm)
plt.xlabel("time step t")
plt.ylabel("uncertainty (sim, normalized)")
plt.title(f"Uncertainty over Time - Episodes {ep_ids}")
plt.grid(True, alpha=0.3)
plt.show()
