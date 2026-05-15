import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw_ndim
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform


# -------- 1. 构造模拟轨迹数据 --------
def create_trajectory(length, offset=0, noise=0.1):
    t = np.linspace(0, 2 * np.pi, length)
    x = np.sin(t) + offset + np.random.normal(0, noise, size=length)
    y = np.cos(t) + offset + np.random.normal(0, noise, size=length)
    return np.stack([x, y], axis=1)  # shape: (length, 2)


trajectories = [
    create_trajectory(50, offset=0),
    create_trajectory(50, offset=0.2),
    create_trajectory(50, offset=0.5),
    create_trajectory(50, offset=3),    # 明显偏移
    create_trajectory(50, offset=3.2),
]

num_traj = len(trajectories)

# -------- 2. 计算轨迹间 DTW 距离矩阵 --------
distance_mat = np.zeros((num_traj, num_traj))

for i in range(num_traj):
    for j in range(num_traj):
        if i == j:
            distance_mat[i][j] = 0
        elif distance_mat[i][j] == 0:
            d = dtw_ndim.distance(trajectories[i], trajectories[j])
            distance_mat[i][j] = distance_mat[j][i] = d

# -------- 3. 聚类操作 --------
distance_vec = squareform(distance_mat)
# print(distance_mat)
Z = linkage(distance_vec, method='average')
# print(distance_vec)
# print(Z)
# 阈值设为最大距离的 0.7，模拟 DDiffPG 中的做法
threshold = 0.7 * max(Z[:, 2])
cluster_ids = fcluster(Z, t=threshold, criterion='distance')
# print(cluster_ids)
# -------- 4. 可视化聚类结果 --------
plt.figure(figsize=(8, 4))
colors = ['r', 'g', 'b', 'm', 'c']

for i in range(num_traj):
    x, y = trajectories[i][:, 0], trajectories[i][:, 1]
    plt.plot(x, y, color=colors[cluster_ids[i] - 1], label=f'Cluster {cluster_ids[i]}')

plt.title("Trajectory Clustering (DTW + Hierarchical)")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# 可选：绘制 dendrogram 层次树
plt.figure(figsize=(6, 3))
dendrogram(Z)
plt.title("Hierarchical Dendrogram")
plt.xlabel("Trajectory Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()
