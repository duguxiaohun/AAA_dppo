import os
import h5py
import json
import numpy as np
from ddiffpg.replay.simple_replay import DiffusionReplayBuffer
import random

class OfflineDatasetImporter:
    """
    从 HDF5 离线数据集导入 trajectory，
    自动解析结构并写入 DiffusionReplayBuffer。
    """

    def __init__(self, h5_path, diffusion_buffer):
        assert os.path.exists(h5_path), f"HDF5 文件不存在: {h5_path}"
        self.h5_path = h5_path
        self.buffer = diffusion_buffer
        self.h5 = h5py.File(h5_path, "r")

    def load(self, filter_success_only=False):
        """遍历所有 episode，逐步解析并写入 diffusion buffer"""
        episodes = self.h5["episodes"]
        print(f"[INFO] 共检测到 {len(episodes)} 个 episodes")

        for ep_name in sorted(episodes.keys()):
            ep = episodes[ep_name]
            label_info = None

            # optional: 判断成功/失败过滤
            if "label_info" in ep:
                try:
                    label_info = json.loads(ep["label_info"][()].decode("utf-8"))
                    if filter_success_only and not label_info.get("finish", 0):
                        print(f"[SKIP] Episode {ep_name} 未完成 (finish=0)")
                        continue
                except Exception:
                    pass

            num_steps = len(ep["reward"])
            print(f"[INFO] 导入 Episode {ep_name} ({num_steps} 步) ...")

            # ====== 读取主要字段 ======
            neighbor_trajs = np.array(ep["neighbor_trajs"], dtype=object)
            ego_state = np.array(ep["ego_state"], dtype=object)
            neighbor_wps = np.array(ep["neighbor_waypoints"], dtype=object)
            next_neighbor_trajs = np.array(ep["next_neighbor_trajs"], dtype=object)
            next_ego_state = np.array(ep["next_ego_state"], dtype=object)
            next_neighbor_wps = np.array(ep["next_neighbor_waypoints"], dtype=object)
            action = np.array(ep["action"], dtype=object)
            reward = np.array(ep["reward"], dtype=np.float32)
            done = np.array(ep["done"], dtype=np.int8)

            for t in range(num_steps):
                # --------- prev_obs_venv ----------
                prev_obs_venv = {
                    "neighbor_trajs": np.asarray(neighbor_trajs[t]),
                    "ego_state": np.asarray(ego_state[t]),
                    "neighbor_waypoints": np.asarray(neighbor_wps[t]),
                }

                # --------- obs_venv ----------
                obs_venv = {
                    "neighbor_trajs": np.asarray(next_neighbor_trajs[t]),
                    "ego_state": np.asarray(next_ego_state[t]),
                    "neighbor_waypoints": np.asarray(next_neighbor_wps[t]),
                }

                # --------- 其他信息 ----------
                action_venv = np.asarray(action[t])
                reward_venv = np.asarray(reward[t])
                done_venv = np.array([done[t]], dtype=np.float32)  # ✅ 修复 done 类型

                # target_action 可根据训练策略决定，这里直接设为 action
                target_action_venv = np.copy(action_venv)

                # --------- 打包 trajectory ----------
                trajectory = (
                    prev_obs_venv,
                    action_venv,
                    target_action_venv,
                    reward_venv,
                    obs_venv,
                    done_venv,  # ✅ 一维 ndarray
                )

                self.buffer.add_to_buffer(trajectory)

            print(f"[OK] Episode {ep_name} 导入完成，共 {num_steps} 步。")

        # ✅ 打印 buffer 状态（兼容不同版本）
        size = getattr(self.buffer, "size", None) or \
               getattr(self.buffer, "current_size", None) or \
               getattr(self.buffer, "ptr", None)

        if size is None:
            print(f"[INFO] 导入完成。")
        else:
            print(f"[INFO] 导入完成，共 {size} 条样本。")

    def close(self):
        self.h5.close()
        print(f"[INFO] 文件 {self.h5_path} 已关闭。")

    def import_single_h5_file_randomly(self, data_dir="./offline_data", max_episodes=100000000, filter_success_only=False):
        """
        随机导入单个 HDF5 文件中的 episodes，最多导入 max_episodes 个。
        如果文件中的 episodes 少于 max_episodes，则导入所有 episodes。
        """
        # 1️⃣ 获取文件列表
        all_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".hdf5") or f.endswith(".h5")
        ]

        if len(all_files) == 0:
            raise FileNotFoundError(f"❌ 未在 {data_dir} 中找到任何 .hdf5 文件")

        # 随机选择一个文件
        h5_file = random.sample(all_files, 1)[0]
        print(f"[INFO] 随机选择导入文件：{h5_file}")

        # 2️⃣ 打开 HDF5 文件

        # 打开 HDF5 文件
        with h5py.File(h5_file, "r") as h5:
            episodes = h5["episodes"]
            total_episodes = len(episodes)
            print(f"[INFO] 文件 {h5_file} 中共有 {total_episodes} 个 episodes")

            # 随机选择最多 max_episodes 个 episode
            episodes_to_import = random.sample(list(episodes.keys()), min(max_episodes, total_episodes))
            print(f"[INFO] 随机选择导入 {len(episodes_to_import)} 个 episode")

            # 遍历选中的 episodes，解析数据并导入到 DiffusionReplayBuffer
            for ep_name in episodes_to_import:
                ep = episodes[ep_name]
                label_info = None

                # optional: 判断成功/失败过滤
                if "label_info" in ep:
                    try:
                        label_info = json.loads(ep["label_info"][()].decode("utf-8"))
                        if filter_success_only and not label_info.get("finish", 0):
                            print(f"[SKIP] Episode {ep_name} 未完成 (finish=0)")
                            continue
                    except Exception:
                        pass

                num_steps = len(ep["reward"])
                print(f"[INFO] 导入 Episode {ep_name} ({num_steps} 步) ...")

                # ====== 读取主要字段 ======
                neighbor_trajs = np.array(ep["neighbor_trajs"], dtype=object)
                ego_state = np.array(ep["ego_state"], dtype=object)
                neighbor_wps = np.array(ep["neighbor_waypoints"], dtype=object)
                next_neighbor_trajs = np.array(ep["next_neighbor_trajs"], dtype=object)
                next_ego_state = np.array(ep["next_ego_state"], dtype=object)
                next_neighbor_wps = np.array(ep["next_neighbor_waypoints"], dtype=object)
                action = np.array(ep["action"], dtype=object)
                reward = np.array(ep["reward"], dtype=np.float32)
                done = np.array(ep["done"], dtype=np.int8)

                # 遍历 episode 中的每个时间步
                for t in range(num_steps):
                    # --------- prev_obs_venv ----------
                    prev_obs_venv = {
                        "neighbor_trajs": np.asarray(neighbor_trajs[t]),
                        "ego_state": np.asarray(ego_state[t]),
                        "neighbor_waypoints": np.asarray(neighbor_wps[t]),
                    }

                    # --------- obs_venv ----------
                    obs_venv = {
                        "neighbor_trajs": np.asarray(next_neighbor_trajs[t]),
                        "ego_state": np.asarray(next_ego_state[t]),
                        "neighbor_waypoints": np.asarray(next_neighbor_wps[t]),
                    }

                    # --------- 其他信息 ----------
                    action_venv = np.asarray(action[t])
                    reward_venv = np.asarray(reward[t])
                    done_venv = np.array([done[t]], dtype=np.float32)  # ✅ 修复 done 类型

                    # target_action 可根据训练策略决定，这里直接设为 action
                    target_action_venv = np.copy(action_venv)

                    # --------- 打包 trajectory ----------
                    trajectory = (
                        prev_obs_venv,
                        action_venv,
                        target_action_venv,
                        reward_venv,
                        obs_venv,
                        done_venv,  # ✅ 一维 ndarray
                    )

                    self.buffer.add_to_buffer(trajectory)

                print(f"[OK] Episode {ep_name} 导入完成，共 {num_steps} 步。")

            # 打印 buffer 状态
            size = getattr(self.buffer, "size", None) or \
                   getattr(self.buffer, "current_size", None) or \
                   getattr(self.buffer, "ptr", None)

            if size is None:
                print(f"[INFO] 导入完成。")
            else:
                print(f"[INFO] 导入完成，共 {size} 条样本。")

    def import_random_h5_datasets(self,
            data_dir="./offline_data",
            n_files=3,
            capacity=int(2e5),
            obs_dim=128,
            action_dim=2,
            device="cuda:0",
            cond_steps=1,
            horizon_steps=1,
            filter_success_only=False,
    ):
        """
        从指定文件夹随机选择 n 个 .hdf5 文件依次导入到 DiffusionReplayBuffer。
        """


        # 2️⃣ 获取文件列表
        all_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".hdf5") or f.endswith(".h5")
        ]

        if len(all_files) == 0:
            raise FileNotFoundError(f"❌ 未在 {data_dir} 中找到任何 .hdf5 文件")

        # 随机选择 n 个文件（若数量不足，则全选）
        selected_files = random.sample(all_files, min(n_files, len(all_files)))

        print(f"[INFO] 在目录 {data_dir} 中共找到 {len(all_files)} 个数据文件")
        print(f"[INFO] 随机选取 {len(selected_files)} 个用于导入：")
        for f in selected_files:
            print("   →", f)

        # 3️⃣ 依次导入
        for idx, h5_file in enumerate(selected_files, start=1):
            print(f"\n[LOAD {idx}/{len(selected_files)}] 正在导入 {os.path.basename(h5_file)} ...")
            importer = OfflineDatasetImporter(h5_file, self.buffer)
            importer.load(filter_success_only=filter_success_only)
            importer.close()

        # 4️⃣ 打印结果
        final_size = getattr(self.buffer, "size", None) or \
                     getattr(self.buffer, "current_size", None) or \
                     getattr(self.buffer, "ptr", None)

        if final_size is None:
            print(f"[DONE] ✅ 全部导入完成。")
        else:
            print(f"[DONE] ✅ Buffer 当前样本数: {final_size}")

        return self.buffer

# =========================================
# 主入口
# =========================================
if __name__ == "__main__":
    # 初始化 DiffusionReplayBuffer
    diffusion_buffer = DiffusionReplayBuffer(
        capacity=int(1e6),
        obs_dim=128,
        action_dim=2,
        device="cuda:0",
        cond_steps=1,
        horizon_steps=1
    )

    # 指定 HDF5 文件路径
    h5_file = "./offline_data/offline_dataset_20251027_115612.hdf5"

    importer = OfflineDatasetImporter(h5_file, diffusion_buffer)
    # importer.load(filter_success_only=False)  # 若只想导入成功轨迹 → True
    # importer.close()
    #
    # # ✅ 打印最终 buffer 状态
    # final_size = getattr(diffusion_buffer, "size", None) or \
    #              getattr(diffusion_buffer, "current_size", None) or \
    #              getattr(diffusion_buffer, "ptr", None)
    #
    # if final_size is None:
    #     print(f"[DONE] Buffer 导入完成。")
    # else:
    #     print(f"[DONE] Buffer 当前样本数: {final_size}")
    importer.import_single_h5_file_randomly( data_dir="./offline_data", max_episodes=20 # 存放数据集的文件夹
)
    # importer.import_random_h5_datasets(
    #     data_dir="./offline_data",  # 存放数据集的文件夹
    #     n_files=1,  # 随机选择 5 个
    #     device="cuda:0",  # 使用 GPU
    #     filter_success_only=True  # 是否只导入成功轨迹
    # )
