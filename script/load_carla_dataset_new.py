import os
import h5py
import json
import numpy as np
import random
from ddiffpg.replay.simple_replay import DiffusionReplayBuffer


class HDF5FileProcessor:
    """处理单个HDF5文件，解析并导入到DiffusionReplayBuffer"""

    def __init__(self, h5_path, diffusion_buffer):
        assert os.path.exists(h5_path), f"HDF5文件不存在: {h5_path}"
        self.h5_path = h5_path
        self.buffer = diffusion_buffer
        self.h5 = h5py.File(h5_path, "r")

    def load_episode(self, ep, ep_name=None, filter_success_only=False):
        """解析单个episode并写入buffer"""
        label_info = None
        # 可选: 判断成功/失败过滤
        if "label_info" in ep:
            try:
                label_info = json.loads(ep["label_info"][()].decode("utf-8"))
                if filter_success_only and not label_info.get("finish", 0):
                    # print(f"[SKIP] Episode {ep_name} 未完成 (finish=0)") if ep_name else None
                    return 0
            except Exception:
                pass

        num_steps = len(ep["reward"])
        # if ep_name:
        #     print(f"[INFO] 导入 Episode {ep_name} ({num_steps} 步)...")

        # 读取主要字段
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
            # 打包trajectory数据
            prev_obs_venv = {
                "neighbor_trajs": np.asarray(neighbor_trajs[t]),
                "ego_state": np.asarray(ego_state[t]),
                "neighbor_waypoints": np.asarray(neighbor_wps[t]),
            }
            obs_venv = {
                "neighbor_trajs": np.asarray(next_neighbor_trajs[t]),
                "ego_state": np.asarray(next_ego_state[t]),
                "neighbor_waypoints": np.asarray(next_neighbor_wps[t]),
            }
            trajectory = (
                prev_obs_venv,
                np.asarray(action[t]),  # action_venv
                np.copy(np.asarray(action[t])),  # target_action_venv
                np.asarray(reward[t]),  # reward_venv
                obs_venv,
                np.array([done[t]], dtype=np.float32),  # done_venv
            )
            self.buffer.add_to_buffer(trajectory)

        # if ep_name:
        #     print(f"[OK] Episode {ep_name} 导入完成")
        return num_steps

    def load_all_episodes(self, filter_success_only=False):
        """加载文件中的所有episodes"""
        episodes = self.h5["episodes"]
        print(f"[INFO] 共检测到 {len(episodes)} 个episodes")
        total_steps = 0
        for ep_name in sorted(episodes.keys()):
            total_steps += self.load_episode(episodes[ep_name], ep_name, filter_success_only)
        # self._print_buffer_status()
        return total_steps

    def load_random_episodes(self, max_episodes, filter_success_only=False):
        """随机加载部分episodes"""
        episodes = self.h5["episodes"]
        selected_episodes = random.sample(list(episodes.keys()), min(max_episodes, len(episodes)))
        print(f"[INFO] 随机选择导入 {len(selected_episodes)} 个episode")
        total_steps = 0
        for ep_name in selected_episodes:
            total_steps += self.load_episode(episodes[ep_name], ep_name, filter_success_only)
        # self._print_buffer_status()
        return total_steps

    def _print_buffer_status(self):
        """打印buffer当前状态"""
        size = getattr(self.buffer, "size", None) or \
               getattr(self.buffer, "current_size", None) or \
               getattr(self.buffer, "ptr", None)
        if size is None:
            print(f"[INFO] 导入完成")
        else:
            print(f"[INFO] 导入完成，共 {size} 条样本")

    def close(self):
        self.h5.close()
        print(f"[INFO] 文件 {self.h5_path} 已关闭")


class HDF5BatchImporter:
    """批量处理目录下的多个HDF5文件"""

    def __init__(self, diffusion_buffer):
        self.buffer = diffusion_buffer

    @staticmethod
    def _get_h5_files(data_dir):
        """获取目录下所有HDF5文件"""
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                 if f.endswith((".hdf5", ".h5"))]
        if not files:
            raise FileNotFoundError(f"未在 {data_dir} 中找到HDF5文件")
        return files

    def import_single_random_file(self, data_dir="/home/codon/github/AAA_dppo/script/offline_data", max_episodes=None, filter_success_only=False):
        """随机导入单个文件中的部分episodes"""
        h5_file = random.choice(self._get_h5_files(data_dir))
        print(f"[INFO] 随机选择导入文件: {h5_file}")

        processor = HDF5FileProcessor(h5_file, self.buffer)
        if max_episodes:
            processor.load_random_episodes(max_episodes, filter_success_only)
        else:
            processor.load_all_episodes(filter_success_only)
        final_size = processor._print_buffer_status()  # 获取最终buffer大小

        processor.close()
        print(f"[DONE] 导入完成，Buffer 最终大小: {final_size}")
        return final_size  # 返回buffer大小

    def import_multiple_files(self, data_dir="/home/codon/github/AAA_dppo/script/offline_data", n_files=None, filter_success_only=False):
        """批量导入多个文件"""
        all_files = self._get_h5_files(data_dir)
        selected_files = random.sample(all_files, min(n_files, len(all_files))) if n_files else all_files

        print(f"[INFO] 共找到 {len(all_files)} 个文件，选择导入 {len(selected_files)} 个:")
        for f in selected_files:
            print("   →", os.path.basename(f))

        total_steps = 0
        for idx, h5_file in enumerate(selected_files, 1):
            print(f"\n[LOAD {idx}/{len(selected_files)}] 正在导入 {os.path.basename(h5_file)}...")
            processor = HDF5FileProcessor(h5_file, self.buffer)
            total_steps += processor.load_all_episodes(filter_success_only)
            processor.close()

        final_size = getattr(self.buffer, "size", None) or \
                     getattr(self.buffer, "current_size", None) or \
                     getattr(self.buffer, "ptr", None)

        print(f"\n[DONE] ✅ 全部导入完成")
        print(f"   → 总步数: {total_steps}")
        print(f"   → Buffer 最终大小: {final_size}")
        return final_size  # 返回buffer大小


# 使用示例
if __name__ == "__main__":
    # 初始化buffer
    diffusion_buffer = DiffusionReplayBuffer(
        capacity=int(2000000),
        obs_dim=128,
        action_dim=2,
        device="cuda:0",
        cond_steps=1,
        horizon_steps=1
    )

    # 示例1: 处理单个文件
    # processor = HDF5FileProcessor("./offline_data/sample.hdf5", diffusion_buffer)
    # processor.load_all_episodes(filter_success_only=True)
    # processor.close()

    # 示例2: 批量处理
    batch_importer = HDF5BatchImporter(diffusion_buffer)
    # 随机导入单个文件中的20个episode
    # batch_importer.import_single_random_file("/home/codon/github/AAA_dppo/script/offline_data_old", max_episodes=20)
    # 或者导入目录下的多个文件
    batch_importer.import_multiple_files(n_files=10, filter_success_only=True)