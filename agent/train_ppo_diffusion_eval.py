"""
DPPO fine-tuning.

"""
import os
import time
import h5py
import numpy as np
import datetime
import json
import subprocess
import psutil
import os
import sys
import pretty_errors
import logging
import torch
import torch.nn.functional as F
import math
import hydra
from omegaconf import OmegaConf
import gdown
import time
# allows arbitrary python code execution in configs using the ${eval:''} resolver

import sys

sys.path.append('../')
import time
import torch
import numpy as np
import psutil
import subprocess
from configs.init_configs import get_argument, set_configs


from copy import deepcopy
from glob import glob
import gym
from env.carla_env_collect import InterSection

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import math
from ddiffpg.replay.simple_replay import create_buffer, DiffusionReplayBuffer
import numpy as np

def mean_std(x):
    x = np.asarray(x)
    return np.mean(x), np.std(x, ddof=1)

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.train_ppo_agent import TrainPPOAgent
from util.scheduler import CosineAnnealingWarmupRestarts
from ddiffpg.replay.diffusion_replay import DiffusionGoalBuffer
from ddiffpg.utils.intrinsic import IntrinsicM
from ddiffpg.utils.torch_util import soft_update
from model.diffusion.policy_v1 import Represent_Learner
import torch.optim as optim


class TrainPPODiffusionAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.load('/home/codon/Desktop/1/2026-01-02_16-23-17_42/checkpoint', 160)
        self.itr = 0


    def process_prev_obs(self, prev_obs_venv):
        """
        统一处理来自不同格式的 prev_obs_venv 输入，返回 torch.Tensor 格式的 cond 字典。

        支持三种输入格式：
        1. list[tuple(len=3)] ：批量多环境输入
        2. tuple(len=3)      ：单环境输入
        3. dict              ：上游已处理为字典格式（仅保留关键键）

        参数:
            prev_obs_venv : list | tuple | dict
                原始输入观测
            device : torch.device 或 str
                张量目标设备，例如 "cuda" 或 "cpu"

        返回:
            cond : dict[str, torch.Tensor]
                处理后的输入，键包括：
                    - "neighbor_trajs"
                    - "ego_state"
                    - "neighbor_waypoints"
        """

        device = self.device
        if isinstance(prev_obs_venv, list) and len(prev_obs_venv) > 0:
            first = prev_obs_venv[0]
            # list[tuple(len=3)] → 批量堆叠成三个 ndarray
            if isinstance(first, tuple) and len(first) == 3:
                neighbor_trajs = np.stack([np.asarray(x[0]) for x in prev_obs_venv], axis=0)
                ego_state = np.stack([np.asarray(x[1]) for x in prev_obs_venv], axis=0)
                neighbor_wps = np.stack([np.asarray(x[2]) for x in prev_obs_venv], axis=0)
                prev_obs_venv = {
                    "neighbor_trajs": neighbor_trajs,
                    "ego_state": ego_state,
                    "neighbor_waypoints": neighbor_wps,
                }
            else:
                raise TypeError(f"Expected list of 3-tuples, got list of {type(first)}")

        elif isinstance(prev_obs_venv, tuple) and len(prev_obs_venv) == 3:
            # 单环境 3 元组 → 加 batch 维
            neighbor_trajs = np.asarray(prev_obs_venv[0])[None, ...]
            ego_state = np.asarray(prev_obs_venv[1])[None, ...]
            neighbor_wps = np.asarray(prev_obs_venv[2])[None, ...]
            prev_obs_venv = {
                "neighbor_trajs": neighbor_trajs,
                "ego_state": ego_state,
                "neighbor_waypoints": neighbor_wps,
            }

        elif isinstance(prev_obs_venv, dict):
            # 若上游已是 dict，则仅保留并转为 ndarray（其它键丢弃）
            keep = ("neighbor_trajs", "ego_state", "neighbor_waypoints")
            prev_obs_venv = {k: np.asarray(prev_obs_venv[k]) for k in keep if k in prev_obs_venv}

        else:
            raise TypeError(f"Unsupported prev_obs_venv type: {type(prev_obs_venv)}")

        # 转为 torch.Tensor 并放入设备
        cond = {
            "neighbor_trajs": torch.from_numpy(prev_obs_venv["neighbor_trajs"]).float().to(device),
            "ego_state": torch.from_numpy(prev_obs_venv["ego_state"]).float().to(device),
            "neighbor_waypoints": torch.from_numpy(prev_obs_venv["neighbor_waypoints"]).float().to(device),
        }

        return cond, prev_obs_venv

    def collect(self):
        ACTION_SPACE = 2
        print("[INFO] 环境准备完成，开始采集数据...")
        # =============== 工具函数 ===============
        def _create_vlen_dataset(grp, name, list_of_lists, dtype=np.float32, compression=None):
            """
            在 grp 下创建一个名为 name 的 vlen 数据集，内容是 list of 1D lists。
            自动把标量转为长度为1的一维数组，避免 h5py vlen 写入报错。
            """
            vlen_dt = h5py.vlen_dtype(np.dtype(dtype))
            data = np.empty(len(list_of_lists), dtype=object)
            for i, item in enumerate(list_of_lists):
                arr = np.asarray(item, dtype=dtype)
                if arr.ndim == 0:  # 标量 -> [scalar]
                    arr = arr.reshape(1)
                elif arr.ndim > 1:  # 高维 -> 扁平
                    arr = arr.reshape(-1)
                data[i] = arr
            return grp.create_dataset(name, data=data, dtype=vlen_dt, compression=compression)


        # =============== 数据集保存器 ===============
        class OfflineDatasetSaver:
            def __init__(self, save_dir="./offline_data", env_name="CARLA_Intersection"):
                os.makedirs(save_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.file_path = os.path.join(save_dir, f"offline_dataset_{timestamp}.hdf5")
                self.h5 = h5py.File(self.file_path, "a")
                self.h5.create_group("episodes")
                self.episode_count = 0
                

                self.meta = {
                    "env_name": env_name,
                    "created_time": timestamp,
                    "episodes": [],
                }
                self.meta_path = os.path.join(save_dir, f"meta_{timestamp}.json")

            def add_episode(self, episode_data):
                ep_name = f"{self.episode_count:05d}"
                grp = self.h5["episodes"].create_group(ep_name)

                try:
                    # ---------- 主数据 ----------
                    # reward 改为普通 1D float 数组
                    if "reward" in episode_data:
                        grp.create_dataset(
                            "reward",
                            data=np.asarray(episode_data["reward"], dtype=np.float32),
                            compression="gzip"
                        )

                    # vlen: 每步向量（ego_state、action、waypoints 等）\
                    vehicle_keys = [k for k in episode_data.keys()     if k.startswith('ego') or k.startswith('v_')]
                    for k in [
                         "neighbor_waypoints", "next_ego_state",
                        "next_neighbor_waypoints", "action", "sim"
                    ] :
                        if k in episode_data:
                            lst = [np.asarray(x).ravel().tolist() for x in episode_data[k]]
                            _create_vlen_dataset(grp, k, lst, dtype=np.float32)
                    for k in vehicle_keys:
                        if k in episode_data:
                            lst = [np.asarray(x).ravel().tolist() for x in episode_data[k]]
                            _create_vlen_dataset(grp, k, lst, dtype=np.float32)

                    # vlen: 列表的列表（neighbor_trajs）
                    for k in ["neighbor_trajs", "next_neighbor_trajs", "chains"]:
                        if k in episode_data:
                            step_lists = []
                            for step in episode_data[k]:
                                arr = np.asarray(step)
                                if arr.ndim > 1:
                                    step_lists.append(arr.reshape(-1).tolist())
                                else:
                                    step_lists.append(arr.tolist())
                            _create_vlen_dataset(grp, k, step_lists, dtype=np.float32)

                    # 步标签 (int)
                    for k in ["finish", "collision", "off_route", "max_time"]:
                        if k in episode_data:
                            grp.create_dataset(k, data=np.asarray(episode_data[k], dtype=np.int32))

                    # done
                    if "done" in episode_data:
                        grp.create_dataset("done", data=np.asarray(episode_data["done"], dtype=np.int8))

                    # success_flag
                    if "success_flag" in episode_data:
                        grp.create_dataset("success_flag", data=np.asarray([int(bool(episode_data["success_flag"]))], dtype=np.int8))

                    # label_info 保存为 JSON 字符串
                    if "label_info" in episode_data:
                        dt = h5py.string_dtype(encoding="utf-8")
                        grp.create_dataset("label_info", data=json.dumps(episode_data["label_info"], ensure_ascii=False), dtype=dt)

                    # ---------- meta 信息 ----------
                    reward_arr = np.asarray(episode_data.get("reward", []), dtype=np.float32)
                    reward_sum = float(reward_arr.sum()) if reward_arr.size > 0 else 0.0
                    success_flag = bool(episode_data.get("success_flag", False))
                    label_info = episode_data.get("label_info", {"finish": 0, "collision": 0, "off_route": 0, "max_time": 0})

                    self.meta["episodes"].append({
                        "id": self.episode_count,
                        "length": int(len(episode_data.get("reward", []))),
                        "reward_sum": reward_sum,
                        "success": success_flag,
                        "label_info": label_info,
                    })

                    self.episode_count += 1
                    self.h5.flush()

                except Exception as e:
                    print(f"[ERROR] 写入 episode {ep_name} 失败: {e}")
                    # 删除坏 group，防止文件损坏
                    try:
                        del self.h5["episodes"][ep_name]
                        self.h5.flush()
                    except Exception as e2:
                        print(f"[WARN] 清理失败: {e2}")
                    raise

            def close(self):
                try:
                    if self.h5:
                        self.h5.attrs["num_episodes"] = self.episode_count
                        self.h5.close()
                    with open(self.meta_path, "w", encoding="utf-8") as f:
                        json.dump(self.meta, f, indent=2, ensure_ascii=False)
                    print(f"[INFO] 数据集已保存至 {self.file_path}")
                except Exception as e:
                    print(f"[WARN] 关闭文件时出现错误: {e}")

            def __del__(self):
                try:
                    self.close()
                except:
                    pass
        num_episodes=200
        max_steps=260
        save_dir="/home/codon/github/AAA_dppo/script/offline_data"

        saver = OfflineDatasetSaver(save_dir=save_dir, env_name="CARLA_Intersection")

        try:
            for ep in range(num_episodes):
                options_venv = [{} for _ in range(self.n_envs)]

                obs = self.reset_env_all(options_venv=options_venv)

                episode_data = {
                    "neighbor_trajs": [],
                    "ego_state": [],
                    "neighbor_waypoints": [],
                    "action": [],
                    "reward": [],
                    "next_neighbor_trajs": [],
                    "next_ego_state": [],
                    "next_neighbor_waypoints": [],
                    "done": [],
                    "finish": [],
                    "collision": [],
                    "off_route": [],
                    "max_time": [],
                    "vehicle_state_dict": [], # 添加一个字段记录车辆状态
                    "sim": [],
                    "chains": [],

                }

                total_reward = 0.0
                success_flag = False
                terminal_label = {"finish": 0, "collision": 0, "off_route": 0, "max_time": 0}
                step_count = 0

                for _ in range(max_steps):
                    with (torch.no_grad()):
                        cond, obs = self.process_prev_obs(obs)

                        # cond   (n_envs, n_cond_step, obs_dim) for furniture

                        # cond["state"] (n_envs, n_cond_step, obs_dim) for furniture
                        samples = self.model(
                            cond=cond,
                            deterministic=True,
                            return_chain=True,
                        )


                        output_venv = (
                            samples.trajectories.cpu().numpy()
                        )

                        # (n_envs, horizon_steps, action_dim)
                        chains_venv = (
                            samples.chains.cpu().numpy()
                        )
                        action = output_venv[:, : self.act_steps]


                    (
                        next_obs,
                        reward,
                        done,
                        info,
                    ) = self.venv.step(action)
                    vehicle_state_dict = info[0][-1]  # 获取车辆状态字典  



                    total_reward += reward
                    step_count += 1
    


                    _, po = self.process_prev_obs(obs)
                    _, pn = self.process_prev_obs(next_obs)
                    
                    prev_sim, _ = self.model.encoder(
                        po['neighbor_trajs'],
                        mask=None,
                        test=po,
                        init_state=po['ego_state'],
                        map_state=po['neighbor_waypoints']
                    )
                    next_sim, _ = self.model.target_encoder(
                        pn['neighbor_trajs'],
                        mask=None,
                        test=pn,
                        init_state=pn['ego_state'],
                        map_state=pn['neighbor_waypoints']
                    )


                    # 假设 prev_sim 和 next_sim 是编码后的特征向量 [batch_size, feature_dim]
                    cos_sim = F.cosine_similarity(prev_sim, next_sim, dim=-1)  # 计算余弦相似度 [-1,1]

                    # 将相似度归一化到[0,1]区间（原余弦相似度范围是[-1,1]）
                    norm_sim = (cos_sim + 1) / 2  # 现在范围是[0,1]

                    # 转化为不确定性：相似→0，不相似→1
                    uncertainty = 1 - norm_sim  # 反转相似度得到不确定性
                    for vehicle_key, current_state in vehicle_state_dict.items():
                        # 如果episode_data中还没有这个车辆的key，就初始化空列表
                        if vehicle_key not in episode_data:
                            episode_data[vehicle_key] = []
                        
                        # 将当前帧的状态追加到对应车辆的列表中
                        episode_data[vehicle_key].append(current_state)
                    
                    episode_data["chains"].append(chains_venv[0, :, 0, :])

                    episode_data["sim"].append(uncertainty.detach().cpu().numpy().tolist())
                    episode_data["neighbor_trajs"].append(po["neighbor_trajs"])
                    episode_data["ego_state"].append(po["ego_state"])
                    episode_data["neighbor_waypoints"].append(po["neighbor_waypoints"])
                    episode_data["action"].append(action)
                    episode_data["reward"].append(reward)
                    episode_data["next_neighbor_trajs"].append(pn["neighbor_trajs"])
                    episode_data["next_ego_state"].append(pn["ego_state"])
                    episode_data["next_neighbor_waypoints"].append(pn["neighbor_waypoints"])
                    episode_data["done"].append(int(bool(done)))

                    if done:


                        if isinstance(info, (list, tuple)) and len(info) == 4:
                            finish, collision, off_route, max_time = info
                            terminal_label = {
                                "finish": int(finish),
                                "collision": int(collision),
                                "off_route": int(off_route),
                                "max_time": int(max_time),
                            }
                            success_flag = bool(finish and not collision and not off_route)
                        break

                    obs = next_obs

                for k, v in terminal_label.items():
                    episode_data[k] = [int(v)] * step_count

                episode_data["label_info"] = terminal_label
                episode_data["success_flag"] = success_flag

                saver.add_episode(episode_data)



                print(f"[EP {ep+1:03d}] steps={step_count:3d} reward={total_reward.item():8.3f} label={terminal_label}")

        finally:
            saver.close()
            print("[INFO] 所有采样完成 ✅，CARLA 已关闭。")


    def run(self):
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        done_venv = np.zeros((1, self.n_envs))
        success_log = []  # 成功记录
        reward_log = []
        count_log = []
        speed_log = []
        time_log = []



        import h5py

        import os


        # ====== 配置保存路径 ======
        save_h5_path = "./diffusion_chains_log2.hdf5"

        # 如果文件已存在，先删除，确保是从头开始存
        if os.path.exists(save_h5_path):
            os.remove(save_h5_path)
            print(f"Old file removed: {save_h5_path}")

        h5_path = "/home/codon/github/AAA_dppo/script/offline_data/offline_dataset_20260107_161451.hdf5"
        idx = 20 + 90  # TRIM_START + TARGET_STEP (real index 110)

        with h5py.File(h5_path, "r") as f:
            grp = f["episodes"]["00016"]
            
            # 一气呵成：读取 -> 展平 -> 重塑 -> 存字典
            miubi = {
                "ego_state":          np.asarray(grp["ego_state"][idx]).reshape(1, 5),
                "neighbor_trajs":     np.asarray(grp["neighbor_trajs"][idx]).reshape(1, 6, 10, 5),
                "neighbor_waypoints": np.asarray(grp["neighbor_waypoints"][idx]).reshape(1, 18, 10, 2),
            }





        while self.itr < self.n_train_itr:
            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]


            # Define train or eval - all envs restart
            eval_mode = True
            self.model.eval()
            last_itr_eval = eval_mode

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            # (n_steps + 1, n_envs) for furniture  是否done
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                # (1, 5) (1, 6, 10, 5) (1, 18, 10, 2)
                # print(prev_obs_venv['ego_state'].shape, prev_obs_venv['neighbor_trajs'].shape, prev_obs_venv['neighbor_waypoints'].shape)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv
            # prev_obs_venv["state"]  #  (n_envs, n_cond_step, obs_dim) for furniture
            # Holder
            obs_trajs = {
                "neighbor_trajs": np.zeros(
                    (self.n_steps, self.n_envs, 6, 10, 5), dtype=np.float32
                ),
                "ego_state": np.zeros(
                    (self.n_steps, self.n_envs, 5), dtype=np.float32
                ),
                "neighbor_waypoints": np.zeros(
                    (self.n_steps, self.n_envs, 18, 10, 2), dtype=np.float32
                ),
            }

            # (n_steps, n_envs, n_cond_step, obs_dim) for furniture
            chains_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            # (n_steps, n_envs, denoising_steps + 1, horizon_steps, action_dim) for furniture
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            # (n_steps, n_envs) for furniture
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            # (n_steps, n_envs) for furniture
            # Collect a set of trajectories from env
            uncertainty = 1.0  # float: 1.0 = fully uncertain → run all denoising steps
            for step in range(self.n_steps):
                with (torch.no_grad()):
                    

                    # 保证 prev_obs_venv 是 dict[str -> np.ndarray]
                    cond, prev_obs_venv = self.process_prev_obs(prev_obs_venv)
                    cond, prev_obs_venv = self.process_prev_obs(miubi)


                    # cond   (n_envs, n_cond_step, obs_dim) for furniture
                    start = time.time()
                    # cond["state"] (n_envs, n_cond_step, obs_dim) for furniture
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        return_chain=True,
                        start_x=uncertainty,
                    )
                    chains_np = samples.chains.detach().cpu().numpy()

                    if chains_np.ndim == 4:
                        # 方式A: 使用 squeeze 去掉维度为1的轴 (更通用)
                        chains_np = chains_np.squeeze(axis=2) 
                        # 方式B: 暴力 reshape (针对你现在的形状)
                        # chains_np = chains_np.reshape(chains_np.shape[0], 21, 2)

                    # 打印一下确认形状 (应该是 (1, 21, 2))
                    # print(f"Saving chains shape: {chains_np.shape}") 

                    # 3. 追加写入 HDF5
                    with h5py.File(save_h5_path, 'a') as f:
                        if 'chains' not in f:
                            # --- 情况 A: 第一次写入 (创建数据集) ---
                            # data=chains_np (3维)
                            # maxshape=(None, 21, 2) (3维) -> 维度匹配了，不会报错了！
                            f.create_dataset('chains', data=chains_np, 
                                            maxshape=(None, chains_np.shape[1], chains_np.shape[2]), 
                                            chunks=True)
                        else:
                            # --- 情况 B: 后续写入 (追加) ---
                            dset = f['chains']
                            
                            # 扩展空间
                            prev_len = dset.shape[0]
                            new_len = prev_len + chains_np.shape[0]
                            dset.resize(new_len, axis=0)
                            
                            # 填入数据
                            dset[prev_len:] = chains_np

                    end = time.time()
                    time_log.append(1000*(end - start))
                    # samples.trajectories.shape   (n_envs, horizon_steps, action_dim)
                    # samples.chains.shape  (n_envs, denoising_steps + 1, horizon_steps, action_dim)

                    output_venv = (
                        samples.trajectories.cpu().numpy()
                    )

                    # (n_envs, horizon_steps, action_dim)
                    chains_venv = (
                        samples.chains.cpu().numpy()
                    )
                    # (n_envs, denoising_steps + 1, horizon_steps, action_dim)
                    action_venv = output_venv[:, : self.act_steps]



                (
                    obs_venv,
                    reward_venv,
                    terminated_venv,
                    info_venv,
                ) = self.venv.step(action_venv)








                reward_venv = reward_venv * self.reward_scale

                target_action_venv = action_venv
                done_venv = terminated_venv
                _, obs_venv = self.process_prev_obs(obs_venv)


                po = prev_obs_venv 
                pn = obs_venv
                prev_sim, _ = self.model.encoder(
                    po['neighbor_trajs'],
                    mask=None,
                    test=po,
                    init_state=po['ego_state'],
                    map_state=po['neighbor_waypoints']
                )
                next_sim, _ = self.model.target_encoder(
                    pn['neighbor_trajs'],
                    mask=None,
                    test=pn,
                    init_state=pn['ego_state'],
                    map_state=pn['neighbor_waypoints']
                )


                # 假设 prev_sim 和 next_sim 是编码后的特征向量 [batch_size, feature_dim]
                cos_sim = F.cosine_similarity(prev_sim, next_sim, dim=-1)  # 计算余弦相似度 [-1,1]

                # 将相似度归一化到[0,1]区间（原余弦相似度范围是[-1,1]）
                norm_sim = (cos_sim + 1) / 2  # 现在范围是[0,1]

                # 转化为不确定性：相似→0，不相似→1
                uncertainty = 1 - norm_sim  # 反转相似度得到不确定性
                un_min, un_max = 0.00011639595031738281, 0.19192007184028628
                uncertainty_scaled = (uncertainty - un_min) / (un_max - un_min)

                # ====== 限制不确定性在 [0, 1] 范围内 ======
                # 将 uncertainty_scaled 从 GPU 转移到 CPU，并转换为 numpy 数组
                uncertainty_scaled = uncertainty_scaled.detach().cpu().numpy()

                # 然后执行 numpy 的 clip 操作
                uncertainty = np.clip(uncertainty_scaled, 0, 1)

                

                # -------- 动作/奖励/终止 也统一为 ndarray（防止后续拼接问题） --------
                if isinstance(action_venv, torch.Tensor):
                    action_venv = action_venv.detach().cpu().numpy()
                action_venv = np.asarray(action_venv)

                if isinstance(target_action_venv, torch.Tensor):
                    target_action_venv = target_action_venv.detach().cpu().numpy()
                target_action_venv = np.asarray(target_action_venv)

                reward_venv = np.asarray(reward_venv)
                done_venv = np.asarray(done_venv)


                obs_trajs["neighbor_trajs"][step] = prev_obs_venv["neighbor_trajs"]
                obs_trajs["ego_state"][step] = prev_obs_venv["ego_state"]
                obs_trajs["neighbor_waypoints"][step] = prev_obs_venv["neighbor_waypoints"]


                # (n_steps, n_envs, n_cond_step, obs_dim) for furniture


                chains_trajs[step] = chains_venv
                # (n_steps, n_envs, denoising_steps + 1, horizon_steps, action_dim) for furniture

                reward_trajs[step] = reward_venv
                # (n_steps, n_envs) for furniture

                terminated_trajs[step] = terminated_venv
                # (n_steps, n_envs) for furniture

                firsts_trajs[step + 1] = done_venv
                # (n_steps + 1, n_envs) for furniture  是否done

                # update for next step
                prev_obs_venv = obs_venv

                # count steps --- not acounting for done within action chunk
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0
                if terminated_venv[0] == True:
                    prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                    firsts_trajs[0] = 1
                    success_log.append(1 if info_venv[0][0] else 0)
                    count_log.append(info_venv[0][4] if info_venv[0][0] else 0)
                    speed_log.append(info_venv[0][5] if info_venv[0][0] else 0)
                    break

            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            # [(0, 21, 87), (0, 88, 209), (0, 210, 331), (0, 332, 444)]
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                reward_log.append(episode_reward*100)
                if (
                    self.furniture_sparse_reward
                ):  # only for furniture tasks, where reward only occurs in one env step
                    episode_best_reward = episode_reward
                else:
                    episode_best_reward = np.array(
                        [
                            np.max(reward_traj) / self.act_steps
                            for reward_traj in reward_trajs_split
                        ]
                    )
                avg_episode_reward = np.mean(episode_reward)
                # 回合平均奖励
                avg_best_reward = np.mean(episode_best_reward)

            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                log.info("[WARNING] No episode completed within the iteration!")



            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            
            success_mean, success_std = mean_std(success_log)
            reward_mean, reward_std = mean_std(reward_log)
            count_mean, count_std = mean_std(count_log)
            speed_mean, speed_std = mean_std(speed_log)
            time_mean, time_std = mean_std(time_log)


            log.info(
                f"epoch: {self.itr:8.0f} | eval | "
                f"success {success_mean*100:.3f} ± {success_std*10:.3f} | "
                f"reward {reward_mean:.2f} ± {reward_std:.2f} | "
                f"count {count_mean:.2f} ± {count_std:.2f} | "
                f"speed {speed_mean:.2f} ± {speed_std:.2f}"
                f"time {time_mean:.4f} ± {time_std:.4f}"
            )

           
            self.itr += 1
