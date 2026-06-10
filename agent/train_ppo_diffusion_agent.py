"""
DPPO fine-tuning.

"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import math
from ddiffpg.replay.simple_replay import create_buffer, DiffusionReplayBuffer

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
        # Reward horizon --- always set to act_steps for now
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)
        self.intrinsic = IntrinsicM(self.obs_dim*self.cond_steps, device=self.device)


        # Eta - between DDIM (=0 for eval) and DDPM (=1 for training)
        self.learn_eta = self.model.learn_eta
        if self.learn_eta:
            self.eta_update_interval = cfg.train.eta_update_interval
            self.eta_optimizer = torch.optim.AdamW(
                self.model.eta.parameters(),
                lr=cfg.train.eta_lr,
                weight_decay=cfg.train.eta_weight_decay,
            )
            self.eta_lr_scheduler = CosineAnnealingWarmupRestarts(
                self.eta_optimizer,
                first_cycle_steps=cfg.train.eta_lr_scheduler.first_cycle_steps,
                cycle_mult=1.0,
                max_lr=cfg.train.eta_lr,
                min_lr=cfg.train.eta_lr_scheduler.min_lr,
                warmup_steps=cfg.train.eta_lr_scheduler.warmup_steps,
                gamma=1.0,
            )


        self.rep_func = Represent_Learner(self.model.encoder, self.model.target_encoder).to(self.device)
        self.rep_optimizer = optim.NAdam(self.rep_func.parameters(), lr=0.0001)

        # Restrict the base critic_optimizer to net_v only (PPO value function).
        # net_q1/net_q2 must NOT be touched during the PPO loop — sharing the optimizer
        # causes Adam momentum from Q updates to "replay" into net_q1/q2 on every PPO
        # mini-batch, effectively undoing Q learning and preventing critic loss convergence.
        self.critic_optimizer = torch.optim.AdamW(
            self.model.critic.net_v.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        # Dedicated optimizer for the distributional Q networks (off-policy updates).
        self.critic_q_optimizer = torch.optim.AdamW(
            list(self.model.critic.net_q1.parameters()) +
            list(self.model.critic.net_q2.parameters()),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        # Number of Q-network gradient steps per iteration (standard off-policy: ~1 per env step)
        self.critic_q_update_steps = cfg.train.get("critic_q_update_steps", 20)



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
    def explore_env(self, warm_up=None):
        options_venv = [{} for _ in range(self.n_envs)]
        self.model.train()

        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))

        firsts_trajs[0] = 1
        success_log = [0]  # 成功记录

        if warm_up is None:
            warm_up = self.warm_up
        for _ in range(warm_up):


            action_venv = self.venv.action_space.sample().reshape(1, 1, -1)
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


            # -------- 动作/奖励/终止 也统一为 ndarray（防止后续拼接问题） --------
            if isinstance(action_venv, torch.Tensor):
                action_venv = action_venv.detach().cpu().numpy()
            action_venv = np.asarray(action_venv)

            if isinstance(target_action_venv, torch.Tensor):
                target_action_venv = target_action_venv.detach().cpu().numpy()
            target_action_venv = np.asarray(target_action_venv)

            reward_venv = np.asarray(reward_venv)
            done_venv = np.asarray(done_venv)

            trajectory = (prev_obs_venv, action_venv, target_action_venv, reward_venv, obs_venv, done_venv)
            self.diffusion_buffer.add_to_buffer(trajectory)
            prev_obs_venv = obs_venv

            if terminated_venv[0] == True:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1








    def run(self):
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        success_log = []  # 每个episode的0/1成功记录
        cnt_episode = 0   # 总完成episode数

        self.diffusion_buffer = DiffusionReplayBuffer(capacity=self.cfg.memory_size,
                              obs_dim=self.obs_dim,
                              action_dim=self.action_dim,
                              device=self.device,
                              cond_steps=self.cond_steps,
                              horizon_steps=self.horizon_steps)
        # self.load('/home/codon/github/AAA_dppo/baseline/checkpoint', 540)
        if 'Town03' in self.cfg.carla_env_interface:
            self.explore_env(self.warm_up)
        else:
            self.explore_env(self.warm_up)
        self.itr = 0

        while self.itr < self.n_train_itr:
            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]


            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            # (n_steps + 1, n_envs) for furniture  是否done
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
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
            for step in range(self.n_steps):
                with (torch.no_grad()):
                    # 保证 prev_obs_venv 是 dict[str -> np.ndarray]
                    cond, prev_obs_venv = self.process_prev_obs(prev_obs_venv)

                    # cond   (n_envs, n_cond_step, obs_dim) for furniture

                    # cond["state"] (n_envs, n_cond_step, obs_dim) for furniture
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        return_chain=True,
                    )
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


                # (n_envs, horizon_steps, action_dim)
                if eval_mode:

                    _, action_venv = self.model.update_target_action(prev_obs_venv, action_venv)

                    with (torch.no_grad()):
                        cond, prev_obs_venv = self.process_prev_obs(prev_obs_venv)


                        # cond   (n_envs, n_cond_step, obs_dim) for furniture

                        # cond["state"] (n_envs, n_cond_step, obs_dim) for furniture
                        samples = self.model.forward_again(
                            cond=cond,
                            deterministic=eval_mode,
                            return_chain=True,
                            start_x=action_venv,
                        )
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
                # Apply multi-step action
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

                # -------- 动作/奖励/终止 也统一为 ndarray（防止后续拼接问题） --------
                if isinstance(action_venv, torch.Tensor):
                    action_venv = action_venv.detach().cpu().numpy()
                action_venv = np.asarray(action_venv)

                if isinstance(target_action_venv, torch.Tensor):
                    target_action_venv = target_action_venv.detach().cpu().numpy()
                target_action_venv = np.asarray(target_action_venv)

                reward_venv = np.asarray(reward_venv)
                done_venv = np.asarray(done_venv)


                trajectory = (prev_obs_venv, action_venv, target_action_venv, reward_venv, obs_venv, done_venv)
                self.diffusion_buffer.add_to_buffer(trajectory)
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
                    cnt_episode += 1

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
                std_episode_reward = np.std(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                std_best_reward = np.std(episode_best_reward)
                success_arr = (episode_best_reward >= self.best_reward_threshold_for_success).astype(float)
                success_rate = np.mean(success_arr)
                std_success_rate = np.std(success_arr)
                episode_steps_arr = np.array(
                    [(end - start) * self.act_steps for _, start, end in episodes_start_end]
                )
                avg_episode_steps = np.mean(episode_steps_arr)
                std_episode_steps = np.std(episode_steps_arr)
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                std_episode_reward = 0
                avg_best_reward = 0
                std_best_reward = 0
                success_rate = 0
                std_success_rate = 0
                avg_episode_steps = 0
                std_episode_steps = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # Update models
            if not eval_mode:
                data_list = self.diffusion_buffer.sample_batch(self.batch_size)
                prev_obs, action_venv, target_action_venv, reward_venv, next_obs, done_venvv = data_list

                # 1. Representation learning update
                self.rep_optimizer.zero_grad()
                simi_loss = self.rep_func(prev_obs, action_venv, next_obs)
                simi_loss.backward()
                self.rep_optimizer.step()

                # 2. Intrinsic reward（encoder 不参与梯度）
                with torch.no_grad():
                    prev_obs1, _ = self.model.encoder(
                        prev_obs['neighbor_trajs'],
                        mask=None,
                        test=prev_obs,
                        init_state=prev_obs['ego_state'],
                        map_state=prev_obs['neighbor_waypoints']
                    )
                    next_obs1, _ = self.model.target_encoder(
                        next_obs['neighbor_trajs'],
                        mask=None,
                        test=next_obs,
                        init_state=next_obs['ego_state'],
                        map_state=next_obs['neighbor_waypoints']
                    )
                reward_intrinsic = self.intrinsic.compute_reward(prev_obs1, next_obs1)
                reward_venv = reward_venv + reward_intrinsic




                with torch.no_grad():

                    obs_trajs["neighbor_trajs"] = torch.from_numpy(obs_trajs["neighbor_trajs"]).float().to(self.device)
                    obs_trajs["ego_state"] = torch.from_numpy(obs_trajs["ego_state"]).float().to(self.device)
                    obs_trajs["neighbor_waypoints"] = torch.from_numpy(obs_trajs["neighbor_waypoints"]).float().to(
                        self.device)

                    # (n_steps, n_envs, n_cond_step, obs_dim) for furniture

                    # Calculate value and logprobs - split into batches to prevent out of memory
                    # num_split = math.ceil(
                    #     self.n_envs * self.n_steps / self.logprob_batch_size
                    # )
                    flat_neighbor_trajs = einops.rearrange(
                        obs_trajs["neighbor_trajs"], "s e ... -> (s e) ..."
                    )
                    flat_ego_state = einops.rearrange(
                        obs_trajs["ego_state"], "s e ... -> (s e) ..."
                    )
                    flat_neighbor_wps = einops.rearrange(
                        obs_trajs["neighbor_waypoints"], "s e ... -> (s e) ..."
                    )

                    # (n_steps*n_envs, n_cond_step, obs_dim) for furniture

                    split_neighbor_trajs = torch.split(flat_neighbor_trajs, self.logprob_batch_size, dim=0)
                    split_ego_state = torch.split(flat_ego_state, self.logprob_batch_size, dim=0)
                    split_neighbor_wps = torch.split(flat_neighbor_wps, self.logprob_batch_size, dim=0)
                    obs_ts_k = []
                    for nt, es, nw in zip(split_neighbor_trajs, split_ego_state, split_neighbor_wps):
                        obs_ts_k.append({
                            "neighbor_trajs": nt,
                            "ego_state": es,
                            "neighbor_waypoints": nw,
                        })

                    obs_k_encoded_list = []
                    values_trajs = np.empty((0, self.n_envs))
                    for obs in obs_ts_k:
                        obs, _ = self.model.encoder(
                            obs['neighbor_trajs'],
                            mask=None,
                            test=obs,
                            init_state=obs['ego_state'],
                            map_state=obs['neighbor_waypoints']
                        )
                        obs_k_encoded_list.append(obs)
                        values = self.model.critic.get_v(obs).cpu().numpy().flatten()
                        values_trajs = np.vstack(
                            (values_trajs, values.reshape(-1, self.n_envs))
                        )
                    obs_k_encoded = torch.cat(obs_k_encoded_list, dim=0)  # (n_steps*n_envs, obs_dim)

                    # (n_steps, n_envs) for furniture

                    chains_t = einops.rearrange(
                        torch.from_numpy(chains_trajs).float().to(self.device),
                        "s e t h d -> (s e) t h d",
                    )
                    chains_ts = torch.split(chains_t, self.logprob_batch_size, dim=0)
                    logprobs_trajs = np.empty(
                        (
                            0,
                            self.model.ft_denoising_steps,
                            self.horizon_steps,
                            self.action_dim,
                        )
                    )
                    # logprobs_trajs (0, 10, 4, 3)
                    for obs, chains in zip(obs_ts_k, chains_ts):
                        logprobs = self.model.get_logprobs(obs, chains).cpu().numpy()
                        # logprobs (100000, 4, 3)
                        #是在 PPO 训练阶段，把 chains 的每一对前后 denoising 步骤 (x_t, x_{t+1}) 作为“真实动作”，
                        # 当前策略再去预测其 mean/std，看是否一致 —— 相当于 “log-prob 的逆推”。
                        logprobs_trajs = np.vstack(
                            (
                                logprobs_trajs,
                                logprobs.reshape(-1, *logprobs_trajs.shape[1:]),
                            )
                        ) # (10000, 10, 4, 3)
                    # logprobs_trajs  (n_steps*n_envs, denoising_steps, horizon_steps, action_dim) for furniture

                    # normalize reward with running variance if specified
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_transpose.T
                    # 反向折扣累计 类似 V_value

                    # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified


                    obs_venv_ts = {
                        "neighbor_trajs": torch.from_numpy(obs_venv["neighbor_trajs"]).float().to(
                            self.device),
                        "ego_state": torch.from_numpy(obs_venv["ego_state"]).float().to(self.device),
                        "neighbor_waypoints": torch.from_numpy(obs_venv["neighbor_waypoints"]).float().to(
                            self.device),
                    }
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    with torch.no_grad():
                        obs_venv_ts, _ = self.model.encoder(
                            obs_venv_ts['neighbor_trajs'],
                            mask=None,
                            test=None,
                            init_state=obs_venv_ts['ego_state'],
                            map_state=obs_venv_ts['neighbor_waypoints']
                        )

                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextvalues = (
                                self.model.critic.get_v(obs_venv_ts)
                                .reshape(1, -1)
                                .cpu()
                                .numpy()
                            )
                        else:
                            nextvalues = values_trajs[t + 1]
                        nonterminal = 1.0 - terminated_trajs[t]
                        # delta = r + gamma*V(st+1) - V(st)
                        delta = (
                            reward_trajs[t] * self.reward_scale_const
                            + self.gamma * nextvalues * nonterminal
                            - values_trajs[t]
                        )
                        # A = delta_t + gamma*lamdba*delta_{t+1} + ...
                        advantages_trajs[t] = lastgaelam = (
                            delta
                            + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                        )
                    returns_trajs = advantages_trajs + values_trajs
                    # (n_steps, n_envs) for furniture
                    # k for environment step
                obs_k = {
                    "neighbor_trajs": einops.rearrange(
                        obs_trajs["neighbor_trajs"],
                        "s e ... -> (s e) ...",
                    ),
                    "ego_state": einops.rearrange(
                        obs_trajs["ego_state"],
                        "s e ... -> (s e) ...",
                    ),
                    "neighbor_waypoints": einops.rearrange(
                        obs_trajs["neighbor_waypoints"],
                        "s e ... -> (s e) ...",
                    ),
                }

                # (n_steps*n_envs, n_cond_step, obs_dim) for furniture

                chains_k = einops.rearrange(
                    torch.tensor(chains_trajs, device=self.device).float(),
                    "s e t h d -> (s e) t h d",
                )
                # (n_steps*n_envs, denoising_steps + 1, horizon_steps, action_dim) for furniture

                returns_k = (
                    torch.tensor(returns_trajs, device=self.device).float().reshape(-1)
                )
                # (n_steps*n_envs) for furniture

                values_k = (
                    torch.tensor(values_trajs, device=self.device).float().reshape(-1)
                )
                # (n_steps*n_envs) for furniture

                advantages_k = (
                    torch.tensor(advantages_trajs, device=self.device)
                    .float()
                    .reshape(-1)
                )
                # (n_steps*n_envs) for furniture

                logprobs_k = torch.tensor(logprobs_trajs, device=self.device).float()
                # logprobs_trajs  (n_steps*n_envs, denoising_steps, horizon_steps, action_dim) for furniture

                # Update policy and critic
                total_steps = self.n_steps * self.n_envs * self.model.ft_denoising_steps
                clipfracs = []
                flag_break = False
                for update_epoch in range(self.update_epochs):
                    # for each epoch, go through all data in batches
                    inds_k = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]  # b for batch
                        batch_inds_b, denoising_inds_b = torch.unravel_index(
                            inds_b,
                            (self.n_steps * self.n_envs, self.model.ft_denoising_steps),
                        )
                        # torch.Size([50000])
                        # torch.Size([50000])
                        # torch.Size([50000])

                        # Use pre-encoded tensor to skip 780 redundant encoder forward passes
                        obs_b = obs_k_encoded[batch_inds_b]

                        chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]
                        # (batch,  horizon_steps, action_dim) for furniture


                        chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]
                        returns_b = returns_k[batch_inds_b]
                        values_b = values_k[batch_inds_b]
                        advantages_b = advantages_k[batch_inds_b]
                        logprobs_b = logprobs_k[batch_inds_b, denoising_inds_b]

                        # get loss
                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            eta,
                        ) = self.model.loss(
                            obs_b,
                            chains_prev_b,
                            chains_next_b,
                            denoising_inds_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                            reward_horizon=self.reward_horizon,
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )

                        clipfracs += [clipfrac]

                        # update policy and critic
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        if self.learn_eta:
                            self.eta_optimizer.zero_grad()
                        loss.backward()
                        if self.itr >= self.n_critic_warmup_itr:
                            if self.max_grad_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.actor_ft.parameters(), self.max_grad_norm
                                )
                            self.actor_optimizer.step()
                            if self.learn_eta and batch % self.eta_update_interval == 0:
                                self.eta_optimizer.step()
                        self.critic_optimizer.step()

                        # log.info(
                        #     f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}"
                        # )

                        # Stop gradient update if KL difference reaches target
                        if self.target_kl is not None and approx_kl > self.target_kl:
                            flag_break = True
                            break
                    if flag_break:
                        break

                # Off-policy 更新在 PPO 之后执行，保证 trust region 完整
                # Clone original actions before Q-improvement: update_target_action modifies action
                # in-place via Adam, so critic must receive the original environment actions.
                action_orig = action_venv.clone()

                # 3. Update target action（使用已收敛的 Q 网络）
                _, new_action = self.model.update_target_action(prev_obs, action_venv)
                self.diffusion_buffer.update_target_action(new_action)

                # 4. Actor BC off-policy 独立 backward
                self.actor_optimizer.zero_grad()
                bcloss = self.model.update_actor(prev_obs, new_action)
                bcloss.backward()  #11111
                if self.itr >= self.n_critic_warmup_itr:
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.actor_ft.parameters(), self.max_grad_norm
                        )
                    self.actor_optimizer.step()
                bcloss = bcloss.item()

                # 5. Q-network critic off-policy 独立 backward（encoder 冻结）
                with torch.no_grad():
                    prev_obs2, _ = self.model.encoder(
                        prev_obs['neighbor_trajs'],
                        mask=None,
                        test=prev_obs,
                        init_state=prev_obs['ego_state'],
                        map_state=prev_obs['neighbor_waypoints']
                    )
                    next_obs2, _ = self.model.target_encoder(
                        next_obs['neighbor_trajs'],
                        mask=None,
                        test=next_obs,
                        init_state=next_obs['ego_state'],
                        map_state=next_obs['neighbor_waypoints']
                    )
                critic_loss = 0.0
                for _ in range(self.critic_q_update_steps):
                    q_batch = self.diffusion_buffer.sample_batch(self.batch_size)
                    qb_obs, qb_action, _, qb_reward, qb_next_obs, qb_done = q_batch
                    with torch.no_grad():
                        qb_obs_enc, _ = self.model.encoder(
                            qb_obs['neighbor_trajs'], mask=None, test=qb_obs,
                            init_state=qb_obs['ego_state'], map_state=qb_obs['neighbor_waypoints']
                        )
                        qb_next_enc, _ = self.model.target_encoder(
                            qb_next_obs['neighbor_trajs'], mask=None, test=qb_next_obs,
                            init_state=qb_next_obs['ego_state'], map_state=qb_next_obs['neighbor_waypoints']
                        )
                        # Buffer stores raw scaled reward; add intrinsic bonus consistent with on-policy collection
                        qb_reward = qb_reward + self.intrinsic.compute_reward(qb_obs_enc, qb_next_enc)
                    self.critic_q_optimizer.zero_grad()
                    _closs = self.model.update_critic(qb_obs_enc, qb_action, qb_reward, qb_next_enc, qb_done)
                    _closs.backward()  #11111
                    if self.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.critic.net_q1.parameters()) +
                            list(self.model.critic.net_q2.parameters()),
                            self.max_grad_norm,
                        )
                    self.critic_q_optimizer.step()
                    # Standard: soft-update target network once per gradient step
                    soft_update(self.model.critic_target, self.model.critic, self.tau)
                    critic_loss += _closs.item()
                critic_loss /= self.critic_q_update_steps
                self.rep_func._update_params(self.tau)

                with torch.no_grad():
                    prev_obs3, _ = self.model.encoder(
                        prev_obs['neighbor_trajs'],
                        mask=None,
                        test=prev_obs,
                        init_state=prev_obs['ego_state'],
                        map_state=prev_obs['neighbor_waypoints']
                    )
                    next_obs3, _ = self.model.target_encoder(
                        next_obs['neighbor_trajs'],
                        mask=None,
                        test=next_obs,
                        init_state=next_obs['ego_state'],
                        map_state=next_obs['neighbor_waypoints']
                    )
                if self.intrinsic.type == 'rnd':
                    dynamic_loss, dynamic_grad_norm = self.intrinsic.update(prev_obs3)
                elif self.intrinsic.type == 'noveld':
                    dynamic_loss, dynamic_grad_norm = self.intrinsic.update(torch.cat([prev_obs3, next_obs3]))
                else:
                    raise NotImplementedError
                # Explained variation of future rewards using value function
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

            # Plot state trajectories (only in D3IL)

            # Update lr, min_sampling_std
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            # 计算多窗口成功率及标准差
            n_ep = len(success_log)
            succ_20      = np.mean(success_log[-20:])  if n_ep >= 1 else 0.0
            succ_20_std  = np.std(success_log[-20:])   if n_ep >= 1 else 0.0
            succ_100     = np.mean(success_log[-100:]) if n_ep >= 1 else 0.0
            succ_100_std = np.std(success_log[-100:])  if n_ep >= 1 else 0.0
            succ_all     = np.mean(success_log)        if n_ep >= 1 else 0.0
            succ_all_std = np.std(success_log)         if n_ep >= 1 else 0.0
            success      = succ_100  # 兼容旧的 wandb key

            if self.save_trajs:
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["chains_trajs"] = chains_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                prog = f"[{self.itr:4d}/{self.n_train_itr} | ep:{cnt_episode:4d}]"
                if eval_mode:
                    log.info(
                        f"{'='*60}\n"
                        f"  EVAL {prog}\n"
                        f"  success(iter)={success_rate*100:.1f}%±{std_success_rate*100:.1f}% | "
                        f"succ(20ep)={succ_20*100:.1f}%±{succ_20_std*100:.1f}% | "
                        f"succ(100ep)={succ_100*100:.1f}%±{succ_100_std*100:.1f}% | "
                        f"succ(all)={succ_all*100:.1f}%±{succ_all_std*100:.1f}%\n"
                        f"  reward={avg_episode_reward*100:.2f}±{std_episode_reward*100:.2f} | "
                        f"best={avg_best_reward*100:.2f}±{std_best_reward*100:.2f} | "
                        f"steps={avg_episode_steps:.1f}±{std_episode_steps:.1f} | "
                        f"episodes={num_episode_finished}\n"
                        f"{'='*60}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate% - eval": success_rate * 100,
                                "std success rate% - eval": std_success_rate * 100,
                                "avg episode reward(x100) - eval": avg_episode_reward * 100,
                                "std episode reward(x100) - eval": std_episode_reward * 100,
                                "avg best reward(x100) - eval": avg_best_reward * 100,
                                "std best reward(x100) - eval": std_best_reward * 100,
                                "num episode - eval": num_episode_finished,
                                "avg episode steps - eval": avg_episode_steps,
                                "std episode steps - eval": std_episode_steps,
                                "success rate% 20ep": succ_20 * 100,
                                "std success rate% 20ep": succ_20_std * 100,
                                "success rate% 100ep": succ_100 * 100,
                                "std success rate% 100ep": succ_100_std * 100,
                                "success rate% all": succ_all * 100,
                                "std success rate% all": succ_all_std * 100,
                            },
                            step=self.itr,
                            commit=False,
                        )
                    run_results[-1]["eval_success_rate_pct"] = success_rate * 100
                    run_results[-1]["eval_std_success_rate_pct"] = std_success_rate * 100
                    run_results[-1]["eval_episode_reward_x100"] = avg_episode_reward * 100
                    run_results[-1]["eval_std_episode_reward_x100"] = std_episode_reward * 100
                    run_results[-1]["eval_best_reward_x100"] = avg_best_reward * 100
                    run_results[-1]["eval_avg_episode_steps"] = avg_episode_steps
                    run_results[-1]["eval_std_episode_steps"] = std_episode_steps
                else:
                    log.info(
                        f"TRAIN {prog} step={cnt_train_step:7d} | "
                        f"succ(20ep)={succ_20*100:.1f}%±{succ_20_std*100:.1f}% | "
                        f"succ(100ep)={succ_100*100:.1f}%±{succ_100_std*100:.1f}% | "
                        f"succ(all)={succ_all*100:.1f}%±{succ_all_std*100:.1f}% | "
                        f"reward={avg_episode_reward*100:.2f}±{std_episode_reward*100:.2f} | "
                        f"steps={avg_episode_steps:.1f}±{std_episode_steps:.1f} | "
                        f"loss={loss:.4f} | pg={pg_loss:.4f} | "
                        f"critic_v={v_loss:.4f} | critic_q={critic_loss:.4f} | bc={bc_loss:.4f} | "
                        f"eta={eta:.4f} | t={time:.1f}s"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "loss": loss,
                                "pg loss": pg_loss,
                                # critic/v_loss: PPO state-value network (net_v), trained with MSE on GAE returns
                                "critic/v_loss": v_loss,
                                # critic/q_loss: distributional Q network (net_q1/q2), trained with C51 cross-entropy
                                "critic/q_loss": critic_loss,
                                "bc loss": bc_loss,
                                "new_bc_loss": bcloss,
                                "dynamic_loss": dynamic_loss,
                                "eta": eta,
                                "approx kl": approx_kl,
                                "ratio": ratio,
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward(x100) - train": avg_episode_reward * 100,
                                "std episode reward(x100) - train": std_episode_reward * 100,
                                "avg episode steps - train": avg_episode_steps,
                                "std episode steps - train": std_episode_steps,
                                "num episode - train": num_episode_finished,
                                "diffusion - min sampling std": diffusion_min_sampling_std,
                                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                                "critic lr": self.critic_optimizer.param_groups[0]["lr"],
                                "simi_loss": simi_loss,
                                "success rate% - train": success * 100,
                                "success rate% 20ep": succ_20 * 100,
                                "std success rate% 20ep": succ_20_std * 100,
                                "success rate% 100ep": succ_100 * 100,
                                "std success rate% 100ep": succ_100_std * 100,
                                "success rate% all": succ_all * 100,
                                "std success rate% all": succ_all_std * 100,
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward_x100"] = avg_episode_reward * 100
                    run_results[-1]["train_std_episode_reward_x100"] = std_episode_reward * 100
                    run_results[-1]["train_avg_episode_steps"] = avg_episode_steps
                    run_results[-1]["train_std_episode_steps"] = std_episode_steps
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
