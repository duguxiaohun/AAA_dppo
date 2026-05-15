"""
DPPO offline-train.

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
from ddiffpg.utils.torch_util import soft_update
from model.diffusion.policy_v1 import Represent_Learner
import torch.optim as optim
from script.load_carla_dataset_new import HDF5BatchImporter

class TrainPPODiffusionAgent(TrainPPOAgent):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Reward horizon --- always set to act_steps for now
        self.reward_horizon = cfg.get("reward_horizon", self.act_steps)

        self.filter_success_only = True

        self.diffusion_buffer = DiffusionReplayBuffer(capacity=self.cfg.memory_size,
                              obs_dim=self.obs_dim,
                              action_dim=self.action_dim,
                              device=self.device,
                              cond_steps=self.cond_steps,
                              horizon_steps=self.horizon_steps)

        self.importer = HDF5BatchImporter(self.diffusion_buffer)
        if self.filter_success_only == True:
            self.diffusion_buffer1 = DiffusionReplayBuffer(capacity=self.cfg.memory_size,
                              obs_dim=self.obs_dim,
                              action_dim=self.action_dim,
                              device=self.device,
                              cond_steps=self.cond_steps,
                              horizon_steps=self.horizon_steps)
            self.importer1 = HDF5BatchImporter(self.diffusion_buffer1)



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

        return cond

    def run(self):
        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        success_log = [0]  # 成功记录
        self.importer.import_multiple_files(
            n_files=15,
            filter_success_only=False  # 是否只导入成功轨迹
        )
        if self.filter_success_only:
            self.importer1.import_multiple_files(
                n_files=25,
                filter_success_only=True  # 是否只导入成功轨迹
            )
        self.itr = 0

        while self.itr < self.n_train_itr:



            self.model.train()
            for update_epoch in range(self.update_epochs):

                data_list = self.diffusion_buffer.sample_batch(self.batch_size)
                prev_obs, action_venv, target_action_venv, reward_venv, next_obs, done_venvv = data_list
                self.rep_optimizer.zero_grad()


                simi_loss = self.rep_func(prev_obs, action_venv, next_obs)
                simi_loss.backward()
                self.rep_optimizer.step()



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




                # mean_action, new_action = self.model.update_target_action(prev_obs, action_venv)
                # self.diffusion_buffer.update_target_action(new_action)
                bcloss = self.model.update_actor(prev_obs, action_venv)


                critic_loss = self.model.update_critic(prev_obs1, action_venv, reward_venv, next_obs1, done_venvv)

                batch_size = len(action_venv)
                t = torch.randint(
                    0, self.model.denoising_steps, (batch_size,), device=action_venv.device
                ).long()
                if self.filter_success_only:
                    data_list1 = self.diffusion_buffer1.sample_batch(self.batch_size)
                    prev_obs3, action_venv1, _, _, _, _ = data_list1
                    prev_obs3, _ = self.model.encoder(
                        prev_obs3['neighbor_trajs'],
                        mask=None,
                        test=prev_obs3,
                        init_state=prev_obs3['ego_state'],
                        map_state=prev_obs3['neighbor_waypoints']
                    )

                    bc_loss = self.model.p_losses(action_venv1, prev_obs3, t)

                else:
                    bc_loss = self.model.p_losses(action_venv, prev_obs1, t)

                loss = (
                        + bc_loss * self.bc_loss_coeff  # 两个策略相似性 # 0
                        + critic_loss * self.vf_coef
                        + bcloss * self.bc_loss_coeff  # 去噪loss
                )




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
                    if self.learn_eta:
                        self.eta_optimizer.step()
                self.critic_optimizer.step()





            soft_update(self.model.critic_target, self.model.critic, self.tau)
            self.rep_func._update_params(self.tau)


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
            success = np.sum(success_log[-20:]) / 20

            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time


                if self.use_wandb:
                    wandb.log(
                        {
                            "total env step": cnt_train_step,
                            "loss": loss,
                            "bc loss": bc_loss,
                            "new_bc_loss": bcloss,
                            "new_critic_loss": critic_loss,

                            "diffusion - min sampling std": diffusion_min_sampling_std,
                            "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                            "critic lr": self.critic_optimizer.param_groups[0][
                                "lr"
                            ],
                            "simi_loss": simi_loss,
                            "success_rate": success
                        },
                        step=self.itr,
                        commit=True,
                    )
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
