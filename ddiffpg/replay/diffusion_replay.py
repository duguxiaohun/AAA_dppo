import torch
import numpy as np
import random
from copy import deepcopy
from ddiffpg.replay.simple_replay import create_buffer, DiffusionReplayBuffer
from ddiffpg.utils.Q_scheduler import Q_scheduler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from dtaidistance import dtw_ndim
from collections import deque


class DiffusionGoalBuffer:
    def __init__(self, cfg, capacity: int, obs_dim: int, action_dim: int, num_envs: int, max_episode_len=1000, device='cpu', cond_steps=1, horizon_steps=4):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.env_num = num_envs
        self.max_episode_len = max_episode_len
        self.capacity = capacity
        self.cond_steps = cond_steps
        self.horizon_steps = horizon_steps
        

        

        self.lengths = deque(maxlen=self.capacity)
        self.count = 0
        self.map = {}
        self.plot = None

        # temp trajectory storage
        ret = create_buffer(capacity=(self.max_episode_len, self.env_num), obs_dim=obs_dim, action_dim=action_dim, device=device)
        self.traj_state, self.traj_action, self.traj_next_state, self.traj_reward, self.traj_done = ret
        self.traj_target_action = torch.empty_like(self.traj_action)
        self.replay_buffer = DiffusionReplayBuffer(capacity=capacity,
                              obs_dim=obs_dim,
                              action_dim=action_dim,
                              device=device,
                              cond_steps=self.cond_steps,
                              horizon_steps=self.horizon_steps)




