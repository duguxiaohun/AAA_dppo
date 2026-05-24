"""
Policy gradient with diffusion policy. VPG: vanilla policy gradient

K: number of denoising steps
To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""
from copy import deepcopy

import copy
import torch
import logging
log = logging.getLogger(__name__)

from model.diffusion.diffusion import DiffusionModel, Sample
from model.diffusion.sampling import make_timesteps, extract
from torch.distributions import Normal
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ddiffpg.algo.ac_base import ActorCriticBase
from ddiffpg.replay.nstep_replay import NStepReplay
from ddiffpg.utils.noise import add_mixed_normal_noise
from ddiffpg.utils.noise import add_normal_noise
from ddiffpg.utils.schedule_util import ExponentialSchedule
from ddiffpg.utils.schedule_util import LinearSchedule
from ddiffpg.utils.torch_util import soft_update
from ddiffpg.utils.common import handle_timeout, DensityTracker
from ddiffpg.utils.distl_util import projection
from ddiffpg.utils.torch_util import add_embedding
from ddiffpg.utils.intrinsic import IntrinsicM
from model.diffusion.policy_v1 import RLEncoder


from configs.init_configs import set_configs, get_argument
import gym

parser = get_argument()
# 兼容 Hydra 的 key=value 覆盖参数，避免在导入阶段因未知参数直接退出。
args, _unknown = parser.parse_known_args()

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    print(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
else:
    print("No GPU found, running on CPU")

args.scenario = 'carla'
args.algo = 'scene_rep'  # 确保 scene_rep 会使用 PyTorch 版本的 SAC
args, params_cfg, runner_params = set_configs(args, test=False)
params_cfg = params_cfg['params']
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,))
OBSERVATION_SPACE = gym.spaces.Box(low=-1000, high=1000, shape=(args.neighbors + 1, args.n_steps, args.dim,))

class VPGDiffusion(DiffusionModel):

    def __init__(
        self,
        actor,
        critic,
        ft_denoising_steps,
        ft_denoising_steps_d=0,
        ft_denoising_steps_t=0,
        network_path=None,
        # modifying denoising schedule
        min_sampling_denoising_std=0.1,
        min_logprob_denoising_std=0.1,
        # eta in DDIM
        eta=None,
        learn_eta=False,
        noise=None,
        params=None,
        **kwargs,
    ):
        super().__init__(
            network=actor,
            network_path=network_path,
            **kwargs,
        )
        assert ft_denoising_steps <= self.denoising_steps
        assert ft_denoising_steps <= self.ddim_steps if self.use_ddim else True
        assert not (learn_eta and not self.use_ddim), "Cannot learn eta with DDPM."


        # Number of denoising steps to use with fine-tuned model. Thus denoising_step - ft_denoising_steps is the number of denoising steps to use with original model.
        self.ft_denoising_steps = ft_denoising_steps
        self.ft_denoising_steps_d = ft_denoising_steps_d  # annealing step size
        self.ft_denoising_steps_t = ft_denoising_steps_t  # annealing interval
        self.ft_denoising_steps_cnt = 0

        # Minimum std used in denoising process when sampling action - helps exploration
        self.min_sampling_denoising_std = min_sampling_denoising_std

        # Minimum std used in calculating denoising logprobs - for stability
        self.min_logprob_denoising_std = min_logprob_denoising_std

        # Learnable eta
        self.learn_eta = learn_eta
        if eta is not None:
            self.eta = eta.to(self.device)
            if not learn_eta:
                for param in self.eta.parameters():
                    param.requires_grad = False
                logging.info("Turned off gradients for eta")

        # Re-name network to actor
        self.actor = self.network

        # Make a copy of the original model
        self.actor_ft = copy.deepcopy(self.actor)
        logging.info("Cloned model for fine-tuning")

        # Turn off gradients for original model
        for param in self.actor.parameters():
            param.requires_grad = False
        logging.info("Turned off gradients of the pretrained network")
        logging.info(
            f"Number of finetuned parameters: {sum(p.numel() for p in self.actor_ft.parameters() if p.requires_grad)}"
        )

        # Value function
        self.critic = critic.to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        if network_path is not None:
            checkpoint = torch.load(
                network_path, map_location=self.device, weights_only=True
            )
            if "ema" not in checkpoint:  # load trained RL model
                self.load_state_dict(checkpoint["model"], strict=False)
                logging.info("Loaded critic from %s", network_path)
        if noise is not None:
            if noise.decay == 'linear':
                self.noise_scheduler = LinearSchedule(start_val=self.cfg.algo.noise.std_max,
                                                      end_val=self.cfg.algo.noise.std_min,
                                                      total_iters=self.cfg.algo.noise.lin_decay_iters
                                                      )
            elif noise.decay == 'exp':
                self.noise_scheduler = ExponentialSchedule(start_val=self.cfg.algo.noise.std_max,
                                                           gamma=self.cfg.algo.exp_decay_rate,
                                                           end_val=self.cfg.algo.noise.std_min)
            else:
                self.noise_scheduler = None



        self.reward_mean = deque(maxlen=int(1e4))

        self.encoder = RLEncoder(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.shape, units=[256] * 3,
                                 trans=False,
                                 cnn_lstm=params_cfg['cnn_lstm'], ego_surr=params_cfg['ego_surr'],
                                 use_trans=params_cfg['use_trans'], neighbours=params_cfg['neighbours'],
                                 time_step=params_cfg['time_step'], debug=False,
                                 make_rotation=params_cfg['make_rotation'], make_prediction=params_cfg['make_prediction'],
                                 use_map=params_cfg['use_map'],
                                 num_traj=params_cfg['traj_nums'], path_length=params_cfg['path_length'], head_dim=params_cfg['head_num'],
                                 cnn=params_cfg['cnn'],
                                 use_hier=params_cfg['use_hier'], random_aug=params_cfg['random_aug'], carla=params_cfg['carla'],
                                 no_ego_fut=params_cfg['no_ego_fut'], no_neighbor_fut=params_cfg['no_neighbor_fut']).to(self.device)
        self.target_encoder = RLEncoder(state_shape=OBSERVATION_SPACE.shape, action_dim=ACTION_SPACE.shape, units=[256] * 3,
                                 trans=False,
                                 cnn_lstm=params_cfg['cnn_lstm'], ego_surr=params_cfg['ego_surr'],
                                 use_trans=params_cfg['use_trans'], neighbours=params_cfg['neighbours'],
                                 time_step=params_cfg['time_step'], debug=False,
                                 make_rotation=params_cfg['make_rotation'],
                                 make_prediction=params_cfg['make_prediction'],
                                 use_map=params_cfg['use_map'],
                                 num_traj=params_cfg['traj_nums'], path_length=params_cfg['path_length'],
                                 head_dim=params_cfg['head_num'],
                                 cnn=params_cfg['cnn'],
                                 use_hier=params_cfg['use_hier'], random_aug=params_cfg['random_aug'],
                                 carla=params_cfg['carla'],
                                 no_ego_fut=params_cfg['no_ego_fut'], no_neighbor_fut=params_cfg['no_neighbor_fut']).to(
            self.device)

    # ---------- Sampling ----------#

    def step(self):
        """
        Anneal min_sampling_denoising_std and fine-tuning denoising steps

        Current configs do not apply annealing
        """
        # anneal min_sampling_denoising_std
        if type(self.min_sampling_denoising_std) is not float:
            self.min_sampling_denoising_std.step()

        # anneal denoising steps
        self.ft_denoising_steps_cnt += 1
        if (
            self.ft_denoising_steps_d > 0
            and self.ft_denoising_steps_t > 0
            and self.ft_denoising_steps_cnt % self.ft_denoising_steps_t == 0
        ):
            self.ft_denoising_steps = max(
                0, self.ft_denoising_steps - self.ft_denoising_steps_d
            )

            # update actor
            self.actor = self.actor_ft
            self.actor_ft = copy.deepcopy(self.actor)
            for param in self.actor.parameters():
                param.requires_grad = False
            logging.info(
                f"Finished annealing fine-tuning denoising steps to {self.ft_denoising_steps}"
            )

    def get_min_sampling_denoising_std(self):
        if type(self.min_sampling_denoising_std) is float:
            return self.min_sampling_denoising_std
        else:
            return self.min_sampling_denoising_std()



    def get_tgt_policy_actions(self, obs, deterministic=False, return_chain=False):
        samples = self.forward(
            cond=obs,
            deterministic=deterministic,
            return_chain=return_chain,
            use_base_policy=True,
        )

        return samples


    def bc_loss(self,
                x_start,
                cond,
                index=None,
                use_base_policy=False,
                deterministic=False,
                ):
        batch_size = len(x_start)
        t = torch.randint(
            0, self.denoising_steps, (batch_size,), device=x_start.device
        ).long()
        device = x_start.device
        noise_random = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise_random)
        noise = self.actor(x_noisy, t, cond=cond)
        if self.use_ddim:
            ft_indices = torch.where(
                index >= (self.ddim_steps - self.ft_denoising_steps)
            )[0]
        else:
            ft_indices = torch.where(t < self.ft_denoising_steps)[0]

        # Use base policy to query expert model, e.g. for imitation loss
        actor = self.actor if use_base_policy else self.actor_ft
        # overwrite noise for fine-tuning steps
        if len(ft_indices) > 0:
            # cond_ft = {key: cond[key][ft_indices] for key in cond}
            cond_ft = cond[ft_indices]
            noise_ft = actor(x_noisy[ft_indices], t[ft_indices], cond=cond_ft)
            noise[ft_indices] = noise_ft

        return F.mse_loss(noise_random, noise, reduction="mean")

    # override
    # 单步预测
    def p_mean_var(
        self,
        x,
        t,
        cond,
        index=None,
        use_base_policy=False,
        deterministic=False,
    ):

        noise = self.actor(x, t, cond=cond)
        if self.use_ddim:
            ft_indices = torch.where(
                index >= (self.ddim_steps - self.ft_denoising_steps)
            )[0]
        else:
            ft_indices = torch.where(t < self.ft_denoising_steps)[0]

        # Use base policy to query expert model, e.g. for imitation loss
        actor = self.actor if use_base_policy else self.actor_ft
        # overwrite noise for fine-tuning steps
        if len(ft_indices) > 0:
            # cond_ft = {key: cond[key][ft_indices] for key in cond}
            cond_ft = cond[ft_indices]
            noise_ft = actor(x[ft_indices], t[ft_indices], cond=cond_ft)
            noise[ft_indices] = noise_ft

        # Predict x_0
        if self.predict_epsilon:
            if self.use_ddim:
                """
                x₀ = (xₜ - √ (1-αₜ) ε )/ √ αₜ
                """
                alpha = extract(self.ddim_alphas, index, x.shape)
                alpha_prev = extract(self.ddim_alphas_prev, index, x.shape)
                sqrt_one_minus_alpha = extract(
                    self.ddim_sqrt_one_minus_alphas, index, x.shape
                )
                x_recon = (x - sqrt_one_minus_alpha * noise) / (alpha**0.5)
            else:
                """
                x₀ = √ 1\α̅ₜ xₜ - √ 1\α̅ₜ-1 ε
                """
                x_recon = (
                    extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                    - extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * noise
                )
        else:  # directly predicting x₀
            x_recon = noise
        if self.denoised_clip_value is not None:
            x_recon.clamp_(-self.denoised_clip_value, self.denoised_clip_value)
            if self.use_ddim:
                # re-calculate noise based on clamped x_recon - default to false in HF, but let's use it here
                noise = (x - alpha ** (0.5) * x_recon) / sqrt_one_minus_alpha

        # Clip epsilon for numerical stability in policy gradient - not sure if this is helpful yet, but the value can be huge sometimes. This has no effect if DDPM is used
        if self.use_ddim and self.eps_clip_value is not None:
            noise.clamp_(-self.eps_clip_value, self.eps_clip_value)

        # Get mu
        if self.use_ddim:
            """
            μ = √ αₜ₋₁ x₀ + √(1-αₜ₋₁ - σₜ²) ε
            """
            if deterministic:
                etas = torch.zeros((x.shape[0], 1, 1)).to(x.device)
            else:
                etas = self.eta(cond).unsqueeze(1)  # B x 1 x (Da or 1)
            sigma = (
                etas
                * ((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)) ** 0.5
            ).clamp_(min=1e-10)
            dir_xt_coef = (1.0 - alpha_prev - sigma**2).clamp_(min=0).sqrt()
            mu = (alpha_prev**0.5) * x_recon + dir_xt_coef * noise
            var = sigma**2
            logvar = torch.log(var)
        else:
            """
            μₜ = β̃ₜ √ α̅ₜ₋₁/(1-α̅ₜ)x₀ + √ αₜ (1-α̅ₜ₋₁)/(1-α̅ₜ)xₜ
            """
            mu = (
                extract(self.ddpm_mu_coef1, t, x.shape) * x_recon
                + extract(self.ddpm_mu_coef2, t, x.shape) * x
            )
            logvar = extract(self.ddpm_logvar_clipped, t, x.shape)
            etas = torch.ones_like(mu).to(mu.device)  # always one for DDPM
        return mu, logvar, etas

    # override
    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        return_chain=True,
        use_base_policy=False,
        mask=None,
        test=False,
        start_x=None,
        use_target_critic_for_improvement=False,
    ):
        """
        Forward pass for sampling actions.

        Args:
            eval: mode
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            deterministic: If true, then std=0 with DDIM, or with DDPM, use normal schedule (instead of clipping at a higher value)
            return_chain: whether to return the entire chain of denoised actions
            use_base_policy: whether to use the frozen pre-trained policy instead
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
                chain: (B, K + 1, Ta, Da)
        """
        device = self.betas.device
        if isinstance(cond, dict):
            cond = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cond.items()}
        elif torch.is_tensor(cond):
            cond = cond.to(device)

        if mask is not None and torch.is_tensor(mask):
            mask = mask.to(device)

        if isinstance(cond, dict):
            cond, _ = self.encoder(
                cond['neighbor_trajs'],
                mask=mask,
                test=test,
                init_state=cond['ego_state'],
                map_state=cond['neighbor_waypoints']
            )

        B = cond.shape[0]

        # torch.Size([1, 1, 128])

        # Get updated minimum sampling denoising std
        min_sampling_denoising_std = self.get_min_sampling_denoising_std()

        # Loop
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        x = torch.clamp(x, -1.0, 1.0)
        if start_x is None:
            start_x = self.denoising_steps
        elif not isinstance(start_x, int):
            # Convert float uncertainty [0,1] to integer skip count.
            # uncertainty=1 (high, state very different) → skip 0 steps (full denoising)
            # uncertainty=0 (low, state similar) → skip denoising_steps-1 steps (1 step)
            if hasattr(start_x, 'item'):
                start_x = start_x.item()
            u = float(start_x)
            start_x = int((1.0 - u) * (self.denoising_steps - 1))
            start_x = max(0, min(start_x, self.denoising_steps))
        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        if start_x < self.denoising_steps:
            t_all = t_all[start_x:]
        chain = [] if return_chain else None
        if return_chain:
            if not self.use_ddim and self.ft_denoising_steps == self.denoising_steps:
                chain.append(x)
            if self.use_ddim and self.ft_denoising_steps == self.ddim_steps:
                chain.append(x)
        x = self.update_targetaction(cond, x, use_target_critic=use_target_critic_for_improvement)
        if start_x < self.denoising_steps:
            for i in range(start_x):
                chain.append(x)
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar, _ = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
                use_base_policy=use_base_policy,
                deterministic=deterministic,
            )
            std = torch.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                if deterministic:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=min_sampling_denoising_std)
            else:
                if deterministic and t == 0:
                    std = torch.zeros_like(std)
                elif deterministic:  # still keep the original noise
                    std = torch.clip(std, min=1e-3)
                else:  # use higher minimum noise
                    std = torch.clip(std, min=min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )

            if return_chain:
                if not self.use_ddim and t <= self.ft_denoising_steps:
                    chain.append(x)
                elif self.use_ddim and i >= (
                    self.ddim_steps - self.ft_denoising_steps - 1
                ):
                    chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)

        return Sample(x, chain)

    def forward_again(
            self,
            cond,
            deterministic=False,
            return_chain=True,
            use_base_policy=False,
            start_x=None,
    ):
        """
        Forward pass for sampling actions.

        Args:
            eval: mode
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            deterministic: If true, then std=0 with DDIM, or with DDPM, use normal schedule (instead of clipping at a higher value)
            return_chain: whether to return the entire chain of denoised actions
            use_base_policy: whether to use the frozen pre-trained policy instead
        Return:
            Sample: namedtuple with fields:
                trajectories: (B, Ta, Da)
                chain: (B, K + 1, Ta, Da)
        """
        device = self.betas.device
        if isinstance(cond, dict):
            cond = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cond.items()}
        elif torch.is_tensor(cond):
            cond = cond.to(device)



        cond, _ = self.encoder(
            cond['neighbor_trajs'],
            mask=None,
            test=None,
            init_state=cond['ego_state'],
            map_state=cond['neighbor_waypoints']
        )
        B = cond.shape[0]

        # Get updated minimum sampling denoising std
        min_sampling_denoising_std = self.get_min_sampling_denoising_std()

        # Loop
        if start_x == None:
            x = torch.randn((B, self.horizon_steps, self.action_dim), device=device)
        else:
            x = start_x.to(device)

        if self.use_ddim:
            t_all = self.ddim_t
        else:
            t_all = list(reversed(range(self.denoising_steps)))
        chain = [] if return_chain else None
        if not self.use_ddim and self.ft_denoising_steps == self.denoising_steps:
            chain.append(x)
        if self.use_ddim and self.ft_denoising_steps == self.ddim_steps:
            chain.append(x)
        for i, t in enumerate(t_all):
            t_b = make_timesteps(B, t, device)
            index_b = make_timesteps(B, i, device)
            mean, logvar, _ = self.p_mean_var(
                x=x,
                t=t_b,
                cond=cond,
                index=index_b,
                use_base_policy=use_base_policy,
                deterministic=deterministic,
            )
            std = torch.exp(0.5 * logvar)

            # Determine noise level
            if self.use_ddim:
                if deterministic:
                    std = torch.zeros_like(std)
                else:
                    std = torch.clip(std, min=min_sampling_denoising_std)
            else:
                if deterministic and t == 0:
                    std = torch.zeros_like(std)
                elif deterministic:  # still keep the original noise
                    std = torch.clip(std, min=1e-3)
                else:  # use higher minimum noise
                    std = torch.clip(std, min=min_sampling_denoising_std)
            noise = torch.randn_like(x).clamp_(
                -self.randn_clip_value, self.randn_clip_value
            )
            x = mean + std * noise

            # clamp action at final step
            if self.final_action_clip_value is not None and i == len(t_all) - 1:
                x = torch.clamp(
                    x, -self.final_action_clip_value, self.final_action_clip_value
                )

            if return_chain:
                if not self.use_ddim and t <= self.ft_denoising_steps:
                    chain.append(x)
                elif self.use_ddim and i >= (
                        self.ddim_steps - self.ft_denoising_steps - 1
                ):
                    chain.append(x)

        if return_chain:
            chain = torch.stack(chain, dim=1)

        return Sample(x, chain)
    # ---------- RL training ----------#

    def get_logprobs(
        self,
        cond,
        chains,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """
        Calculating the logprobs of the entire chain of denoised actions.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            get_ent: flag for returning entropy
            use_base_policy: flag for using base policy

        Returns:
            logprobs: (B x K, Ta, Da)
            entropy (if get_ent=True):  (B x K, Ta)
        """
        # Repeat cond for denoising_steps, flatten batch and time dimensions
        if isinstance(cond, dict):
            cond_enc, _ = self.encoder(
                cond['neighbor_trajs'],
                mask=None,
                test=None,
                init_state=cond['ego_state'],
                map_state=cond['neighbor_waypoints']
            )
            # Encoder returns (B, num_modes, obs_dim) or (B, obs_dim).
            # Flatten to (B, obs_dim) before repeating so unsqueeze always produces 3D.
            cond_enc = cond_enc.flatten(start_dim=1)  # (B, obs_dim)

            # Repeat the time axis only: (B, obs_dim) → (B, ft, obs_dim)
            cond_enc = cond_enc.unsqueeze(1).repeat(1, self.ft_denoising_steps, 1)

            # (B, ft, obs_dim) → (B*ft, obs_dim), row b*ft+t = obs for batch b
            cond_enc = cond_enc.flatten(start_dim=0, end_dim=1)

            cond = cond_enc

        # Repeat t for batch dim, keep it 1-dim
        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = torch.arange(
                start=self.ft_denoising_steps - 1,
                end=-1,
                step=-1,
                device=self.device,
            )
            # 4,3,2,1,0,4,3,2,1,0,...,4,3,2,1,0
        t_all = t_single.repeat(chains.shape[0], 1).flatten()
        if self.use_ddim:
            indices_single = torch.arange(
                start=self.ddim_steps - self.ft_denoising_steps,
                end=self.ddim_steps,
                device=self.device,
            )  # only used for DDIM
            indices = indices_single.repeat(chains.shape[0])
        else:
            indices = None

        # Split chains
        chains_prev = chains[:, :-1]
        chains_next = chains[:, 1:]

        # Flatten first two dimensions
        chains_prev = chains_prev.reshape(-1, self.horizon_steps, self.action_dim)
        chains_next = chains_next.reshape(-1, self.horizon_steps, self.action_dim)

        # Forward pass with previous chains
        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=indices,
            use_base_policy=use_base_policy,
        )
        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        dist = Normal(next_mean, std)

        # Get logprobs with gaussian
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    def get_logprobs_subsample(
        self,
        cond,
        chains_prev,
        chains_next,
        denoising_inds,
        get_ent: bool = False,
        use_base_policy: bool = False,
    ):
        """
        Calculating the logprobs of random samples of denoised chains.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            get_ent: flag for returning entropy
            use_base_policy: flag for using base policy

        Returns:
            logprobs: (B, Ta, Da)
            entropy (if get_ent=True):  (B, Ta)
            denoising_indices: (B, )
        """
        # Sample t for batch dim, keep it 1-dim
        if self.use_ddim:
            t_single = self.ddim_t[-self.ft_denoising_steps :]
        else:
            t_single = torch.arange(
                start=self.ft_denoising_steps - 1,
                end=-1,
                step=-1,
                device=self.device,
            )
            # 4,3,2,1,0,4,3,2,1,0,...,4,3,2,1,0
        t_all = t_single[denoising_inds]
        if self.use_ddim:
            ddim_indices_single = torch.arange(
                start=self.ddim_steps - self.ft_denoising_steps,
                end=self.ddim_steps,
                device=self.device,
            )  # only used for DDIM
            ddim_indices = ddim_indices_single[denoising_inds]
        else:
            ddim_indices = None

        # Forward pass with previous chains
        next_mean, logvar, eta = self.p_mean_var(
            chains_prev,
            t_all,
            cond=cond,
            index=ddim_indices,
            use_base_policy=use_base_policy,
        )

        std = torch.exp(0.5 * logvar)
        std = torch.clip(std, min=self.min_logprob_denoising_std)
        dist = Normal(next_mean, std)

        # Get logprobs with gaussian
        log_prob = dist.log_prob(chains_next)
        if get_ent:
            return log_prob, eta
        return log_prob

    def loss(self, cond, chains, reward):
        """
        REINFORCE loss. Not used right now.

        Args:
            cond: dict with key state/rgb; more recent obs at the end
                state: (B, To, Do)
                rgb: (B, To, C, H, W)
            chains: (B, K+1, Ta, Da)
            reward (to go): (b,)
        """
        # Get advantage
        with torch.no_grad():
            value = self.critic(cond).squeeze()
        advantage = reward - value

        # Get logprobs for denoising steps from T-1 to 0
        logprobs, eta = self.get_logprobs(cond, chains, get_ent=True)
        # (n_steps x n_envs x K) x Ta x (Do+Da)

        # Ignore obs dimension, and then sum over action dimension
        logprobs = logprobs[:, :, : self.action_dim].sum(-1)
        # -> (n_steps x n_envs x K) x Ta

        # -> (n_steps x n_envs) x K x Ta
        logprobs = logprobs.reshape((-1, self.denoising_steps, self.horizon_steps))

        # Sum/avg over denoising steps
        logprobs = logprobs.mean(-2)  # -> (n_steps x n_envs) x Ta

        # Sum/avg over horizon steps
        logprobs = logprobs.mean(-1)  # -> (n_steps x n_envs)

        # Get REINFORCE loss
        loss_actor = torch.mean(-logprobs * advantage)

        # Train critic to predict state value
        pred = self.critic(cond).squeeze()
        loss_critic = F.mse_loss(pred, reward)
        return loss_actor, loss_critic, eta
    def update_targetaction(self, obs, action, use_target_critic=False):
        import random
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        lim = 1 - 1e-5
        action = action.clamp(-lim, lim)

        # Use target critic when computing Bellman targets to avoid circular dependency.
        # Use current critic during rollouts for online action improvement.
        q_net = self.critic_target if use_target_critic else self.critic
        q_net.requires_grad_(False)

        rangee = 0.2 + 0.4 * random.random()
        candidates = [action + delta for delta in [-rangee, 0.0, rangee]]
        best_action = action
        best_q = -float('inf')
        for cand in candidates:
            q = q_net.get_q_min(obs, cand).mean().item()
            if q > best_q:
                best_q = q
                best_action = cand

        q_net.requires_grad_(True)
        return deepcopy(best_action.clamp(-lim, lim).detach())
