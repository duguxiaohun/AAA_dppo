"""
DPPO: Diffusion Policy Policy Optimization. 

K: number of denoising steps
To: observation sequence length
Ta: action chunk size
Do: observation dimension
Da: action dimension

C: image channels
H, W: image height and width

"""
from copy import deepcopy
from typing import Optional
import torch
import logging
import math
import torch.nn.functional as F
from ddiffpg.utils.distl_util import projection

log = logging.getLogger(__name__)
from model.diffusion.diffusion_vpg import VPGDiffusion


class PPODiffusion(VPGDiffusion):
    def __init__(
        self,
        gamma_denoising: float,
        clip_ploss_coef: float,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0,
        clip_advantage_upper_quantile: float = 1,
        norm_adv: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Whether to normalize advantages within batch
        self.norm_adv = norm_adv

        # Clipping value for policy loss
        self.clip_ploss_coef = clip_ploss_coef
        self.clip_ploss_coef_base = clip_ploss_coef_base
        self.clip_ploss_coef_rate = clip_ploss_coef_rate

        # Clipping value for value loss
        self.clip_vloss_coef = clip_vloss_coef

        # Discount factor for diffusion MDP
        self.gamma_denoising = gamma_denoising

        # Quantiles for clipping advantages
        self.clip_advantage_lower_quantile = clip_advantage_lower_quantile
        self.clip_advantage_upper_quantile = clip_advantage_upper_quantile

    def loss(
        self,
        obs,
        chains_prev,
        chains_next,
        denoising_inds,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        reward_horizon=4,
    ):
        """
        PPO loss

        obs: dict with key state/rgb; more recent obs at the end
            state: (batch, n_cond_step, obs_dim) for furniture
            rgb: (B, To, C, H, W)
        chains: (B, K+1, Ta, Da)
        returns: (B, )
        values: (B, )
        advantages: (B,)
        oldlogprobs: (B, K, Ta, Da)
        use_bc_loss: whether to add BC regularization loss
        reward_horizon: action horizon that backpropagates gradient
        """
        # Get new logprobs for denoising steps from T-1 to 0 - entropy is fixed fod diffusion
        if isinstance(obs, dict):
            obs, _ = self.encoder(
                obs['neighbor_trajs'],
                mask=None,
                test=obs,
                init_state=obs['ego_state'],
                map_state=obs['neighbor_waypoints']
            )


        newlogprobs, eta = self.get_logprobs_subsample(
            obs,
            chains_prev,
            chains_next,
            denoising_inds,
            get_ent=True,
            # use_base_policy=True,
        )
        # newlogprobs  shape: torch.Size([5000, 4, 3]), eta shape: torch.Size([5000, 4, 3])
        entropy_loss = -eta.mean()
        newlogprobs = newlogprobs.clamp(min=-5, max=2)
        oldlogprobs = oldlogprobs.clamp(min=-5, max=2)
        # torch.Size([50000, 4, 3])
        # torch.Size([50000, 4, 3])

        # only backpropagate through the earlier steps (e.g., ones actually executed in the environment)
        newlogprobs = newlogprobs[:, :reward_horizon, :]
        oldlogprobs = oldlogprobs[:, :reward_horizon, :]

        # sum over action dims (joint log_prob), then mean over horizon
        newlogprobs = newlogprobs.sum(dim=-1).mean(dim=-1).view(-1)
        oldlogprobs = oldlogprobs.sum(dim=-1).mean(dim=-1).view(-1)
        # B

        bc_loss = 0
        if use_bc_loss:
            # See Eqn. 2 of https://arxiv.org/pdf/2403.03949.pdf
            # Give a reward for maximizing probability of teacher policy's action with current policy.
            # Actions are chosen along trajectory induced by current policy.

            # Get counterfactual teacher actions
            samples = self.forward(
                cond=obs,
                deterministic=False,
                return_chain=True,
                use_base_policy=True,
            )
            # Get logprobs of teacher actions under this policy
            bc_logprobs = self.get_logprobs(
                obs,
                samples.chains,
                get_ent=False,
                use_base_policy=False,
            )
            bc_logprobs = bc_logprobs.clamp(min=-5, max=2)
            bc_logprobs = bc_logprobs.mean(dim=(-1, -2)).view(-1)
            bc_loss = -bc_logprobs.mean()
            # B

        # normalize advantages
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Clip advantages by 5th and 95th percentile
        advantage_min = torch.quantile(advantages, self.clip_advantage_lower_quantile)
        advantage_max = torch.quantile(advantages, self.clip_advantage_upper_quantile)
        advantages = advantages.clamp(min=advantage_min, max=advantage_max)

        # denoising discount
        discount = torch.tensor(
            [
                self.gamma_denoising ** (self.ft_denoising_steps - i - 1)
                for i in denoising_inds
            ]
        ).to(self.device)
        advantages *= discount

        # get ratio
        logratio = newlogprobs - oldlogprobs

        ratio = logratio.exp()

        # exponentially interpolate between the base and the current clipping value over denoising steps and repeat
        if self.ft_denoising_steps > 1:
            t = (denoising_inds.float() / (self.ft_denoising_steps - 1)).to(self.device)
            clip_ploss_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (torch.exp(self.clip_ploss_coef_rate * t) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1
            )
        else:
            clip_ploss_coef = torch.full_like(
                denoising_inds.float(), self.clip_ploss_coef, device=self.device
            )

        # get kl difference and whether value clipped
        with torch.no_grad():
            # old_approx_kl: the approximate Kullback–Leibler divergence, measured by (-logratio).mean(), which corresponds to the k1 estimator in John Schulman’s blog post on approximating KL http://joschu.net/blog/kl-approx.html
            # approx_kl: better alternative to old_approx_kl measured by (logratio.exp() - 1) - logratio, which corresponds to the k3 estimator in approximating KL http://joschu.net/blog/kl-approx.html
            # old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfrac = ((ratio - 1.0).abs() > clip_ploss_coef).float().mean().item()

        # Policy loss with clipping
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(
            ratio, 1 - clip_ploss_coef, 1 + clip_ploss_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss optionally with clipping
        newvalues = self.critic.get_v(obs).view(-1)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch.clamp(
                newvalues - oldvalues,
                -self.clip_vloss_coef,
                self.clip_vloss_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        return (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            approx_kl.item(),
            ratio.mean().item(),
            bc_loss,
            eta.mean().item(),
        )


    def update_critic(self, obs, action, reward, next_obs, done):
        # next_actions must not depend on self.critic (only critic_target) to avoid
        # a circular dependency where the Bellman target moves with the critic itself.
        with torch.no_grad():
            next_actions = self.forward(
                cond=next_obs, deterministic=True, return_chain=False,
                use_base_policy=False, use_target_critic_for_improvement=True,
            ).trajectories

        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target.get_q1_q2(next_obs, next_actions)
            target_Q1_projected = projection(next_dist=target_Q1,
                                             reward=reward,
                                             done=done,
                                             gamma=self.gamma ** self.nstep,
                                             v_min=self.critic.v_min,
                                             v_max=self.critic.v_max,
                                             num_atoms=self.critic.num_atoms,
                                             support=self.critic.z_atoms,
                                             device=self.device)
            target_Q2_projected = projection(next_dist=target_Q2,
                                             reward=reward,
                                             done=done,
                                             gamma=self.gamma ** self.nstep,
                                             v_min=self.critic.v_min,
                                             v_max=self.critic.v_max,
                                             num_atoms=self.critic.num_atoms,
                                             support=self.critic.z_atoms,
                                             device=self.device)
            target_Q = torch.min(target_Q1_projected, target_Q2_projected)

        current_Q1, current_Q2 = self.critic.get_q1_q2(obs, action)
        # critic_loss = F.binary_cross_entropy(current_Q1, target_Q) + F.binary_cross_entropy(current_Q2, target_Q)
        critic_loss = -torch.sum(target_Q * torch.log(current_Q1 + 1e-8), dim=1).mean() \
                      - torch.sum(target_Q * torch.log(current_Q2 + 1e-8), dim=1).mean()

        return critic_loss



    def update_actor(self, obs, target_action):
        with torch.no_grad():
            obs, _ = self.encoder(
                obs['neighbor_trajs'],
                mask=None,
                test=obs,
                init_state=obs['ego_state'],
                map_state=obs['neighbor_waypoints']
            )
        actor_loss = self.bc_loss(target_action, obs)
        return actor_loss

    def update_target_action(self, obs, action):
        """
        Update target action based on current obs and action.
        This is done by optimizing the critic's Q value w.r.t. action.
        """
        # action: (B, Ta, Da)
        # obs: dict with key state/rgb; more recent obs at the end
        # state: (B, n_cond_step, obs_dim) for furniture
        # rgb: (B, To, C, H, W)
        # obs = obs['state'] if isinstance(obs, dict) else obs
        # if not isinstance(obs, torch.Tensor):
        #     obs = torch.tensor(obs, dtype=torch.float32, device=self.device)  # 指定 device 和类型
        # obs = obs.view(obs.shape[0], -1)
        if isinstance(obs, dict):
            with torch.no_grad():
                obs, _ = self.encoder(
                    obs['neighbor_trajs'],
                    mask=None,
                    test=obs,
                    init_state=obs['ego_state'],
                    map_state=obs['neighbor_waypoints']
                )

        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)  # 指定 device 和类型
        self.critic.requires_grad_(False)
        lim = 1 - 1e-5
        action.clamp_(-lim, lim)

        action_optimizer = torch.optim.Adam([action], lr=self.action_lr, eps=1e-5)

        for _ in range(self.update_times):
            action.requires_grad_(True)

            Q = self.critic.get_q_min(obs, action)
            loss = -Q.mean()


            self.optimizer_update(action_optimizer, loss)
            action.requires_grad_(False)
            action.clamp_(-lim, lim)  # ⚠️ 保持 in-place，不断链

        target_action = action.detach().to(self.device)
        update = deepcopy(target_action)
        self.critic.requires_grad_(True)
        return torch.abs(action).mean().item(), update


    def optimizer_update(self, optimizer, objective):
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        from torch.nn.utils import clip_grad_norm_

        if self.max_grad_norm is not None:
            grad_norm = clip_grad_norm_(parameters=optimizer.param_groups[0]["params"],
                                        max_norm=self.max_grad_norm)
        else:
            grad_norm = None
        optimizer.step()
        return grad_norm





    def offline_loss(
        self,
        obs,
        denoising_inds,
        returns,
        oldvalues,
        oldlogprobs,
        use_bc_loss=False,
        reward_horizon=4,
    ):
        """
        PPO loss

        obs: dict with key state/rgb; more recent obs at the end
            state: (batch, n_cond_step, obs_dim) for furniture
            rgb: (B, To, C, H, W)
        chains: (B, K+1, Ta, Da)
        returns: (B, )
        values: (B, )
        oldlogprobs: (B, K, Ta, Da)
        use_bc_loss: whether to add BC regularization loss
        reward_horizon: action horizon that backpropagates gradient
        """
        # Get new logprobs for denoising steps from T-1 to 0 - entropy is fixed fod diffusion
        if isinstance(obs, dict):
            obs, _ = self.encoder(
                obs['neighbor_trajs'],
                mask=None,
                test=obs,
                init_state=obs['ego_state'],
                map_state=obs['neighbor_waypoints']
            )





        bc_loss = 0
        if use_bc_loss:
            # See Eqn. 2 of https://arxiv.org/pdf/2403.03949.pdf
            # Give a reward for maximizing probability of teacher policy's action with current policy.
            # Actions are chosen along trajectory induced by current policy.

            # Get counterfactual teacher actions
            samples = self.forward(
                cond=obs,
                deterministic=False,
                return_chain=True,
                use_base_policy=True,
            )
            # Get logprobs of teacher actions under this policy
            bc_logprobs = self.get_logprobs(
                obs,
                samples.chains,
                get_ent=False,
                use_base_policy=False,
            )
            bc_logprobs = bc_logprobs.clamp(min=-5, max=2)
            bc_logprobs = bc_logprobs.mean(dim=(-1, -2)).view(-1)
            bc_loss = -bc_logprobs.mean()
            # B





        # denoising discount
        discount = torch.tensor(
            [
                self.gamma_denoising ** (self.ft_denoising_steps - i - 1)
                for i in denoising_inds
            ]
        ).to(self.device)

        # get ratio


        # exponentially interpolate between the base and the current clipping value over denoising steps and repeat
        t = (denoising_inds.float() / (self.ft_denoising_steps - 1)).to(self.device)
        if self.ft_denoising_steps > 1:
            clip_ploss_coef = self.clip_ploss_coef_base + (
                self.clip_ploss_coef - self.clip_ploss_coef_base
            ) * (torch.exp(self.clip_ploss_coef_rate * t) - 1) / (
                math.exp(self.clip_ploss_coef_rate) - 1
            )
        else:
            clip_ploss_coef = t


        # Policy loss with clipping


        # Value loss optionally with clipping
        newvalues = self.critic.get_v(obs).view(-1)
        if self.clip_vloss_coef is not None:
            v_loss_unclipped = (newvalues - returns) ** 2
            v_clipped = oldvalues + torch.clamp(
                newvalues - oldvalues,
                -self.clip_vloss_coef,
                self.clip_vloss_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalues - returns) ** 2).mean()
        return (
            v_loss,
            bc_loss,
        )
