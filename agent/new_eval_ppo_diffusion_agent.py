"""
新的 DPPO CARLA 评估 Agent：动作生成方式与训练步骤完全一致。

关键区别（对比 EvalPPODiffusionAgent）：
  - 不调用 update_target_action（无 critic 引导）
  - 不调用 forward_again
  - 纯扩散采样后直接 step CARLA，与 train 模式保持一致
  - 额外记录加速度幅值、jerk 等舒适度指标
"""

import logging
import os
from datetime import datetime

import numpy as np
import torch
import wandb

from agent.eval_ppo_diffusion_agent import EvalPPODiffusionAgent

log = logging.getLogger(__name__)


class NewEvalPPODiffusionAgent(EvalPPODiffusionAgent):
    """
    动作生成与训练步骤完全一致的评估 Agent：
      model(deterministic=True) → trajectories[:, :act_steps] → venv.step()
    不经过 update_target_action / forward_again。
    """

    def _run_one_episode(self, max_steps):
        obs = self.reset_env_all(options_venv=[{} for _ in range(self.n_envs)])
        done = False
        step_count = 0
        ep_reward = 0.0
        ep_best_reward = -1e9

        finish = False
        collision = False
        off_route = False
        max_time = False
        avg_speed = 0.0
        end_timestep = 0

        ego_acc_mags = []  # ||a|| per step
        acc_xs = []         # ax per step
        acc_ys = []         # ay per step

        jerk_dt = float(self.eval_cfg.get("jerk_dt", 0.1))
        if jerk_dt <= 0:
            jerk_dt = 0.1

        while (not done) and (step_count < max_steps):
            with torch.no_grad():
                cond, obs = self.process_prev_obs(obs)
                # 与训练步骤完全一致：纯扩散采样，不做任何 critic 引导
                samples = self.model(cond=cond, deterministic=True, return_chain=True)
                action = samples.trajectories.cpu().numpy()[:, : self.act_steps]

            # 直接 step 环境，不经过 update_target_action / forward_again
            next_obs, reward, terminated, info = self.venv.step(action)
            reward_val = float(np.asarray(reward).reshape(-1)[0]) * float(self.reward_scale)
            ep_reward += reward_val
            ep_best_reward = max(ep_best_reward, reward_val / max(1, self.act_steps))
            step_count += 1
            obs = next_obs

            # 解析每步 info，提取加速度
            parsed = info
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0 and isinstance(parsed[0], (list, tuple)):
                parsed = parsed[0]

            # ego state: [x, y, vx, vy, ax, ay, steer, yaw]
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 7:
                vehicle_state_dict = parsed[6]
                if isinstance(vehicle_state_dict, dict) and "ego" in vehicle_state_dict:
                    ego_state = vehicle_state_dict["ego"]
                    if isinstance(ego_state, (list, tuple)) and len(ego_state) >= 6:
                        ax = float(ego_state[4])
                        ay = float(ego_state[5])
                        acc_xs.append(ax)
                        acc_ys.append(ay)
                        ego_acc_mags.append(float(np.sqrt(ax * ax + ay * ay)))

            done = bool(np.asarray(terminated).reshape(-1)[0])
            if done:
                if isinstance(parsed, (list, tuple)) and len(parsed) > 0 and isinstance(parsed[0], (list, tuple)):
                    parsed = parsed[0]
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 4:
                    finish = bool(parsed[0])
                    collision = bool(parsed[1])
                    off_route = bool(parsed[2])
                    max_time = bool(parsed[3])
                    if len(parsed) >= 5:
                        end_timestep = int(parsed[4])
                    if len(parsed) >= 6:
                        avg_speed = float(parsed[5])
                break

        success = bool(finish and (not collision) and (not off_route))

        # Jerk：加速度幅值的变化率，用于衡量舒适度
        if len(ego_acc_mags) >= 2:
            jerk = np.diff(np.asarray(ego_acc_mags, dtype=np.float32)) / jerk_dt
            comfort_jerk_abs_mean = float(np.mean(np.abs(jerk)))
            comfort_jerk_rms = float(np.sqrt(np.mean(np.square(jerk))))
        else:
            comfort_jerk_abs_mean = 0.0
            comfort_jerk_rms = 0.0

        avg_acc_mag = float(np.mean(ego_acc_mags)) if ego_acc_mags else 0.0
        max_acc_mag = float(np.max(ego_acc_mags)) if ego_acc_mags else 0.0
        avg_acc_x_abs = float(np.mean(np.abs(acc_xs))) if acc_xs else 0.0
        avg_acc_y_abs = float(np.mean(np.abs(acc_ys))) if acc_ys else 0.0

        return {
            "success": int(success),
            "finish": int(finish),
            "collision": int(collision),
            "off_route": int(off_route),
            "max_time": int(max_time),
            "episode_reward": float(ep_reward),
            "episode_return": float(ep_reward),
            "episode_reward_per_step": float(ep_reward / max(1, step_count)),
            "best_reward": float(ep_best_reward),
            "episode_length": int(step_count),
            "episode_steps": int(step_count),
            "episode_end_timestep": int(end_timestep),
            "avg_speed": float(avg_speed),
            "comfort_jerk_abs_mean": comfort_jerk_abs_mean,
            "comfort_jerk_rms": comfort_jerk_rms,
            "avg_acc_mag": avg_acc_mag,
            "max_acc_mag": max_acc_mag,
            "avg_acc_x_abs": avg_acc_x_abs,
            "avg_acc_y_abs": avg_acc_y_abs,
        }

    def evaluate_checkpoint(self, ckpt_path, episodes, max_steps):
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_step = self._extract_step(ckpt_path)
        run_name = os.path.basename(os.path.dirname(ckpt_dir))
        self.load(ckpt_dir, ckpt_step)
        self.model.eval()

        rows = []
        for ep in range(episodes):
            ep_result = self._run_one_episode(max_steps=max_steps)
            ep_result["episode_index"] = ep
            ep_result["checkpoint"] = ckpt_path
            ep_result["checkpoint_run"] = run_name
            ep_result["checkpoint_step"] = ckpt_step
            rows.append(ep_result)

            def _rm(key):
                return float(np.mean([r[key] for r in rows]))

            def _rs(key):
                return float(np.std([r[key] for r in rows]))

            r_succ_m = _rm("success");       r_succ_s = _rs("success")
            r_speed_m = _rm("avg_speed");    r_speed_s = _rs("avg_speed")
            r_rew_m = _rm("episode_reward"); r_rew_s = _rs("episode_reward")
            r_ret_m = _rm("episode_return"); r_ret_s = _rs("episode_return")
            r_rps_m = _rm("episode_reward_per_step"); r_rps_s = _rs("episode_reward_per_step")
            r_step_m = _rm("episode_steps"); r_step_s = _rs("episode_steps")
            r_ts_m = _rm("episode_end_timestep"); r_ts_s = _rs("episode_end_timestep")
            r_jerk_m = _rm("comfort_jerk_abs_mean"); r_jerk_s = _rs("comfort_jerk_abs_mean")
            r_acc_m = _rm("avg_acc_mag");    r_acc_s = _rs("avg_acc_mag")

            log.info(
                "[EP %d/%d] step=%d | "
                "succ=%.1f%%±%.1f%% | speed=%.4f±%.4f | "
                "reward=%.2f±%.2f | return=%.2f±%.2f | rps=%.2f±%.2f | "
                "steps=%.1f±%.1f | ts=%.1f±%.1f | "
                "jerk=%.4f±%.4f | acc=%.4f±%.4f",
                ep + 1, episodes, ckpt_step,
                r_succ_m * 100, r_succ_s * 100,
                r_speed_m, r_speed_s,
                r_rew_m * 100, r_rew_s * 100,
                r_ret_m * 100, r_ret_s * 100,
                r_rps_m * 100, r_rps_s * 100,
                r_step_m, r_step_s, r_ts_m, r_ts_s,
                r_jerk_m, r_jerk_s, r_acc_m, r_acc_s,
            )

            if self.use_wandb:
                wandb.log(
                    {
                        "eval/episode_success": ep_result["success"],
                        "eval/episode_finish": ep_result["finish"],
                        "eval/episode_collision": ep_result["collision"],
                        "eval/episode_off_route": ep_result["off_route"],
                        "eval/episode_max_time": ep_result["max_time"],
                        "eval/episode_speed": ep_result["avg_speed"],
                        "eval/episode_reward": ep_result["episode_reward"],
                        "eval/episode_return": ep_result["episode_return"],
                        "eval/episode_reward_per_step": ep_result["episode_reward_per_step"],
                        "eval/episode_steps": ep_result["episode_steps"],
                        "eval/episode_end_timestep": ep_result["episode_end_timestep"],
                        "eval/episode_comfort_jerk_abs_mean": ep_result["comfort_jerk_abs_mean"],
                        "eval/episode_comfort_jerk_rms": ep_result["comfort_jerk_rms"],
                        "eval/episode_avg_acc_mag": ep_result["avg_acc_mag"],
                        "eval/episode_max_acc_mag": ep_result["max_acc_mag"],
                        "eval/episode_avg_acc_x_abs": ep_result["avg_acc_x_abs"],
                        "eval/episode_avg_acc_y_abs": ep_result["avg_acc_y_abs"],
                        "eval/running_success%_mean": r_succ_m * 100,
                        "eval/running_success%_std": r_succ_s * 100,
                        "eval/running_avg_speed_mean": r_speed_m,
                        "eval/running_avg_speed_std": r_speed_s,
                        "eval/running_reward(x100)_mean": r_rew_m * 100,
                        "eval/running_reward(x100)_std": r_rew_s * 100,
                        "eval/running_return(x100)_mean": r_ret_m * 100,
                        "eval/running_return(x100)_std": r_ret_s * 100,
                        "eval/running_steps_mean": r_step_m,
                        "eval/running_steps_std": r_step_s,
                        "eval/running_jerk_mean": r_jerk_m,
                        "eval/running_jerk_std": r_jerk_s,
                        "eval/running_acc_mean": r_acc_m,
                        "eval/running_acc_std": r_acc_s,
                        "eval/checkpoint_step": ckpt_step,
                    },
                    step=ep + 1,
                )

        def _mean(key):
            return float(np.mean([r[key] for r in rows])) if rows else 0.0

        def _std(key):
            return float(np.std([r[key] for r in rows])) if rows else 0.0

        summary = {
            "checkpoint": ckpt_path,
            "checkpoint_run": run_name,
            "checkpoint_step": ckpt_step,
            "episodes": int(episodes),
            "success_rate%": _mean("success") * 100,
            "std_success_rate%": _std("success") * 100,
            "finish_rate%": _mean("finish") * 100,
            "collision_rate%": _mean("collision") * 100,
            "off_route_rate%": _mean("off_route") * 100,
            "max_time_rate%": _mean("max_time") * 100,
            "avg_episode_reward_x100": _mean("episode_reward") * 100,
            "std_episode_reward_x100": _std("episode_reward") * 100,
            "avg_episode_return_x100": _mean("episode_return") * 100,
            "std_episode_return_x100": _std("episode_return") * 100,
            "avg_episode_rps_x100": _mean("episode_reward_per_step") * 100,
            "std_episode_rps_x100": _std("episode_reward_per_step") * 100,
            "avg_best_reward_x100": _mean("best_reward") * 100,
            "std_best_reward_x100": _std("best_reward") * 100,
            "avg_episode_steps": _mean("episode_steps"),
            "std_episode_steps": _std("episode_steps"),
            "avg_episode_end_timestep": _mean("episode_end_timestep"),
            "std_episode_end_timestep": _std("episode_end_timestep"),
            "avg_speed": _mean("avg_speed"),
            "std_speed": _std("avg_speed"),
            "avg_comfort_jerk_abs_mean": _mean("comfort_jerk_abs_mean"),
            "std_comfort_jerk_abs_mean": _std("comfort_jerk_abs_mean"),
            "avg_comfort_jerk_rms": _mean("comfort_jerk_rms"),
            "std_comfort_jerk_rms": _std("comfort_jerk_rms"),
            "avg_acc_mag": _mean("avg_acc_mag"),
            "std_acc_mag": _std("avg_acc_mag"),
            "avg_max_acc_mag": _mean("max_acc_mag"),
            "std_max_acc_mag": _std("max_acc_mag"),
            "avg_acc_x_abs": _mean("avg_acc_x_abs"),
            "std_acc_x_abs": _std("avg_acc_x_abs"),
            "avg_acc_y_abs": _mean("avg_acc_y_abs"),
            "std_acc_y_abs": _std("avg_acc_y_abs"),
        }
        return summary, rows

    def run(self):
        ckpts = self._collect_checkpoints()
        if len(ckpts) == 0:
            src = self.eval_cfg.get("checkpoint_dir", "") or self.eval_cfg.get("checkpoint_root", "")
            raise FileNotFoundError(
                f"No checkpoint found under: {src} with pattern {self.eval_cfg.checkpoint_pattern}"
            )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        first_step = self._extract_step(ckpts[0])
        first_run = os.path.basename(os.path.dirname(os.path.dirname(ckpts[0])))
        out_dir = os.path.join(
            os.path.abspath(self.eval_cfg.results_root),
            f"{self.env_name}_{first_run}_state_{first_step}_neweval{int(self.eval_cfg.episodes_per_checkpoint)}_{timestamp}",
        )

        log.info(
            "Start new-eval (no critic guidance) for %d checkpoints, each %d episodes.",
            len(ckpts), self.eval_cfg.episodes_per_checkpoint,
        )
        summary_rows = []
        episode_rows = []

        for idx, ckpt in enumerate(ckpts):
            summary, ep_rows = self.evaluate_checkpoint(
                ckpt_path=ckpt,
                episodes=int(self.eval_cfg.episodes_per_checkpoint),
                max_steps=int(self.eval_cfg.max_episode_steps),
            )
            summary["index"] = idx
            summary_rows.append(summary)
            episode_rows.extend(ep_rows)

            log.info(
                "[%d/%d] step=%d | "
                "succ=%.1f%%±%.1f%% | finish=%.1f%% | coll=%.1f%% | off=%.1f%% | to=%.1f%% | "
                "reward=%.2f±%.2f | steps=%.1f±%.1f | "
                "speed=%.4f±%.4f | jerk=%.4f±%.4f | acc=%.4f±%.4f",
                idx + 1, len(ckpts), summary["checkpoint_step"],
                summary["success_rate%"], summary["std_success_rate%"],
                summary["finish_rate%"], summary["collision_rate%"],
                summary["off_route_rate%"], summary["max_time_rate%"],
                summary["avg_episode_reward_x100"], summary["std_episode_reward_x100"],
                summary["avg_episode_steps"], summary["std_episode_steps"],
                summary["avg_speed"], summary["std_speed"],
                summary["avg_comfort_jerk_abs_mean"], summary["std_comfort_jerk_abs_mean"],
                summary["avg_acc_mag"], summary["std_acc_mag"],
            )

            if self.use_wandb:
                wandb.log(
                    {
                        "eval/checkpoint_step": summary["checkpoint_step"],
                        "eval/success_rate%": summary["success_rate%"],
                        "eval/std_success_rate%": summary["std_success_rate%"],
                        "eval/finish_rate%": summary["finish_rate%"],
                        "eval/collision_rate%": summary["collision_rate%"],
                        "eval/off_route_rate%": summary["off_route_rate%"],
                        "eval/max_time_rate%": summary["max_time_rate%"],
                        "eval/avg_episode_reward_x100": summary["avg_episode_reward_x100"],
                        "eval/std_episode_reward_x100": summary["std_episode_reward_x100"],
                        "eval/avg_best_reward_x100": summary["avg_best_reward_x100"],
                        "eval/std_best_reward_x100": summary["std_best_reward_x100"],
                        "eval/avg_episode_steps": summary["avg_episode_steps"],
                        "eval/std_episode_steps": summary["std_episode_steps"],
                        "eval/avg_speed": summary["avg_speed"],
                        "eval/std_speed": summary["std_speed"],
                        "eval/avg_comfort_jerk_abs_mean": summary["avg_comfort_jerk_abs_mean"],
                        "eval/std_comfort_jerk_abs_mean": summary["std_comfort_jerk_abs_mean"],
                        "eval/avg_comfort_jerk_rms": summary["avg_comfort_jerk_rms"],
                        "eval/avg_acc_mag": summary["avg_acc_mag"],
                        "eval/std_acc_mag": summary["std_acc_mag"],
                        "eval/avg_max_acc_mag": summary["avg_max_acc_mag"],
                        "eval/avg_acc_x_abs": summary["avg_acc_x_abs"],
                        "eval/avg_acc_y_abs": summary["avg_acc_y_abs"],
                    },
                    step=int(idx),
                )

        saved = self._save_results(out_dir=out_dir, summary_rows=summary_rows, episode_rows=episode_rows)
        log.info("New-eval finished. Output dir: %s", out_dir)
        log.info("Saved summary csv:  %s", saved["summary_csv"])
        log.info("Saved episode csv:  %s", saved["episode_csv"])
        log.info("Saved summary json: %s", saved["summary_json"])
