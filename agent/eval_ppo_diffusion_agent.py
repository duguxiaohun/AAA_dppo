"""
DPPO CARLA 批量评估 Agent（仅评估，不训练）。
"""

import csv
import glob
import json
import logging
import os
import re
from datetime import datetime

import numpy as np
import torch
import wandb

from agent.train_agent import TrainAgent

log = logging.getLogger(__name__)


class EvalPPODiffusionAgent(TrainAgent):
    """批量加载 checkpoint 并执行纯评估。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.eval_cfg = cfg.eval

    def process_prev_obs(self, prev_obs_venv):
        """统一处理观测格式，返回模型输入 cond。"""
        device = self.device
        if isinstance(prev_obs_venv, list) and len(prev_obs_venv) > 0:
            first = prev_obs_venv[0]
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
            neighbor_trajs = np.asarray(prev_obs_venv[0])[None, ...]
            ego_state = np.asarray(prev_obs_venv[1])[None, ...]
            neighbor_wps = np.asarray(prev_obs_venv[2])[None, ...]
            prev_obs_venv = {
                "neighbor_trajs": neighbor_trajs,
                "ego_state": ego_state,
                "neighbor_waypoints": neighbor_wps,
            }
        elif isinstance(prev_obs_venv, dict):
            keep = ("neighbor_trajs", "ego_state", "neighbor_waypoints")
            prev_obs_venv = {k: np.asarray(prev_obs_venv[k]) for k in keep if k in prev_obs_venv}
        else:
            raise TypeError(f"Unsupported prev_obs_venv type: {type(prev_obs_venv)}")

        cond = {
            "neighbor_trajs": torch.from_numpy(prev_obs_venv["neighbor_trajs"]).float().to(device),
            "ego_state": torch.from_numpy(prev_obs_venv["ego_state"]).float().to(device),
            "neighbor_waypoints": torch.from_numpy(prev_obs_venv["neighbor_waypoints"]).float().to(device),
        }
        return cond, prev_obs_venv

    @staticmethod
    def _extract_step(ckpt_path):
        m = re.search(r"state_(\d+)\.pt$", ckpt_path)
        return int(m.group(1)) if m else -1

    def _collect_checkpoints(self):
        ckpt_dir = os.path.abspath(self.eval_cfg.checkpoint_dir)
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"checkpoint_dir does not exist: {ckpt_dir}")

        step = int(self.eval_cfg.get("checkpoint_step", -1))
        if step >= 0:
            ckpt_path = os.path.join(ckpt_dir, f"state_{step}.pt")
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
            return [ckpt_path]

        pattern = os.path.join(ckpt_dir, self.eval_cfg.checkpoint_pattern)
        ckpts = [p for p in glob.glob(pattern) if os.path.isfile(p)]
        if len(ckpts) == 0:
            raise FileNotFoundError(
                f"No checkpoint found under: {ckpt_dir} with pattern {self.eval_cfg.checkpoint_pattern}"
            )

        # 默认每次只评估一个：取目录内最后一个 ckpt。
        latest = sorted(ckpts, key=lambda p: (self._extract_step(p), p))[-1]
        return [latest]

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
        ego_acc_mags = []

        jerk_dt = float(self.eval_cfg.get("jerk_dt", 0.1))
        if jerk_dt <= 0:
            jerk_dt = 0.1

        while (not done) and (step_count < max_steps):
            with torch.no_grad():
                cond, obs = self.process_prev_obs(obs)
                samples = self.model(cond=cond, deterministic=True, return_chain=True)
                action = samples.trajectories.cpu().numpy()[:, : self.act_steps]

            if self.eval_cfg.get("use_target_action_refine", True):
                _, action = self.model.update_target_action(obs, action)
                with torch.no_grad():
                    cond, obs = self.process_prev_obs(obs)
                    samples = self.model.forward_again(
                        cond=cond,
                        deterministic=True,
                        return_chain=True,
                        start_x=action,
                    )
                    action = samples.trajectories.cpu().numpy()[:, : self.act_steps]

            next_obs, reward, terminated, info = self.venv.step(action)
            reward_val = float(np.asarray(reward).reshape(-1)[0]) * float(self.reward_scale)
            ep_reward += reward_val
            ep_best_reward = max(ep_best_reward, reward_val / max(1, self.act_steps))
            step_count += 1
            obs = next_obs

            parsed = info
            if isinstance(parsed, (list, tuple)) and len(parsed) > 0 and isinstance(parsed[0], (list, tuple)):
                parsed = parsed[0]

            # 车辆状态里 ego = [x, y, vx, vy, ax, ay, steer, yaw]
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 7:
                vehicle_state_dict = parsed[6]
                if isinstance(vehicle_state_dict, dict) and "ego" in vehicle_state_dict:
                    ego_state = vehicle_state_dict["ego"]
                    if isinstance(ego_state, (list, tuple)) and len(ego_state) >= 6:
                        ax = float(ego_state[4])
                        ay = float(ego_state[5])
                        ego_acc_mags.append(float(np.sqrt(ax * ax + ay * ay)))

            done = bool(np.asarray(terminated).reshape(-1)[0])
            if done:
                # 向量环境下常见为 [env0_info]，这里提取单环境 info。
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

        if len(ego_acc_mags) >= 2:
            jerk = np.diff(np.asarray(ego_acc_mags, dtype=np.float32)) / jerk_dt
            comfort_jerk_abs_mean = float(np.mean(np.abs(jerk)))
            comfort_jerk_rms = float(np.sqrt(np.mean(np.square(jerk))))
        else:
            comfort_jerk_abs_mean = 0.0
            comfort_jerk_rms = 0.0

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

            running_success_rate = float(np.mean([r["success"] for r in rows]))
            running_avg_speed = float(np.mean([r["avg_speed"] for r in rows]))
            running_avg_reward = float(np.mean([r["episode_reward"] for r in rows]))
            running_avg_return = float(np.mean([r["episode_return"] for r in rows]))
            running_avg_reward_per_step = float(np.mean([r["episode_reward_per_step"] for r in rows]))
            running_avg_steps = float(np.mean([r["episode_steps"] for r in rows]))
            running_avg_end_timestep = float(np.mean([r["episode_end_timestep"] for r in rows]))
            running_comfort = float(np.mean([r["comfort_jerk_abs_mean"] for r in rows]))

            log.info(
                "[EP %d/%d] step=%d | succ_rate=%.4f | avg_speed=%.4f | avg_reward=%.4f | avg_return=%.4f | avg_rps=%.4f | avg_steps=%.2f | avg_timestep=%.2f | comfort_jerk=%.4f",
                ep + 1,
                episodes,
                ckpt_step,
                running_success_rate,
                running_avg_speed,
                running_avg_reward,
                running_avg_return,
                running_avg_reward_per_step,
                running_avg_steps,
                running_avg_end_timestep,
                running_comfort,
            )

            if self.use_wandb:
                wandb.log(
                    {
                        "eval/episode_success": ep_result["success"],
                        "eval/episode_speed": ep_result["avg_speed"],
                        "eval/episode_reward": ep_result["episode_reward"],
                        "eval/episode_return": ep_result["episode_return"],
                        "eval/episode_reward_per_step": ep_result["episode_reward_per_step"],
                        "eval/episode_steps": ep_result["episode_steps"],
                        "eval/episode_end_timestep": ep_result["episode_end_timestep"],
                        "eval/episode_comfort_jerk_abs_mean": ep_result["comfort_jerk_abs_mean"],
                        "eval/episode_comfort_jerk_rms": ep_result["comfort_jerk_rms"],
                        "eval/running_success_rate": running_success_rate,
                        "eval/running_avg_speed": running_avg_speed,
                        "eval/running_avg_reward": running_avg_reward,
                        "eval/running_avg_return": running_avg_return,
                        "eval/running_avg_reward_per_step": running_avg_reward_per_step,
                        "eval/running_avg_steps": running_avg_steps,
                        "eval/running_avg_end_timestep": running_avg_end_timestep,
                        "eval/running_avg_comfort_jerk_abs_mean": running_comfort,
                        "eval/checkpoint_step": ckpt_step,
                    },
                    step=ep + 1,
                )

        def _mean(key):
            return float(np.mean([r[key] for r in rows])) if rows else 0.0

        summary = {
            "checkpoint": ckpt_path,
            "checkpoint_run": run_name,
            "checkpoint_step": ckpt_step,
            "episodes": int(episodes),
            "success_rate": _mean("success"),
            "finish_rate": _mean("finish"),
            "collision_rate": _mean("collision"),
            "off_route_rate": _mean("off_route"),
            "max_time_rate": _mean("max_time"),
            "avg_episode_reward": _mean("episode_reward"),
            "avg_episode_return": _mean("episode_return"),
            "avg_episode_reward_per_step": _mean("episode_reward_per_step"),
            "avg_best_reward": _mean("best_reward"),
            "avg_episode_length": _mean("episode_length"),
            "avg_episode_steps": _mean("episode_steps"),
            "avg_episode_end_timestep": _mean("episode_end_timestep"),
            "avg_speed": _mean("avg_speed"),
            "avg_comfort_jerk_abs_mean": _mean("comfort_jerk_abs_mean"),
            "avg_comfort_jerk_rms": _mean("comfort_jerk_rms"),
        }
        return summary, rows

    def _save_results(self, out_dir, summary_rows, episode_rows):
        os.makedirs(out_dir, exist_ok=True)

        summary_csv = os.path.join(out_dir, "summary.csv")
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
            if summary_rows:
                writer.writeheader()
                writer.writerows(summary_rows)

        episode_csv = os.path.join(out_dir, "per_episode.csv")
        with open(episode_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()) if episode_rows else [])
            if episode_rows:
                writer.writeheader()
                writer.writerows(episode_rows)

        summary_json = os.path.join(out_dir, "summary.json")
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "env_name": self.env_name,
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_checkpoints": len(summary_rows),
                    "episodes_per_checkpoint": int(self.eval_cfg.episodes_per_checkpoint),
                    "results": summary_rows,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        return {
            "summary_csv": summary_csv,
            "episode_csv": episode_csv,
            "summary_json": summary_json,
        }

    def run(self):
        ckpts = self._collect_checkpoints()
        if len(ckpts) == 0:
            raise FileNotFoundError(
                f"No checkpoint found under: {self.eval_cfg.checkpoint_dir} with pattern {self.eval_cfg.checkpoint_pattern}"
            )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        first_step = self._extract_step(ckpts[0])
        first_run = os.path.basename(os.path.dirname(os.path.dirname(ckpts[0])))
        out_dir = os.path.join(
            os.path.abspath(self.eval_cfg.results_root),
            f"{self.env_name}_{first_run}_state_{first_step}_eval{int(self.eval_cfg.episodes_per_checkpoint)}_{timestamp}",
        )

        log.info("Start evaluating %d checkpoints, each %d episodes.", len(ckpts), self.eval_cfg.episodes_per_checkpoint)
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
                "[%d/%d] step=%d success=%.4f finish=%.4f collision=%.4f off_route=%.4f max_time=%.4f",
                idx + 1,
                len(ckpts),
                summary["checkpoint_step"],
                summary["success_rate"],
                summary["finish_rate"],
                summary["collision_rate"],
                summary["off_route_rate"],
                summary["max_time_rate"],
            )

            if self.use_wandb:
                wandb.log(
                    {
                        "eval/checkpoint_step": summary["checkpoint_step"],
                        "eval/success_rate": summary["success_rate"],
                        "eval/finish_rate": summary["finish_rate"],
                        "eval/collision_rate": summary["collision_rate"],
                        "eval/off_route_rate": summary["off_route_rate"],
                        "eval/max_time_rate": summary["max_time_rate"],
                        "eval/avg_episode_reward": summary["avg_episode_reward"],
                        "eval/avg_best_reward": summary["avg_best_reward"],
                        "eval/avg_episode_length": summary["avg_episode_length"],
                        "eval/avg_speed": summary["avg_speed"],
                    },
                    step=int(idx),
                )

        saved = self._save_results(out_dir=out_dir, summary_rows=summary_rows, episode_rows=episode_rows)
        log.info("Evaluation finished. Output dir: %s", out_dir)
        log.info("Saved summary csv: %s", saved["summary_csv"])
        log.info("Saved episode csv: %s", saved["episode_csv"])
        log.info("Saved summary json: %s", saved["summary_json"])
