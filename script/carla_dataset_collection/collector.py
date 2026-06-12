import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from script.carla_dataset_collection.hdf5_saver import OfflineDatasetSaver


def town_slug_from_env(env_name):
    name = str(env_name).lower()
    if "town03" in name:
        return "town03"
    if "town05" in name:
        return "town05"
    if "town10" in name:
        return "town10"
    return name.replace("carlatown", "town").replace("cross-v0", "").replace("-", "_")


def build_save_dir(output_root, env_name, timestamp):
    town = town_slug_from_env(env_name)
    return os.path.join(output_root, town, timestamp), town


def _as_bool(value):
    arr = np.asarray(value)
    return bool(arr.reshape(-1)[0]) if arr.size else bool(value)


def _as_float(value):
    arr = np.asarray(value, dtype=np.float32)
    return float(arr.reshape(-1)[0]) if arr.size else float(value)


def _extract_info(info_venv):
    info = info_venv[0] if isinstance(info_venv, (list, tuple)) else info_venv
    result = {
        "finish": 0,
        "collision": 0,
        "off_route": 0,
        "max_time": 0,
        "vehicle_state_dict": {},
    }
    if isinstance(info, (list, tuple)) and len(info) >= 7:
        result.update({
            "finish": int(bool(info[0])),
            "collision": int(bool(info[1])),
            "off_route": int(bool(info[2])),
            "max_time": int(bool(info[3])),
            "vehicle_state_dict": info[6] if isinstance(info[6], dict) else {},
        })
    return result


def _chains_for_storage(chains_venv):
    chains = np.asarray(chains_venv)
    if chains.ndim >= 4:
        return chains[0, :, 0, :]
    if chains.ndim >= 2:
        return chains[0]
    return chains


def collect_hdf5(agent, collect_cfg, timestamp):
    env_name = agent.cfg.env.name
    output_root = os.path.abspath(str(collect_cfg.output_root))
    save_dir, town = build_save_dir(output_root, env_name, timestamp)

    num_episodes = int(collect_cfg.get("num_episodes", 200))
    max_steps = int(collect_cfg.get("max_steps", 260))
    deterministic = bool(collect_cfg.get("deterministic", True))
    save_chains = bool(collect_cfg.get("save_chains", True))
    save_sim = bool(collect_cfg.get("save_sim", True))

    print(f"[INFO] env={env_name}, town={town}")
    print(f"[INFO] saving HDF5 dataset under: {save_dir}")

    agent.model.eval()

    with OfflineDatasetSaver(save_dir=save_dir, timestamp=timestamp, env_name=env_name, town=town) as saver:
        for ep in range(num_episodes):
            obs = agent.reset_env_all(options_venv=[{} for _ in range(agent.n_envs)])
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
            }
            if save_chains:
                episode_data["chains"] = []
            if save_sim:
                episode_data["sim"] = []

            total_reward = 0.0
            terminal_label = {"finish": 0, "collision": 0, "off_route": 0, "max_time": 0}
            success_flag = False

            started_at = time.time()
            for step in range(max_steps):
                with torch.no_grad():
                    cond, obs_np = agent.process_prev_obs(obs)
                    samples = agent.model(
                        cond=cond,
                        deterministic=deterministic,
                        return_chain=save_chains,
                    )
                    output_venv = samples.trajectories.detach().cpu().numpy()
                    action = output_venv[:, : agent.act_steps]
                    chains_venv = (
                        samples.chains.detach().cpu().numpy()
                        if save_chains and hasattr(samples, "chains")
                        else None
                    )

                next_obs, reward, done, info = agent.venv.step(action)
                total_reward += _as_float(reward)

                _, prev_obs = agent.process_prev_obs(obs)
                _, next_obs_np = agent.process_prev_obs(next_obs)

                if save_sim:
                    with torch.no_grad():
                        prev_sim, _ = agent.model.encoder(
                            prev_obs["neighbor_trajs"],
                            mask=None,
                            test=prev_obs,
                            init_state=prev_obs["ego_state"],
                            map_state=prev_obs["neighbor_waypoints"],
                        )
                        next_sim, _ = agent.model.target_encoder(
                            next_obs_np["neighbor_trajs"],
                            mask=None,
                            test=next_obs_np,
                            init_state=next_obs_np["ego_state"],
                            map_state=next_obs_np["neighbor_waypoints"],
                        )
                        sim = 1 - ((F.cosine_similarity(prev_sim, next_sim, dim=-1) + 1) / 2)
                    episode_data["sim"].append(sim.detach().cpu().numpy().tolist())

                info_data = _extract_info(info)
                for vehicle_key, state in info_data["vehicle_state_dict"].items():
                    episode_data.setdefault(vehicle_key, []).append(state)

                if save_chains and chains_venv is not None:
                    episode_data["chains"].append(_chains_for_storage(chains_venv))

                episode_data["neighbor_trajs"].append(prev_obs["neighbor_trajs"])
                episode_data["ego_state"].append(prev_obs["ego_state"])
                episode_data["neighbor_waypoints"].append(prev_obs["neighbor_waypoints"])
                episode_data["action"].append(action)
                episode_data["reward"].append(_as_float(reward))
                episode_data["next_neighbor_trajs"].append(next_obs_np["neighbor_trajs"])
                episode_data["next_ego_state"].append(next_obs_np["ego_state"])
                episode_data["next_neighbor_waypoints"].append(next_obs_np["neighbor_waypoints"])
                episode_data["done"].append(int(_as_bool(done)))

                if _as_bool(done):
                    terminal_label = {
                        "finish": info_data["finish"],
                        "collision": info_data["collision"],
                        "off_route": info_data["off_route"],
                        "max_time": info_data["max_time"],
                    }
                    success_flag = bool(
                        terminal_label["finish"]
                        and not terminal_label["collision"]
                        and not terminal_label["off_route"]
                    )
                    break

                obs = next_obs

            steps = len(episode_data["done"])
            for key, value in terminal_label.items():
                episode_data[key] = [int(value)] * steps
            episode_data["label_info"] = terminal_label
            episode_data["success_flag"] = success_flag

            saver.add_episode(episode_data)
            elapsed = time.time() - started_at
            print(
                f"[EP {ep + 1:03d}/{num_episodes:03d}] "
                f"steps={steps:3d} reward={total_reward:8.3f} "
                f"label={terminal_label} elapsed={elapsed:.1f}s"
            )

    print(f"[INFO] dataset saved: {os.path.join(save_dir, f'offline_dataset_{timestamp}.hdf5')}")
    return save_dir

