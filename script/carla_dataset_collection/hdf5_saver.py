import json
import os

import h5py
import numpy as np


def create_vlen_dataset(group, name, values, dtype=np.float32, compression=None):
    """Write a list of scalar/vector/tensor values as a variable-length dataset."""
    vlen_dtype = h5py.vlen_dtype(np.dtype(dtype))
    data = np.empty(len(values), dtype=object)

    for i, value in enumerate(values):
        arr = np.asarray(value, dtype=dtype)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim > 1:
            arr = arr.reshape(-1)
        data[i] = arr

    return group.create_dataset(name, data=data, dtype=vlen_dtype, compression=compression)


class OfflineDatasetSaver:
    """Save collected CARLA episodes to one HDF5 file plus a matching meta JSON."""

    def __init__(self, save_dir, timestamp, env_name, town):
        os.makedirs(save_dir, exist_ok=True)
        self.file_path = os.path.join(save_dir, f"offline_dataset_{timestamp}.hdf5")
        self.meta_path = os.path.join(save_dir, f"meta_{timestamp}.json")
        self.h5 = h5py.File(self.file_path, "a")
        if "episodes" not in self.h5:
            self.h5.create_group("episodes")
        self.episode_count = len(self.h5["episodes"])
        self.meta = {
            "env_name": env_name,
            "town": town,
            "created_time": timestamp,
            "hdf5_file": self.file_path,
            "episodes": [],
        }

    def add_episode(self, episode_data):
        ep_name = f"{self.episode_count:05d}"
        group = self.h5["episodes"].create_group(ep_name)

        try:
            if "reward" in episode_data:
                group.create_dataset(
                    "reward",
                    data=np.asarray(episode_data["reward"], dtype=np.float32),
                    compression="gzip",
                )

            vehicle_keys = [
                key for key in episode_data
                if key.startswith("ego") or key.startswith("v_")
            ]
            for key in [
                "neighbor_waypoints",
                "next_ego_state",
                "next_neighbor_waypoints",
                "action",
                "sim",
            ]:
                if key in episode_data:
                    create_vlen_dataset(group, key, episode_data[key], dtype=np.float32)

            for key in vehicle_keys:
                if key in episode_data:
                    create_vlen_dataset(group, key, episode_data[key], dtype=np.float32)

            for key in ["neighbor_trajs", "next_neighbor_trajs", "chains"]:
                if key in episode_data:
                    create_vlen_dataset(group, key, episode_data[key], dtype=np.float32)

            for key in ["finish", "collision", "off_route", "max_time"]:
                if key in episode_data:
                    group.create_dataset(key, data=np.asarray(episode_data[key], dtype=np.int32))

            if "done" in episode_data:
                group.create_dataset("done", data=np.asarray(episode_data["done"], dtype=np.int8))

            if "success_flag" in episode_data:
                group.create_dataset(
                    "success_flag",
                    data=np.asarray([int(bool(episode_data["success_flag"]))], dtype=np.int8),
                )

            if "label_info" in episode_data:
                dt = h5py.string_dtype(encoding="utf-8")
                group.create_dataset(
                    "label_info",
                    data=json.dumps(episode_data["label_info"], ensure_ascii=False),
                    dtype=dt,
                )

            rewards = np.asarray(episode_data.get("reward", []), dtype=np.float32)
            label_info = episode_data.get(
                "label_info",
                {"finish": 0, "collision": 0, "off_route": 0, "max_time": 0},
            )
            self.meta["episodes"].append({
                "id": self.episode_count,
                "length": int(len(episode_data.get("done", []))),
                "reward_sum": float(rewards.sum()) if rewards.size else 0.0,
                "success": bool(episode_data.get("success_flag", False)),
                "label_info": label_info,
            })
            self.episode_count += 1
            self.h5.flush()
        except Exception:
            if ep_name in self.h5["episodes"]:
                del self.h5["episodes"][ep_name]
                self.h5.flush()
            raise

    def close(self):
        if self.h5:
            self.h5.attrs["num_episodes"] = self.episode_count
            self.h5.attrs["env_name"] = self.meta["env_name"]
            self.h5.attrs["town"] = self.meta["town"]
            self.h5.close()
            self.h5 = None

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

