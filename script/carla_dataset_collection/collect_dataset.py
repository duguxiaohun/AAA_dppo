import datetime
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import hydra
import psutil
import torch
from omegaconf import OmegaConf, open_dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"


def kill_carla():
    current_pid = os.getpid()
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            name = process.info.get("name") or ""
            cmdline = " ".join(process.info.get("cmdline") or [])
            if "CarlaUE4" in name:
                print("正在杀死CARLA进程...")
                process.kill()
            elif (
                process.info.get("pid") != current_pid
                and "collect_dataset.py" in cmdline
                and "python" in name
            ):
                print(f"正在杀死残留采集进程 (pid={process.info['pid']})...")
                process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def start_carla(port, visualize=False, quality=None):
    cmd = ["/home/codon/CARLA/CARLA_0.9.12/CarlaUE4.sh", f"-port={port}"]
    if quality:
        cmd.append(f"-quality={quality}")
    if not visualize:
        cmd.append("-RenderOffScreen")
    print("正在启动CARLA...")
    subprocess.Popen(cmd)


def town_slug_from_env(env_name):
    name = str(env_name).lower()
    if "town03" in name:
        return "town03"
    if "town05" in name:
        return "town05"
    if "town10" in name:
        return "town10"
    return name.replace("carlatown", "town").replace("cross-v0", "").replace("-", "_")


def find_checkpoint_for_env(cfg, collect_cfg):
    explicit_path = str(collect_cfg.get("checkpoint_path", ""))
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()

    checkpoint_dir = str(collect_cfg.get("checkpoint_dir", ""))
    if checkpoint_dir:
        return Path(checkpoint_dir).expanduser().resolve()

    if not bool(collect_cfg.get("auto_checkpoint", True)):
        return None

    root = Path(str(collect_cfg.get("checkpoint_root", ROOT / "script" / "carla_dataset_collection")))
    root = root.expanduser().resolve()
    town = town_slug_from_env(cfg.env.name)
    subdir_template = str(collect_cfg.get("checkpoint_subdir_template", "checkpoint_{town}"))
    ckpt_dir = root / subdir_template.format(town=town)
    if ckpt_dir.exists():
        return ckpt_dir
    return None


def latest_state_file(checkpoint_dir):
    state_file = checkpoint_dir / "state.pt"
    if state_file.exists():
        return state_file

    candidates = sorted(
        checkpoint_dir.glob("state_*.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def load_checkpoint(agent, checkpoint_path, checkpoint_itr=-1):
    if checkpoint_path is None:
        print("[INFO] 未找到自动匹配的 checkpoint，使用 YAML 初始化模型。")
        return

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if checkpoint_path.is_dir() and int(checkpoint_itr) >= 0:
        agent.load(str(checkpoint_path), int(checkpoint_itr))
        print(f"[INFO] 已加载 checkpoint: {checkpoint_path}/state_{int(checkpoint_itr)}.pt")
        return

    ckpt_file = latest_state_file(checkpoint_path) if checkpoint_path.is_dir() else checkpoint_path
    if ckpt_file is None or not ckpt_file.exists():
        print(f"[WARN] checkpoint 不存在或目录中没有 state*.pt: {checkpoint_path}")
        return

    data = torch.load(str(ckpt_file), map_location=agent.device, weights_only=True)
    state_dict = data["model"] if isinstance(data, dict) and "model" in data else data
    agent.model.load_state_dict(state_dict)
    if isinstance(data, dict) and "itr" in data:
        agent.itr = data["itr"]
    print(f"[INFO] 已加载 checkpoint: {ckpt_file}")


@hydra.main(
    version_base=None,
    config_path="../../cfg/gym/finetune/hopper-v2",
    config_name="ft_ppo_diffusion_mlp",
)
def main(cfg):
    OmegaConf.resolve(cfg)

    collect_cfg = cfg.get("dataset_collect", {})
    if collect_cfg.get("agent_target", ""):
        with open_dict(cfg):
            cfg._target_ = collect_cfg.agent_target

    if bool(collect_cfg.get("disable_wandb", True)):
        with open_dict(cfg):
            cfg.wandb = None

    if bool(collect_cfg.get("kill_carla_before_start", True)):
        kill_carla()
        time.sleep(float(collect_cfg.get("kill_wait_seconds", 5)))

    visualize = bool(cfg.get("visualize", False))
    os.environ["CARLA_VISUALIZE"] = str(visualize)
    if bool(collect_cfg.get("start_carla", True)):
        start_carla(
            port=int(collect_cfg.get("carla_port", 2000)),
            visualize=visualize,
            quality=collect_cfg.get("carla_quality", None),
        )
        time.sleep(float(collect_cfg.get("carla_warmup_seconds", 5)))

    from script.carla_dataset_collection.collector import collect_hdf5

    cls = hydra.utils.get_class(cfg._target_)
    from script.carla_dataset_collection.envs import register_local_carla_envs

    register_local_carla_envs()
    agent = cls(cfg)

    checkpoint_path = find_checkpoint_for_env(cfg, collect_cfg)
    load_checkpoint(agent, checkpoint_path, int(collect_cfg.get("checkpoint_itr", -1)))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    collect_hdf5(agent, collect_cfg, timestamp)


if __name__ == "__main__":
    main()
