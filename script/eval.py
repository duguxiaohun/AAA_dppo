import logging
import math
import os
import subprocess
import sys
import time

import hydra
import pretty_errors
import psutil
from omegaconf import OmegaConf
from datetime import datetime

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

# suppress d4rl import error
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

sys.path.append("../")

# 关键：导入 env 包触发 gym.register，确保 CarlaTown05/10 环境可被 gym.make 识别
import env  # noqa: F401


# DrQ is combined in SAC codes
def kill_carla():
    """杀死 CARLA 进程。"""
    for process in psutil.process_iter(["name"]):
        process_name = process.info.get("name", "")
        if "CarlaUE4" in process_name:
            print("正在杀死 CARLA 进程...")
            process.kill()


# def start_carla(port):
#     """启动CARLA"""
#     print("正在启动CARLA...")
#     # 启动CARLA并指定端口
#     subprocess.Popen(['/home/codon/CARLA/CARLA_0.9.12/CarlaUE4.sh',
#                       '-port={}'.format(port)])
    
def start_carla(port, quality="Epic", offscreen=True):
    """启动 CARLA，并设置端口与画质。"""
    print("正在启动 CARLA...")
    cmd = [
        "/home/codon/CARLA/CARLA_0.9.12/CarlaUE4.sh",
        f"-port={port}",
        f"-quality={quality}",
    ]
    if offscreen:
        cmd.append("-RenderOffScreen")
    subprocess.Popen(cmd)

@hydra.main(
    version_base=None,
    config_path="../cfg/gym/finetune/hopper-v2",
    config_name="ft_ppo_diffusion_mlp",
)
def main(cfg: OmegaConf):
    # 评估脚本强制使用纯评估 agent，避免误触发训练逻辑
    cfg._target_ = cfg.eval.agent_target

    # 评估脚本独立 wandb 命名，避免和训练记录混淆
    if "wandb" in cfg and cfg.wandb is not None:
        cfg.wandb.project = cfg.eval.wandb_project
        cfg.wandb.run = f"{cfg.eval.wandb_run_prefix}_{cfg.env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    kill_carla()
    time.sleep(5)
    start_carla(
        port=int(cfg.eval.carla_port),
        quality=str(cfg.eval.carla_quality),
        offscreen=bool(cfg.eval.carla_offscreen),
    )
    time.sleep(5)

    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)

    if "env" in cfg and "env_type" in cfg.env and cfg.env.env_type == "furniture":
        import furniture_bench

    # run eval agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()

