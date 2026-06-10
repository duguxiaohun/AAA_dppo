"""
new_eval.py：与训练步骤动作一致的评估脚本（无 critic 引导）。

配置与 eval.py 相同，区别：
  - 底层 Agent 换成 NewEvalPPODiffusionAgent
  - 动作生成：model(deterministic=True) 直接 step CARLA，不经过 update_target_action / forward_again

运行示例：
  PYTHONPATH=$(pwd) python script/new_eval.py
  PYTHONPATH=$(pwd) python script/new_eval.py eval.checkpoint_step=540
  PYTHONPATH=$(pwd) python script/new_eval.py carla_env_interface=CarlaTown10Cross-v0
"""

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

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

log = logging.getLogger(__name__)

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

sys.path.append("../")

import env  # noqa: F401


def kill_carla():
    for process in psutil.process_iter(["name"]):
        process_name = process.info.get("name", "")
        if "CarlaUE4" in process_name:
            print("正在杀死 CARLA 进程...")
            process.kill()


def start_carla(port, quality="Epic", offscreen=True):
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
    config_name="new_eval_ppo_diffusion_carla",
)
def main(cfg: OmegaConf):
    kill_carla()
    time.sleep(5)

    visualize = bool(cfg.get("visualize", False))
    os.environ["CARLA_VISUALIZE"] = str(visualize)
    start_carla(
        port=int(cfg.eval.carla_port),
        quality=str(cfg.eval.carla_quality),
        offscreen=not visualize,
    )
    time.sleep(5)

    OmegaConf.resolve(cfg)

    if "env" in cfg and "env_type" in cfg.env and cfg.env.env_type == "furniture":
        import furniture_bench

    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.run()


if __name__ == "__main__":
    main()
