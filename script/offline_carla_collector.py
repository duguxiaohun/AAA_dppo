import os
import sys
import pretty_errors
import logging

import math
import hydra
from omegaconf import OmegaConf
import gdown
from download_url import (
    get_dataset_download_url,
    get_normalization_download_url,
    get_checkpoint_download_url,
)
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)
# suppress d4rl import error
os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"

# add logger
log = logging.getLogger(__name__)

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
import sys

sys.path.append('../')
import time
import torch
import numpy as np
import psutil
import subprocess
from configs.init_configs import get_argument, set_configs


from copy import deepcopy
from glob import glob
import gym


# DrQ is combined in SAC codes
def kill_carla():
    """杀死CARLA进程"""
    for process in psutil.process_iter(['name']):
        if 'CarlaUE4' in process.info['name']:
            print("正在杀死CARLA进程...")
            process.kill()


# def start_carla(port):
#     """启动CARLA"""
#     print("正在启动CARLA...")
#     # 启动CARLA并指定端口
#     subprocess.Popen(['/home/codon/CARLA/CARLA_0.9.12/CarlaUE4.sh',
#                       '-port={}'.format(port)])
    
def start_carla(port, quality="Epic"):
    """启动CARLA，并设置图形质量"""
    print("正在启动CARLA...")
    # 启动CARLA并指定端口和图形质量
    subprocess.Popen(['/home/codon/CARLA/CARLA_0.9.12/CarlaUE4.sh',
                      '-port={}'.format(port),
                      '-quality={}'.format(quality)])  # 设置图形质量为Epic

@hydra.main(
    version_base=None,
    config_path="../cfg/gym/finetune/hopper-v2",
    config_name="ft_ppo_diffusion_mlp_eval",
)
def main(cfg: OmegaConf):
    kill_carla()
    time.sleep(5)
    start_carla(port=2000)
    time.sleep(5)
    # resolve immediately so all the ${now:} resolvers will use the same time.
    OmegaConf.resolve(cfg)

    # For pre-training: download dataset if needed
    # if "train_dataset_path" in cfg and not os.path.exists(cfg.train_dataset_path):
    #     download_url = get_dataset_download_url(cfg)
    #     download_target = os.path.dirname(cfg.train_dataset_path)
    #     log.info(f"Downloading dataset from {download_url} to {download_target}")
    #     gdown.download_folder(url=download_url, output=download_target)

    # For for-tuning: download normalization if needed
    # if "normalization_path" in cfg and not os.path.exists(cfg.normalization_path):
    #     download_url = get_normalization_download_url(cfg)
    #     download_target = cfg.normalization_path
    #     dir_name = os.path.dirname(download_target)
    #     if not os.path.exists(dir_name):
    #         os.makedirs(dir_name)
    #     log.info(
    #         f"Downloading normalization statistics from {download_url} to {download_target}"
    #     )
    #     gdown.download(url=download_url, output=download_target, fuzzy=True)

    # For for-tuning: download checkpoint if needed
    # if "base_policy_path" in cfg and not os.path.exists(cfg.base_policy_path):
    #     download_url = get_checkpoint_download_url(cfg)
    #     if download_url is None:
    #         raise ValueError(
    #             f"Unknown checkpoint path. Did you specify the correct path to the policy you trained?"
    #         )
    #     download_target = cfg.base_policy_path
    #     dir_name = os.path.dirname(download_target)
    #     if not os.path.exists(dir_name):
    #         os.makedirs(dir_name)
    #     log.info(f"Downloading checkpoint from {download_url} to {download_target}")
    #     gdown.download(url=download_url, output=download_target, fuzzy=True)

    # Deal with isaacgym needs to be imported before torch
    if "env" in cfg and "env_type" in cfg.env and cfg.env.env_type == "furniture":
        import furniture_bench

    # run agent
    cls = hydra.utils.get_class(cfg._target_)
    agent = cls(cfg)
    agent.collect()


if __name__ == "__main__":
    main()
# export https_proxy=http://127.0.0.1:7890
# PYTHONPATH=$(pwd) python script/run.py  --config-dir=cfg/gym/finetune/hopper-v2   --config-name=ft_ppo_diffusion_mlp

# export PYTHONPATH="$PYTHONPATH:/home/codon/github/第二篇"
# source ~/.bashrc  # 或 source ~/.zshrc
# PYTHONPATH=$(pwd) python -c "from model.diffusion.diffusion_ppo import PPODiffusion"
