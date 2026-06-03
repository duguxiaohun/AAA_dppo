# env/gym_utils/__init__.py

import os
import json
import numpy as np

try:
    from collections.abc import Iterable
except ImportError:
    Iterable = (tuple, list)


def make_async(
    id,
    num_envs=1,
    asynchronous=True,
    wrappers=None,
    render=False,
    obs_dim=23,
    action_dim=7,
    env_type=None,
    max_episode_steps=None,
    # furniture only
    gpu_id=0,
    headless=True,
    record=False,
    normalization_path=None,
    furniture="one_leg",
    randomness="low",
    obs_steps=1,
    act_steps=8,
    sparse_reward=False,
    # robomimic only
    robomimic_env_cfg_path=None,
    use_image_obs=False,
    render_offscreen=False,
    reward_shaping=False,
    shape_meta=None,
    **kwargs,
):
    """
    Create a (vectorized) environment.

    - If num_envs == 1: returns a _SingleEnvAdapter that wraps a single gym env (recommended for CARLA).
    - Else: returns AsyncVectorEnv / SyncVectorEnv with multiple envs.
    """

    # ======================
    # furniture 专用分支（保留原来功能）
    # ======================
    if env_type == "furniture":
        from furniture_bench.envs.observation import DEFAULT_STATE_OBS
        from furniture_bench.envs.furniture_rl_sim_env import FurnitureRLSimEnv
        from env.gym_utils.wrapper.furniture import FurnitureRLSimEnvMultiStepWrapper

        env = FurnitureRLSimEnv(
            act_rot_repr="rot_6d",
            action_type="pos",
            april_tags=False,
            concat_robot_state=True,
            ctrl_mode="diffik",
            obs_keys=DEFAULT_STATE_OBS,
            furniture=furniture,
            gpu_id=gpu_id,
            headless=headless,
            num_envs=num_envs,
            observation_space="state",
            randomness=randomness,
            max_env_steps=max_episode_steps,
            record=record,
            pos_scalar=1,
            rot_scalar=1,
            stiffness=1_000,
            damping=200,
        )
        env = FurnitureRLSimEnvMultiStepWrapper(
            env,
            n_obs_steps=obs_steps,
            n_action_steps=act_steps,
            prev_action=False,
            reset_within_step=False,
            pass_full_observations=False,
            normalization_path=normalization_path,
            sparse_reward=sparse_reward,
        )
        return env

    # ======================
    # 统一使用 gym（不是 gymnasium）
    # ======================
    import gym
    from gym import spaces
    # 你的项目内的向量环境实现
    from env.gym_utils.async_vector_env import AsyncVectorEnv
    from env.gym_utils.sync_vector_env import SyncVectorEnv
    from env.gym_utils.wrapper import wrapper_dict

    # 可能会用到的 env 包（按需导入）
    if robomimic_env_cfg_path is not None:
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
    elif "avoiding" in id:
        import gym_avoiding
    else:
        # d4rl 不是必须，有就导入；没有也别影响 CARLA
        try:
            import d4rl.gym_mujoco  # noqa: F401
        except Exception:
            pass

    # ======================
    # 单环境直通（强烈推荐 CARLA 用）
    # ======================
    if int(num_envs) == 1:
        env = _make_one_env(
            id=id,
            wrappers=wrappers,
            render=render,
            # robomimic 相关
            robomimic_env_cfg_path=robomimic_env_cfg_path,
            use_image_obs=use_image_obs,
            render_offscreen=render_offscreen,
            reward_shaping=reward_shaping,
            # dummy meta
            obs_dim=obs_dim,
            action_dim=action_dim,
            shape_meta=shape_meta,
            **kwargs,
        )
        return _SingleEnvAdapter(env)

    # ======================
    # 多环境（慎用于 CARLA）
    # ======================
    def _make_env():
        # robomimic
        if robomimic_env_cfg_path is not None:
            obs_modality_dict = {
                "low_dim": (
                    wrappers.robomimic_image.low_dim_keys
                    if wrappers and ("robomimic_image" in wrappers)
                    else wrappers.robomimic_lowdim.low_dim_keys
                ),
                "rgb": (
                    wrappers.robomimic_image.image_keys
                    if wrappers and ("robomimic_image" in wrappers)
                    else None
                ),
            }
            if obs_modality_dict.get("rgb") is None:
                obs_modality_dict.pop("rgb", None)
            ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
            if render_offscreen or use_image_obs:
                os.environ["MUJOCO_GL"] = "egl"
            with open(robomimic_env_cfg_path, "r") as f:
                env_meta = json.load(f)
            env_meta["reward_shaping"] = reward_shaping
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=render,
                render_offscreen=render_offscreen,
                use_image_obs=use_image_obs,
            )
            # robosuite memory workaround
            try:
                env.env.hard_reset = False
            except Exception:
                pass
        else:
            # 普通 Gym / D4RL / 你的自定义（如 CARLA）
            env = gym.make(id, **kwargs)

        # 加 wrappers（如果配置了）
        if wrappers is not None:
            for wrapper, args in wrappers.items():
                env = wrapper_dict[wrapper](env, **args)
        return env

    # 一些项目里的 dummy_env 用于先拿到 space / metadata，这里保留一个合理的实现
    def dummy_env_fn():
        from env.gym_utils.wrapper.multi_step import MultiStep
        env = gym.Env()
        observation_space = spaces.Dict()

        if shape_meta is not None:
            for key, value in shape_meta["obs"].items():
                shape = value["shape"]
                if key.endswith("rgb"):
                    low, high = -1.0, 1.0
                elif key.endswith("state"):
                    low, high = -1.0, 1.0
                else:
                    raise RuntimeError(f"Unsupported obs key {key}")
                observation_space[key] = spaces.Box(
                    low=low, high=high, shape=shape, dtype=np.float32
                )
        else:
            observation_space["state"] = spaces.Box(
                low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
            )

        env.observation_space = observation_space
        env.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        env.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 12,
        }
        # MultiStep 可能要 n_obs_steps，从 wrappers 里读取（若无则默认 1）
        n_obs_steps = 1
        if wrappers and ("multi_step" in wrappers):
            n_obs_steps = wrappers["multi_step"].get("n_obs_steps", 1)
        return MultiStep(env=env, n_obs_steps=n_obs_steps)

    env_fns = [_make_env for _ in range(int(num_envs))]

    if asynchronous:
        # 兼容不同项目里 AsyncVectorEnv 的签名差异
        try:
            # 新实现：支持 dummy_env_fn / delay_init
            return AsyncVectorEnv(
                env_fns,
                dummy_env_fn=dummy_env_fn if (render or render_offscreen or use_image_obs) else None,
                delay_init=("avoiding" in id),
            )
        except TypeError:
            # 老实现：不支持这些参数
            return AsyncVectorEnv(env_fns)
    else:
        return SyncVectorEnv(env_fns)


# =========================
# 工具函数 & 单环境适配器
# =========================

def _make_one_env(
    id,
    wrappers=None,
    render=False,
    robomimic_env_cfg_path=None,
    use_image_obs=False,
    render_offscreen=False,
    reward_shaping=False,
    obs_dim=23,
    action_dim=7,
    shape_meta=None,
    **kwargs,
):
    """只创建一个 gym 环境实例，并按需套 wrappers。"""
    import gym
    if robomimic_env_cfg_path is not None:
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
        obs_modality_dict = {
            "low_dim": (
                wrappers.robomimic_image.low_dim_keys
                if wrappers and ("robomimic_image" in wrappers)
                else wrappers.robomimic_lowdim.low_dim_keys
            ),
            "rgb": (
                wrappers.robomimic_image.image_keys
                if wrappers and ("robomimic_image" in wrappers)
                else None
            ),
        }
        if obs_modality_dict.get("rgb") is None:
            obs_modality_dict.pop("rgb", None)
        ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)
        if render_offscreen or use_image_obs:
            os.environ["MUJOCO_GL"] = "egl"
        with open(robomimic_env_cfg_path, "r") as f:
            env_meta = json.load(f)
        env_meta["reward_shaping"] = reward_shaping
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=render,
            render_offscreen=render_offscreen,
            use_image_obs=use_image_obs,
        )
        try:
            env.env.hard_reset = False
        except Exception:
            pass
    else:
        env = gym.make(id, **kwargs)

    if wrappers is not None:
        from env.gym_utils.wrapper import wrapper_dict
        for wrapper, args in wrappers.items():
            env = wrapper_dict[wrapper](env, **args)
    return env


class _SingleEnvAdapter:
    """
    把单个 gym.Env 伪装成“向量环境(batch=1)”以兼容训练管线。
    适配的是老 Gym（reset->obs, step->(obs,reward,done,info)）接口。
    如果你的 Carla 环境是 gymnasium 风格，告诉我，我给你替换成相应版本。
    """
    def __init__(self, env):
        import gym
        self.env = env
        self.num_envs = 1
        self.is_vector_env = True
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.metadata = getattr(env, "metadata", {})

    # 训练器常用
    def reset_arg(self, options_list=None, **kwargs):
        obs = self.reset()
        if isinstance(obs, np.ndarray):
            return np.expand_dims(obs, 0)
        else:
            return [obs]

    def reset_one_arg(self, env_ind, options=None):
        assert env_ind == 0
        return self.reset()

    def reset(self, **kwargs):
        # 老 Gym：reset() -> obs
        return self.env.reset()

    def step(self, actions):
        # actions 可能是 batch=[1, ...]
        if isinstance(actions, (list, tuple)) or getattr(actions, "ndim", 0) > 0:
            action = actions[0]
        else:
            action = actions
        obs, reward, done, info = self.env.step(action)

        obs_b = np.expand_dims(obs, 0) if isinstance(obs, np.ndarray) else [obs]
        rew_b = np.asarray([reward], dtype=np.float32)
        done_b = np.asarray([done], dtype=bool)
        info_b = [info]
        return obs_b, rew_b, done_b, info_b

    def seed(self, seeds=None):
        """
        尽量使用 reset(seed=...) 以避开某些环境把 `seed` 当属性造成的冲突。
        失败时再尝试 env.seed(...)（仅当其为可调用），否则尝试设置属性。
        """
        # 取单个种子值
        if isinstance(seeds, int) or seeds is None:
            s = seeds if isinstance(seeds, int) else None
        elif isinstance(seeds, (list, tuple)) and len(seeds) > 0:
            s = seeds[0]
        else:
            s = None

        # 优先：用 reset(seed=...)（Gym>=0.21 支持）
        try:
            # 有些环境的 reset 不接受 seed；用 TypeError 捕获签名不匹配
            self.env.reset(seed=s)
            return [s]
        except TypeError:
            pass
        except Exception:
            # 其它异常也继续尝试 fallback
            pass

        # 其次：如果 env.seed 是“可调用”的再调用（避免 int 属性被调用）
        if hasattr(self.env, "seed") and callable(getattr(self.env, "seed")):
            try:
                return self.env.seed(s)
            except Exception:
                pass

        # 最后：如果 env.seed 是个属性（非可调用），尝试赋值
        if hasattr(self.env, "seed") and not callable(getattr(self.env, "seed")):
            try:
                setattr(self.env, "seed", s)
            except Exception:
                pass

        # 额外：常见随机库的播种（可选）
        try:
            import random;
            random.seed(s)
        except Exception:
            pass
        try:
            import numpy as np;
            np.random.seed(s if s is not None else 0)
        except Exception:
            pass

        return [s]

    def render(self, *args, **kwargs):
        if hasattr(self.env, "render"):
            return self.env.render(*args, **kwargs)

    def close(self, **kwargs):
        if hasattr(self.env, "close"):
            self.env.close()
