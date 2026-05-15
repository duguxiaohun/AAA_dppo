from typing import Optional, Union, List
import gym
from gym.vector.utils import batch_space
from gym.logger import warn, deprecation


__all__ = ["VectorEnv"]


class VectorEnv(gym.Env):
    """Base class for vectorized environments (Gym style)."""

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.is_vector_env = True
        self.observation_space = batch_space(observation_space, n=num_envs)
        self.action_space = batch_space(action_space, n=num_envs)

        self.closed = False
        self.viewer = None

        # 单个环境的空间
        self.single_observation_space = observation_space
        self.single_action_space = action_space

    def reset_async(self, seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None):
        pass

    def reset_wait(self, **kwargs):
        raise NotImplementedError()

    def reset(self, *, seed: Optional[Union[int, List[int]]] = None, options: Optional[dict] = None):
        self.reset_async(seed=seed, options=options)
        return self.reset_wait()

    def step_async(self, actions):
        pass

    def step_wait(self):
        raise NotImplementedError()

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def call_async(self, name, *args, **kwargs):
        pass

    def call_wait(self, **kwargs):
        raise NotImplementedError()

    def call(self, name, *args, **kwargs):
        self.call_async(name, *args, **kwargs)
        return self.call_wait()

    def get_attr(self, name):
        return self.call(name)

    def set_attr(self, name, values):
        raise NotImplementedError()

    def close_extras(self, **kwargs):
        pass

    def close(self, **kwargs):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras(**kwargs)
        self.closed = True

    def seed(self, seed=None):
        """Gym 允许直接 seed。"""
        deprecation("env.seed(seed)` is deprecated in Gym >=0.22, "
                    "but still supported here for compatibility.")

    def __del__(self):
        if not getattr(self, "closed", True):
            self.close()

    def __repr__(self):
        if self.spec is None:
            return f"{self.__class__.__name__}({self.num_envs})"
        else:
            return f"{self.__class__.__name__}({self.spec.id}, {self.num_envs})"
