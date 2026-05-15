import numpy as np
import multiprocessing as mp
import sys
import time
from enum import Enum
from copy import deepcopy
import gym
from gym import logger
from gym.error import AlreadyPendingCallError, NoAsyncCallError, ClosedEnvironmentError, CustomSpaceError
from gym.vector.utils import (
    create_shared_memory,
    create_empty_array,
    write_to_shared_memory,
    read_from_shared_memory,
    concatenate,
    iterate,
    CloudpickleWrapper,
    clear_mpi_env_vars,
)
from env.gym_utils.vector_env import VectorEnv


class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class AsyncVectorEnv(VectorEnv):
    """Vectorized environment that runs multiple environments in parallel (Gym style)."""

    def __init__(
        self,
        env_fns,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
    ):
        ctx = mp.get_context(context)
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy

        dummy_env = env_fns[0]()
        self.metadata = dummy_env.metadata
        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or dummy_env.observation_space
            action_space = action_space or dummy_env.action_space
        dummy_env.close()

        super().__init__(num_envs=len(env_fns),
                         observation_space=observation_space,
                         action_space=action_space)

        if self.shared_memory:
            try:
                _obs_buffer = create_shared_memory(
                    self.single_observation_space, n=self.num_envs, ctx=ctx
                )
                self.observations = read_from_shared_memory(
                    self.single_observation_space, _obs_buffer, n=self.num_envs
                )
            except CustomSpaceError:
                raise ValueError("`shared_memory=True` only works with Box/Dict/Tuple spaces.")
        else:
            _obs_buffer = None
            self.observations = create_empty_array(
                self.single_observation_space, n=self.num_envs, fn=np.zeros
            )

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = _worker_shared_memory if self.shared_memory else _worker

        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name=f"Worker-{idx}",
                    args=(idx, CloudpickleWrapper(env_fn),
                          child_pipe, parent_pipe, _obs_buffer, self.error_queue),
                )
                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)
                process.daemon = daemon
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT

    def reset_async(self, seed=None, options=None):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError("reset_async called but state not DEFAULT", self._state.value)

        for pipe in self.parent_pipes:
            kwargs = {}
            if seed is not None:
                kwargs["seed"] = seed
            if options is not None:
                kwargs["options"] = options
            pipe.send(("reset", kwargs))
        self._state = AsyncState.WAITING_RESET

    def reset_wait(self, timeout=None):
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError("reset_wait called without reset_async", AsyncState.WAITING_RESET.value)

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError("reset_wait timed out")

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        if not self.shared_memory:
            self.observations = concatenate(self.single_observation_space, results, self.observations)

        return deepcopy(self.observations) if self.copy else self.observations

    def step_async(self, actions):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError("step_async called but not DEFAULT", self._state.value)

        actions = iterate(self.action_space, actions)
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))
        self._state = AsyncState.WAITING_STEP

    def step_wait(self, timeout=None):
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError("step_wait called without step_async", AsyncState.WAITING_STEP.value)

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError("step_wait timed out")

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        observations_list, rewards, dones, infos = zip(*results)

        if not self.shared_memory:
            self.observations = concatenate(
                self.single_observation_space, observations_list, self.observations
            )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.array(rewards),
            np.array(dones, dtype=np.bool_),
            infos,
        )

    def close_extras(self, timeout=None, terminate=False):
        for pipe in self.parent_pipes:
            if pipe is not None and not pipe.closed:
                pipe.send(("close", None))
                pipe.recv()
        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def _poll(self, timeout=None):
        if timeout is None:
            return True
        end_time = time.perf_counter() + timeout
        for pipe in self.parent_pipes:
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(max(end_time - time.perf_counter(), 0))):
                return False
        return True

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError("Trying to use AsyncVectorEnv after close().")

    def _raise_if_errors(self, successes):
        if all(successes):
            return
        index, exctype, value = self.error_queue.get()
        logger.error(f"Worker-{index} crashed: {exctype.__name__}: {value}")
        raise exctype(value)


def _worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == "reset":
                obs = env.reset(**data)
                pipe.send((obs, True))
            elif cmd == "step":
                obs, rew, done, info = env.step(data)
                pipe.send(((obs, rew, done, info), True))
            elif cmd == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif cmd == "close":
                pipe.send((None, True))
                break
            else:
                raise RuntimeError(f"Unknown command {cmd}")
    except Exception:
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    env = env_fn()
    obs_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == "reset":
                obs = env.reset(**data)
                write_to_shared_memory(obs_space, index, obs, shared_memory)
                pipe.send((None, True))
            elif cmd == "step":
                obs, rew, done, info = env.step(data)
                write_to_shared_memory(obs_space, index, obs, shared_memory)
                pipe.send(((None, rew, done, info), True))
            elif cmd == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif cmd == "close":
                pipe.send((None, True))
                break
            else:
                raise RuntimeError(f"Unknown command {cmd}")
    except Exception:
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
