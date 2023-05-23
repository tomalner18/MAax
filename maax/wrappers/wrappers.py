from maax.envs.base import Base
from brax.envs.env import Env, State

import jax
from jax import numpy as jp

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Optional,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
)

class MWrapper(Env):
    """Wraps the environment to allow modular transformations."""

    def __init__(self, env: Env):
        super().__init__(config=None)
        self.env = env

        self._metadata: Optional[dict] = None

    def reset(self, rng: jp.ndarray) -> State:
        return self.env.reset(rng)

    def step(self, state: State, action: jp.ndarray) -> State:
        return self.env.step(state, action)

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    @property

    def __getattr__(self, name):
        if name == '__setstate__':
            raise AttributeError(name)
        return getattr(self.env, name)

class RewardWrapper(MWrapper):
    """Superclass of wrappers that can modify the returning reward from a step.

    If you would like to apply a function to the reward that is returned by the base environment before
    passing it to learning code, you can simply inherit from :class:`RewardWrapper` and overwrite the method
    :meth:`reward` to implement that transformation.
    This transformation might change the reward range; to specify the reward range of your wrapper,
    you can simply define :attr:`self.reward_range` in :meth:`__init__`.
    """

    def step(self, state, action):
        """Modifies the reward using :meth:`self.reward` after the environment :meth:`env.step`."""
        dst_state = self.env.step(action)
        reward = self.reward(dst_state.reward)
        return dst_state.replace(reward=reward)

    def reward(self, reward):
        """Returns a modified ``reward``."""
        raise NotImplementedError

class ObservationWrapper(MWrapper):
    """Superclass of wrappers that can modify observations using :meth:`observation` for :meth:`reset` and :meth:`step`.

    If you would like to apply a function to the observation that is returned by the base environment before
    passing it to learning code, you can simply inherit from :class:`ObservationWrapper` and overwrite the method
    :meth:`observation` to implement that transformation. The transformation defined in that method must be
    defined on the base environment’s observation space. However, it may take values in a different space.
    In that case, you need to specify the new observation space of the wrapper by setting :attr:`self.observation_space`
    in the :meth:`__init__` method of your wrapper.

    """

    def reset(self, **kwargs):
        """Resets the environment, returning a modified observation using :meth:`self.observation`."""
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, state, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        dst_state = self.env.step(state, action)
        obs = self.observation(dst_state.obs)
        return dst_state.replace(obs=obs)

    def observation(self, observation):
        """Returns a modified observation."""
        raise NotImplementedError

class ActionWrapper(MWrapper):
    """Superclass of wrappers that can modify the action before :meth:`env.step`.

    If you would like to apply a function to the action before passing it to the base environment,
    you can simply inherit from :class:`ActionWrapper` and overwrite the method :meth:`action` to implement
    that transformation. The transformation defined in that method must take values in the base environment’s
    action space. However, its domain might differ from the original action space.
    In that case, you need to specify the new action space of the wrapper by setting :attr:`self.action_space` in
    the :meth:`__init__` method of your wrapper.
    """

    def step(self, state, action):
        """Runs the environment :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        return self.env.step(state, self.action(action))

    def action(self, action):
        """Returns a modified action before :meth:`env.step` is called."""
        raise NotImplementedError

    def reverse_action(self, action):
        """Returns a reversed ``action``."""
        raise NotImplementedError