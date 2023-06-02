import gym
from gym.spaces import Dict, Box
import numpy as np
from copy import deepcopy
import logging

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


class MWrapper(Base):
    """Wraps the environment to allow modular transformations."""

    def __init__(self, env: Env):
        self.env = env
        self._metadata: Optional[dict] = None

    def reset(self, rng: jp.ndarray) -> State:
        return self.env.reset(rng)

    def step(self, state: State, action: jp.ndarray) -> State:
        return self.env.step(state, action)

    def gen_sys(self, seed):
        self.env.gen_sys(seed)

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
    def backend(self) -> str:
        return self.unwrapped.backend

    @property
    def metadata(self) -> dict:
        """Returns the environment metadata."""
        if self._metadata is None:
            return self.env.metadata
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

    @classmethod
    def class_name(cls):
        """Returns the class name of the wrapper."""
        return cls.__name__

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

    def reset(self, rng):
        """Resets the environment, returning a state with modified observation using :meth:`self.observation`."""
        state = self.env.reset(rng)
        return state.replace(obs=self.observation(state.obs))

    def step(self, state, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        dst_state = self.env.step(state, action)
        obs = self.observation(state.obs)
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


class JPArrayRewardWrapper(RewardWrapper):
    """
        Convenience wrapper that casts rewards to the multiagent format
        (jax-numpy array of shape (n_agents,))
    """
    def __init__(self, env):
        super().__init__(env)

    def reward(self, rew):
        return jp.zeros((self.unwrapped.n_agents,)) + rew


class MaskActionWrapper(MWrapper):
    '''
        For a boolean action, sets it to zero given a mask from the previous step.
            For example you could mask the grab action based on whether you can see the box
        Args:
            action_key (string): key in action dictionary to be masked
            mask_keys (string): keys in observation dictionary with which to mask. The shape
                of the concatenation of the masks (along the 1st dimension) should exactly
                match that of action_key
    '''
    def __init__(self, env, action_key, mask_keys):
        super().__init__(env)
        self.action_key = action_key
        self.mask_keys = mask_keys

    def reset(self, rng):
        state = self.env.reset(rng)
        self.prev_obs = deepcopy(state.obs)
        return state

    def step(self, state, action):
        mask = jp.concatenate([self.prev_obs[k] for k in self.mask_keys], -1)
        action[self.action_key] = jp.logical_and(action[self.action_key], mask)
        dst_state = self.env.step(state, action)
        dst_state = dst_state.replace(obs=deepcopy(self.prev_obs))
        return dst_state


class AddConstantObservationsWrapper(ObservationWrapper):
    '''
        Adds new constant observations to the environment.
        Args:
            new_obs: Dictionary with the new observations.
    '''
    def __init__(self, env, new_obs):
        super().__init__(env)
        self.new_obs = new_obs
        for obs_key in self.new_obs:
            if type(self.new_obs[obs_key]) in [list, tuple]:
                self.new_obs[obs_key] = jp.array(self.new_obs[obs_key])
            shape = self.new_obs[obs_key].shape
            # self.observation_space = update_obs_space(self, {obs_key: shape})

    def observation(self, obs):
        for key, val in self.new_obs.items():
            obs[key] = val
        return obs


class SpoofEntityWrapper(ObservationWrapper):
    '''
        Add extra entities along entity dimension such that shapes can match between
            environments with differing number of entities. This is meant to be used
            after SplitObservations and SelectKeysWrapper. This will also add masks that are
            1 except along the new columns (which could be used by fully observed value function)
        Args:
            total_n_entities (int): total number of entities after spoofing (including spoofed ones)
            keys (list): observation keys with which to add entities along the second dimension
            mask_keys (list): mask keys with which to add columns.
    '''
    def __init__(self, env, total_n_entities, keys, mask_keys):
        super().__init__(env)
        self.total_n_entities = total_n_entities
        self.keys = keys
        self.mask_keys = mask_keys
        # for key in self.keys + self.mask_keys:
        #     shape = list(self.observation_space.spaces[key].shape)
        #     shape[1] = total_n_entities
        #     self.observation_space = update_obs_space(self, {key: shape})
        # for key in self.mask_keys:
        #     shape = list(self.observation_space.spaces[key].shape)
        #     self.observation_space = update_obs_space(self, {key + '_spoof': shape})

    def observation(self, obs):
        for key in self.keys:
            n_to_spoof = self.total_n_entities - obs[key].shape[1]
            if n_to_spoof > 0:
                obs[key] = jp.concatenate([obs[key], jp.zeros((obs[key].shape[0], n_to_spoof, obs[key].shape[-1]))], 1)
        for key in self.mask_keys:
            n_to_spoof = self.total_n_entities - obs[key].shape[1]
            obs[key + '_spoof'] = jp.concatenate([jp.ones_like(obs[key]), jp.zeros((obs[key].shape[0], n_to_spoof))], -1)
            if n_to_spoof > 0:
                obs[key] = jp.concatenate([obs[key], jp.zeros((obs[key].shape[0], n_to_spoof))], -1)

        return obs


class ConcatenateObsWrapper(ObservationWrapper):
    '''
        Group multiple observations under the same key in the observation dictionary.
        Args:
            obs_groups: dict of {key_to_save: [keys to concat]}
    '''
    def __init__(self, env, obs_groups):
        super().__init__(env)
        self.obs_groups = obs_groups
    def observation(self, obs):
        for key_to_save, keys_to_concat in self.obs_groups.items():
            obs[key_to_save] = jp.concatenate([obs[k] for k in keys_to_concat], -1)
        return obs
