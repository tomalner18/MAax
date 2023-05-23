import gym
# from mujoco_py import MujocoException
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


def update_obs_space(env, delta):
    spaces = env.observation_space.spaces.copy()
    for key, shape in delta.items():
        spaces[key] = Box(-np.inf, np.inf, shape, np.float32)
    return Dict(spaces)

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
    def backend(self) -> str:
        return self.unwrapped.backend

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
        """Resets the environment, returning a state with modified observation using :meth:`self.observation`."""
        state = self.env.reset(**kwargs)
        return state.replace(obs=self.observation(state.obs))

    def step(self, state, action):
        """Returns a modified observation using :meth:`self.observation` after calling :meth:`env.step`."""
        dst_state = self.env.step(state, action)
        obs = self.observation(state)
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


class NumpyArrayRewardWrapper(gym.RewardWrapper):
    """
        Convenience wrapper that casts rewards to the multiagent format
        (numpy array of shape (n_agents,))
    """
    def __init__(self, env):
        super().__init__(env)

    def reward(self, rew):
        return np.zeros((self.unwrapped.n_agents,)) + rew


class DiscretizeActionWrapper(gym.ActionWrapper):
    '''
        Take a Box action and convert it to a MultiDiscrete Action through quantization
        Args:
            action_key: (string) action to discretize
            nbuckets: (int) number of discrete actions per dimension. It should be odd such
                        that actions centered around 0 will have the middle action be 0.
    '''
    def __init__(self, env, action_key, nbuckets=11):
        super().__init__(env)
        self.action_key = action_key
        self.discrete_to_continuous_act_map = []
        for i, ac_space in enumerate(self.action_space.spaces[action_key].spaces):
            assert isinstance(ac_space, Box)
            action_map = np.array([np.linspace(low, high, nbuckets)
                                   for low, high in zip(ac_space.low, ac_space.high)])
            _nbuckets = np.ones((len(action_map))) * nbuckets
            self.action_space.spaces[action_key].spaces[i] = gym.spaces.MultiDiscrete(_nbuckets)
            self.discrete_to_continuous_act_map.append(action_map)
        self.discrete_to_continuous_act_map = np.array(self.discrete_to_continuous_act_map)

    def action(self, state, action):
        action = deepcopy(action)
        ac = action[self.action_key]

        # helper variables for indexing the discrete-to-continuous action map
        agent_idxs = np.tile(np.arange(ac.shape[0])[:, None], ac.shape[1])
        ac_idxs = np.tile(np.arange(ac.shape[1]), ac.shape[0]).reshape(ac.shape)

        action[self.action_key] = self.discrete_to_continuous_act_map[agent_idxs, ac_idxs, ac]
        return action


class MaskActionWrapper(gym.Wrapper):
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

    def reset(self):
        self.prev_obs = self.env.reset()
        return deepcopy(self.prev_obs)

    def step(self, action):
        mask = np.concatenate([self.prev_obs[k] for k in self.mask_keys], -1)
        action[self.action_key] = np.logical_and(action[self.action_key], mask)
        self.prev_obs, rew, done, info = self.env.step(action)
        return deepcopy(self.prev_obs), rew, done, info


class AddConstantObservationsWrapper(gym.ObservationWrapper):
    '''
        Adds new constant observations to the environment.
        Args:
            new_obs: Dictionary with the new observations.
    '''
    def __init__(self, env, new_obs):
        super().__init__(env)
        self.new_obs = new_obs
        for obs_key in self.new_obs:
            assert obs_key not in self.observation_space.spaces, (
                f'Observation key {obs_key} exists in original observation space')
            if type(self.new_obs[obs_key]) in [list, tuple]:
                self.new_obs[obs_key] = np.array(self.new_obs[obs_key])
            shape = self.new_obs[obs_key].shape
            self.observation_space = update_obs_space(self, {obs_key: shape})

    def observation(self, obs):
        for key, val in self.new_obs.items():
            obs[key] = val
        return obs


class SpoofEntityWrapper(gym.ObservationWrapper):
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
        for key in self.keys + self.mask_keys:
            shape = list(self.observation_space.spaces[key].shape)
            shape[1] = total_n_entities
            self.observation_space = update_obs_space(self, {key: shape})
        for key in self.mask_keys:
            shape = list(self.observation_space.spaces[key].shape)
            self.observation_space = update_obs_space(self, {key + '_spoof': shape})

    def observation(self, obs):
        for key in self.keys:
            n_to_spoof = self.total_n_entities - obs[key].shape[1]
            if n_to_spoof > 0:
                obs[key] = np.concatenate([obs[key], np.zeros((obs[key].shape[0], n_to_spoof, obs[key].shape[-1]))], 1)
        for key in self.mask_keys:
            n_to_spoof = self.total_n_entities - obs[key].shape[1]
            obs[key + '_spoof'] = np.concatenate([np.ones_like(obs[key]), np.zeros((obs[key].shape[0], n_to_spoof))], -1)
            if n_to_spoof > 0:
                obs[key] = np.concatenate([obs[key], np.zeros((obs[key].shape[0], n_to_spoof))], -1)

        return obs


class ConcatenateObsWrapper(gym.ObservationWrapper):
    '''
        Group multiple observations under the same key in the observation dictionary.
        Args:
            obs_groups: dict of {key_to_save: [keys to concat]}
    '''
    def __init__(self, env, obs_groups):
        super().__init__(env)
        self.obs_groups = obs_groups
        for key_to_save, keys_to_concat in obs_groups.items():
            assert np.all([np.array(self.observation_space.spaces[keys_to_concat[0]].shape[:-1]) ==
                           np.array(self.observation_space.spaces[k].shape[:-1])
                           for k in keys_to_concat]), \
                f"Spaces were {[(k, v) for k, v in self.observation_space.spaces.items() if k in keys_to_concat]}"
            new_last_dim = sum([self.observation_space.spaces[k].shape[-1] for k in keys_to_concat])
            new_shape = list(self.observation_space.spaces[keys_to_concat[0]].shape[:-1]) + [new_last_dim]
            self.observation_space = update_obs_space(self, {key_to_save: new_shape})

    def observation(self, obs):
        for key_to_save, keys_to_concat in self.obs_groups.items():
            obs[key_to_save] = np.concatenate([obs[k] for k in keys_to_concat], -1)
        return obs
