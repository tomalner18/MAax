"""Trains an agent to navigate a maze."""

import brax
from brax import jumpy as jp
from brax import math
from brax.envs import env

class Maze(env.Env):
    def __init__(self, **kwargs):
        config = _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)

    def reset(self, rng: jp.ndarray) -> env.State:
        """Resets the environment."""
        pass

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Steps the environment forward one timestep."""
        pass

    def reward(self, state: env.State) -> jp.ndarray:
        """Returns the reward for the current timestep."""
        pass

    def _get_obs(self, qp: brax.QP) -> jp.ndarray:
        """Returns the observation for the current timestep."""
        pass

_SYSTEM_CONFIG = """"""