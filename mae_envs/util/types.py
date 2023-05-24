"""Defines some types used in Maax."""

from typing import Dict, Generic, TypeVar, Union, Any

import brax.envs
import jax
import jax.numpy as jnp
from typing_extensions import TypeAlias

# MDP types
Observation: TypeAlias = jnp.ndarray
Action: TypeAlias = jnp.ndarray
Reward: TypeAlias = jnp.ndarray
Done: TypeAlias = jnp.ndarray
EnvState: TypeAlias = brax.envs.State

RNGKey: TypeAlias = jax.random.KeyArray
PipelineState: TypeAlias = Any