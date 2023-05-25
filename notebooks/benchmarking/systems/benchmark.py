import os, sys
sys.path.append('/Users/tom/dev/imperial/FYP/MAax/')

import time
import json

from typing import Any, Callable, Tuple
from functools import partial

from worldgen import Floor, WorldBuilder, WorldParams
from mae_envs.modules.agents import Agents
from mae_envs.modules.walls import RandomWalls, WallScenarios
from mae_envs.modules.world import FloorAttributes, WorldConstants
from mae_envs.modules.objects import Boxes, Cylinders, LidarSites, Ramps
from mae_envs.modules.util import uniform_placement, center_placement
from mae_envs.envs.hide_and_seek import quadrant_placement

import brax
import numpy as np
from brax.io import mjcf, html
from maax.envs.base import Base
from mae_envs.util.types import RNGKey, PipelineState, Action
from brax.generalized import pipeline

import jax
from jax import numpy as jp

from jax import random

from brax.envs.env import State


def make_env(seed, n_substeps=15, horizon=80, deterministic_mode=False,
             floor_size=6.0, grid_size=30, door_size=2,
             n_hiders=1, n_seekers=1, max_n_agents=None,
             n_boxes=1, n_ramps=1, n_elongated_boxes=0,
             rand_num_elongated_boxes=False, n_min_boxes=None,
             box_size=0.5, boxid_obs=True, boxsize_obs=True, box_only_z_rot=True,
             pad_ramp_size=True,
             rew_type='joint_zero_sum',
             lock_box=True, grab_box=True, lock_ramp=True,
             lock_type='any_lock_specific',
             lock_grab_radius=0.25, lock_out_of_vision=True, grab_exclusive=False,
             grab_out_of_vision=False, grab_selective=False,
             box_floor_friction=0.2, other_friction=0.01, gravity=[0, 0, -50],
             action_lims=(-0.9, 0.9), polar_obs=True,
             scenario='quadrant', quadrant_game_hider_uniform_placement=False,
             p_door_dropout=0.0,
             n_rooms=4, random_room_number=True, prob_outside_walls=1.0,
             n_lidar_per_agent=0, visualize_lidar=False, compress_lidar_scale=None,
             hiders_together_radius=None, seekers_together_radius=None,
             prep_fraction=0.4, prep_obs=False,
             team_size_obs=False,
             restrict_rect=None, penalize_objects_out=False,
             ):
    '''
        This make_env function is not used anywhere; it exists to provide a simple, bare-bones
        example of how to construct a multi-agent environment using the modules framework.
    '''
    n_agents = n_seekers + n_hiders
    env = Base(n_agents=n_agents, n_substeps=n_substeps, horizon=horizon, grid_size=grid_size,
               deterministic_mode=deterministic_mode, seed=seed)
    env.add_module(WallScenarios(grid_size=grid_size, door_size=door_size,
                                     scenario=scenario, friction=other_friction,
                                     p_door_dropout=p_door_dropout))
    box_placement_fn = uniform_placement
    ramp_placement_fn = uniform_placement
    agent_placement_fn = uniform_placement

    env.add_module(Agents(n_agents,
                          placement_fn=agent_placement_fn,
                          color=[np.array((66., 235., 244., 255.)) / 255] * n_agents,
                          friction=other_friction,
                          polar_obs=polar_obs))

    if np.max(n_boxes) > 0:
        env.add_module(Boxes(n_boxes=n_boxes, placement_fn=box_placement_fn,
                             friction=box_floor_friction, polar_obs=polar_obs,
                             n_elongated_boxes=0,
                             boxid_obs=boxid_obs,
                             box_only_z_rot=box_only_z_rot,
                             boxsize_obs=boxsize_obs,
                             free=True))

    if n_ramps > 0:
        env.add_module(Ramps(n_ramps=n_ramps, placement_fn=ramp_placement_fn,
                             friction=other_friction, polar_obs=polar_obs,
                             pad_ramp_size=pad_ramp_size, free=True))

    # if n_lidar_per_agent > 0 and visualize_lidar:
    #     env.add_module(LidarSites(n_agents=n_agents, n_lidar_per_agent=n_lidar_per_agent))

    # env.add_module(WorldConstants(gravity=gravity))

    return env

def rollout(env, batch_size, step_fn, reset_fn, random_key, unroll_fn):
    # Start timer
    start_time = time.process_time()
    print("Starting Batch Size: ", batch_size)
    print("at: ", start_time)
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=batch_size)
    # Define initial batches states and actions
    init_states = reset_fn(keys)
    acts = jp.zeros(shape=(batch_size, env.sys.act_size()), dtype=jp.float32)
    dst_states, rollouts = jax.vmap(unroll_fn)(init_states, acts, keys)
    et = time.process_time()
    dt = et - start_time
    return dt

def main():
    seed = 10
    episode_length = 50
    random_key = jax.random.PRNGKey(seed)

    env = make_env(seed)
    env.gen_sys(seed)

    batch_sizes = [1, 2, 4, 8, 16, 32]

    jit_step_fn = jax.jit(env.step)
    jit_batch_reset_fn = jax.jit(jax.vmap(env.reset))

    @jax.jit
    def randomise_action(act, random_key):
        random_key, _ = random.split(random_key)
        return random.uniform(random_key, shape=act.shape, minval=-0.25, maxval=0.25), random_key

    @jax.jit
    def play_step_fn(state: State, act: Action, random_key: RNGKey):
        act, random_key = randomise_action(act, random_key)
        state = jit_step_fn(state, act)
        return state, act, random_key, state.pipeline_state

    @partial(jax.jit, static_argnames=("play_step_fn", "episode_length"))
    def generate_unroll(
        init_state: State,
        act: Action,
        random_key: RNGKey,
        episode_length: int,
        play_step_fn) -> Tuple[State, Action, RNGKey]:
        """Generates an episode according to random action, returns the final state of
        the episode and the transitions of the episode.

        Args:
            init_state: first state of the rollout.
            act: The initial action
            random_key: random key for stochasiticity handling.
            episode_length: length of the rollout.
            index: index of the rollout.
            play_step_fn: function describing how a step need to be taken.

        Returns:
            A new state, the experienced transition.
        """
        def scan_play_step_fn(
            carry: Tuple[State, Action, RNGKey], unused_arg: Any) ->Tuple[Tuple[State, Action, RNGKey], PipelineState]:
            state, act, random_key, p_states = play_step_fn(*carry)
            return (state, act, random_key), p_states


        (dst_state, dst_act, key), rollout = jax.lax.scan(
            scan_play_step_fn, (init_state, act, random_key), None, length=episode_length)

        return dst_state, rollout


    unroll_fn = partial(
        generate_unroll,
        episode_length=episode_length,
        play_step_fn=play_step_fn,
    )

    # Run rollouts and time them
    batch_time= dict()
    for batch_size in batch_sizes:
        batch_time[batch_size] = rollout(env, batch_size, jit_step_fn, jit_batch_reset_fn, random_key, unroll_fn)

    # Save batch times
    with open('batch_times.json', 'w') as f: 
        json.dump(batch_time, f)


if __name__ == "__main__":
    main()
