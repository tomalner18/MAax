import os, sys
sys.path.append('/Users/tom/dev/imperial/FYP/MAax/')

from mujoco_worldgen import Floor, WorldBuilder, WorldParams, Env
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
from brax.generalized import pipeline

import jax
from jax import numpy as jp

from jax import random

from jax import config

config.update("jax_debug_nans", True)

seed = 7
rng = jax.random.PRNGKey(seed)

def make_env(n_substeps=15, horizon=80, deterministic_mode=False,
             floor_size=6.0, grid_size=30, door_size=2,
             n_agents=2, fixed_agent_spawn=False,
             lock_box=True, grab_box=True, grab_selective=False,
             lock_type='any_lock_specific',
             lock_grab_radius=0.25, grab_exclusive=False, grab_out_of_vision=False,
             lock_out_of_vision=True,
             box_floor_friction=0.2, other_friction=0.01, gravity=[0, 0, -50],
             action_lims=(-0.9, 0.9), polar_obs=True,
             scenario='quadrant', p_door_dropout=0.0,
             n_rooms=4, random_room_number=True,
             n_lidar_per_agent=1, visualize_lidar=True, compress_lidar_scale=None,
             n_boxes=4, box_size=0.5, box_only_z_rot=False,
             boxid_obs=True, boxsize_obs=True, pad_ramp_size=True, additional_obs={},
             # lock-box task
             task_type='all', lock_reward=5.0, unlock_penalty=7.0, shaped_reward_scale=0.25,
             return_threshold=0.1,
             # ramps
             n_ramps=4):
    '''
        This make_env function is not used anywhere; it exists to provide a simple, bare-bones
        example of how to construct a multi-agent environment using the modules framework.
    '''
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

test_env = make_env()

test_env.gen_sys(seed)

state = jax.jit(test_env.reset)(rng)

html.save('agents.html', test_env.sys, [state.pipeline_state])

rollout = []

for i in range(5):
    if i % 50 == 0:
        act = random.uniform(rng, (test_env.sys.act_size(),), dtype=jp.float64, minval=-0.25, maxval=0.25)
        rng, _ = random.split(rng)
    rollout.append(state.pipeline_state)
    state = jax.jit(test_env.step)(state, act)

# HTML(html.render(sys, rollout))

html.save('agents.html', test_env.sys, rollout)