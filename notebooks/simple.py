# %%
import os, sys
sys.path.append('/Users/tom/dev/imperial/FYP/MAax/')

# %%
from typing import Any, Callable, Tuple

# %%
from worldgen import Floor, WorldBuilder, WorldParams
from mae_envs.modules.agents import Agents
from mae_envs.modules.walls import RandomWalls, WallScenarios
from mae_envs.modules.world import FloorAttributes, WorldConstants
from mae_envs.modules.objects import Boxes, Cylinders, LidarSites, Ramps
from mae_envs.modules.util import uniform_placement, center_placement, close_to_other_object_placement
from mae_envs.envs.hide_and_seek import quadrant_placement, outside_quadrant_placement, HideAndSeekRewardWrapper, TrackStatWrapper
from maax.envs.base import Base
from mae_envs.wrappers.multi_agent import (SplitMultiAgentActions,
                                           SplitObservations, SelectKeysWrapper)
from mae_envs.wrappers.util import (ConcatenateObsWrapper,
                                    MaskActionWrapper, SpoofEntityWrapper,
                                    AddConstantObservationsWrapper, MWrapper)
from mae_envs.wrappers.manipulation import (LockObjWrapper, LockAllWrapper)
from mae_envs.wrappers.lidar import Lidar
from mae_envs.wrappers.line_of_sight import (AgentAgentObsMask2D, AgentGeomObsMask2D,
                                             AgentSiteObsMask2D)
from mae_envs.wrappers.prep_phase import (PreparationPhase, NoActionsInPrepPhase,
                                          MaskPrepPhaseAction)
from mae_envs.wrappers.limit_mvmnt import RestrictAgentsRect
from mae_envs.wrappers.team import TeamMembership
from mae_envs.wrappers.food import FoodHealthWrapper, AlwaysEatWrapper


# %%
import brax
import numpy as np
from brax.io import mjcf, html
from maax.envs.base import Base
from mae_envs.util.types import RNGKey, PipelineState, Action
from brax.generalized import pipeline

import jax
from jax import numpy as jp

from jax import random

from IPython.display import HTML, clear_output
clear_output()


# %%
from maax.envs.base import State

# %%
seed = 14
random_key = jax.random.PRNGKey(seed)

# %%
def make_env(n_substeps=15, horizon=80, deterministic_mode=False,
             floor_size=6.0, grid_size=30, door_size=2,
             n_hiders=1, n_seekers=1, max_n_agents=None,
             n_boxes=2, n_ramps=1, n_elongated_boxes=0,
             rand_num_elongated_boxes=False, n_min_boxes=None,
             box_size=0.5, boxid_obs=False, box_only_z_rot=True,
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
             n_food=0, food_radius=None, food_respawn_time=None, max_food_health=1,
             food_together_radius=None, food_rew_type='selfish', eat_when_caught=False,
             food_reward_scale=1.0, food_normal_centered=False, food_box_centered=False,
             n_food_cluster=1):

    grab_radius_multiplier = lock_grab_radius / box_size
    lock_radius_multiplier = lock_grab_radius / box_size

    env = Base(n_agents=n_hiders + n_seekers, n_substeps=n_substeps, horizon=horizon,
               floor_size=floor_size, grid_size=grid_size,
               action_lims=action_lims,
               deterministic_mode=deterministic_mode)

    if scenario == 'randomwalls':
        env.add_module(RandomWalls(
            grid_size=grid_size, num_rooms=n_rooms,
            random_room_number=random_room_number, min_room_size=6,
            door_size=door_size,
            prob_outside_walls=prob_outside_walls, gen_door_obs=False))
        box_placement_fn = uniform_placement
        ramp_placement_fn = uniform_placement
        cell_size = floor_size / grid_size

        first_hider_placement = uniform_placement
        if hiders_together_radius is not None:
            htr_in_cells = np.ceil(hiders_together_radius / cell_size).astype(int)

            env.metadata['hiders_together_radius'] = htr_in_cells

            close_to_first_hider_placement = close_to_other_object_placement(
                                                "agent", 0, "hiders_together_radius")

            agent_placement_fn = [first_hider_placement] + \
                                 [close_to_first_hider_placement] * (n_hiders - 1)
        else:
            agent_placement_fn = [first_hider_placement] * n_hiders

        first_seeker_placement = uniform_placement

        if seekers_together_radius is not None:
            str_in_cells = np.ceil(seekers_together_radius / cell_size).astype(int)

            env.metadata['seekers_together_radius'] = str_in_cells

            close_to_first_seeker_placement = close_to_other_object_placement(
                                                "agent", n_hiders, "seekers_together_radius")

            agent_placement_fn += [first_seeker_placement] + \
                                  [close_to_first_seeker_placement] * (n_seekers - 1)
        else:
            agent_placement_fn += [first_seeker_placement] * (n_seekers)

    elif scenario == 'quadrant':
        env.add_module(WallScenarios(grid_size=grid_size, door_size=door_size,
                                     scenario=scenario, friction=other_friction,
                                     p_door_dropout=p_door_dropout))
        box_placement_fn = quadrant_placement
        ramp_placement_fn = uniform_placement
        hider_placement = uniform_placement if quadrant_game_hider_uniform_placement else quadrant_placement
        agent_placement_fn = [hider_placement] * n_hiders + [outside_quadrant_placement] * n_seekers
    else:
        raise ValueError(f"Scenario {scenario} not supported.")

    env.add_module(Agents(n_hiders + n_seekers,
                          placement_fn=agent_placement_fn,
                          color=[np.array((66., 235., 244., 255.)) / 255] * n_hiders + [(1., 0., 0., 1.)] * n_seekers,
                          friction=other_friction,
                          polar_obs=polar_obs))
    if np.max(n_boxes) > 0:
        env.add_module(Boxes(n_boxes=n_boxes, placement_fn=box_placement_fn,
                             friction=box_floor_friction, polar_obs=polar_obs,
                             n_elongated_boxes=n_elongated_boxes,
                             boxid_obs=boxid_obs, box_only_z_rot=box_only_z_rot))
    if n_ramps > 0:
        env.add_module(Ramps(n_ramps=n_ramps, placement_fn=ramp_placement_fn, friction=other_friction, polar_obs=polar_obs,
                             pad_ramp_size=(np.max(n_elongated_boxes) > 0)))

    if box_floor_friction is not None:
        env.add_module(FloorAttributes(friction=box_floor_friction))

    keys_self = ['agent_qpos_qvel', 'hider', 'prep_obs']
    keys_mask_self = ['mask_aa_obs']
    keys_external = ['agent_qpos_qvel']
    keys_copy = ['you_lock', 'team_lock', 'ramp_you_lock', 'ramp_team_lock']
    keys_mask_external = []
    env = SplitMultiAgentActions(env)
    if team_size_obs:
        keys_self += ['team_size']
    env = TeamMembership(env, np.append(np.zeros((n_hiders,)), np.ones((n_seekers,))))
    env = AgentAgentObsMask2D(env)
    hider_obs = np.array([[1]] * n_hiders + [[0]] * n_seekers)
    env = AddConstantObservationsWrapper(env, new_obs={'hider': hider_obs})
    env = HideAndSeekRewardWrapper(env, n_hiders=n_hiders, n_seekers=n_seekers,
                                   rew_type=rew_type)

    env = PreparationPhase(env, prep_fraction=prep_fraction)

    if np.max(n_boxes) > 0:
        env = AgentGeomObsMask2D(env, pos_obs_key='box_pos', mask_obs_key='mask_ab_obs',
                                 geom_idxs_obs_key='box_geom_idxs')
        keys_external += ['box_obs']

    # if lock_box and np.max(n_boxes) > 0:
    #     env = LockObjWrapper(env, body_names=[f'moveable_box{i}' for i in range(np.max(n_boxes))],
    #                          agent_idx_allowed_to_lock=np.arange(n_hiders+n_seekers),
    #                          lock_type=lock_type, radius_multiplier=lock_radius_multiplier,
    #                          obj_in_game_metadata_keys=["curr_n_boxes"],
    #                          agent_allowed_to_lock_keys=None if lock_out_of_vision else ["mask_ab_obs"])

    if n_ramps > 0:
        # if lock_ramp:
        #     env = LockObjWrapper(env, body_names=[f'ramp{i}:ramp' for i in range(n_ramps)],
        #                          agent_idx_allowed_to_lock=np.arange(n_hiders+n_seekers),
        #                          lock_type=lock_type, ac_obs_prefix='ramp_',
        #                          radius_multiplier=lock_radius_multiplier,
        #                          obj_in_game_metadata_keys=['curr_n_ramps'],
        #                          agent_allowed_to_lock_keys=None if lock_out_of_vision else ["mask_ar_obs"])
        keys_external += ['ramp_obs']


    if prep_obs:
        env = TrackStatWrapper(env, np.max(n_boxes), n_ramps, n_food)
    env = SplitObservations(env, keys_self + keys_mask_self, keys_copy=keys_copy, keys_self_matrices=keys_mask_self)
    env = SpoofEntityWrapper(env, np.max(n_boxes), ['box_obs', 'you_lock', 'team_lock', 'obj_lock'], ['mask_ab_obs'])

    if max_n_agents is not None:
        env = SpoofEntityWrapper(env, max_n_agents, ['agent_qpos_qvel', 'hider', 'prep_obs'], ['mask_aa_obs'])
    # env = LockAllWrapper(env, remove_object_specific_lock=True)
    env = NoActionsInPrepPhase(env, np.arange(n_hiders, n_hiders + n_seekers))
    env = ConcatenateObsWrapper(env, {'agent_qpos_qvel': ['agent_qpos_qvel', 'hider', 'prep_obs'],
                                      'box_obs': ['box_obs', 'you_lock', 'team_lock', 'obj_lock'],
                                      'ramp_obs': ['ramp_obs'] + (['ramp_you_lock', 'ramp_team_lock', 'ramp_obj_lock'] if lock_ramp else [])})
    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_other=keys_external + keys_mask_self + keys_mask_external)
    return env

# %%
test_env = make_env(random_key)

test_env.gen_sys(seed)

# print(test_env.sys.geoms)
# print(test_env.sys.dof)

state = jax.jit(test_env.reset)(random_key)

# %%
def randomise_action(act, random_key):
    random_key, _ = random.split(random_key)
    return random.uniform(random_key, shape=act.shape, minval=-0.25, maxval=0.25), random_key

# %%
# rollout = []

# for i in range(500):
#     print(i)
    # act, rng = jax.lax.cond(i % 50 == 0, randomise_action, maintain_action, act, act_size, rng)
#     rollout.append(state.pipeline_state)
#     state = jit_step_fn(state, act)

# html.save('agents.html', test_env.sys, rollout)

# %%

jit_step_fn = jax.jit(test_env.step)
act_size = test_env.sys.act_size()

episode_length = 100
act = jp.zeros(shape=act_size)

@jax.jit
def play_step_fn(state: State, act: Action, random_key: RNGKey, index: int):
    act, random_key = jax.lax.cond(index % 50 == 0, randomise_action, lambda x, y: (x, y), act, random_key)
    state = jit_step_fn(state, act)
    return state, act, random_key, index + 1, state.pipeline_state

def scan_play_step_fn(
    carry: Tuple[State, Action, RNGKey, int], unused_arg: Any
) ->Tuple[Tuple[State, RNGKey, int], PipelineState]:
    state, act, random_key, index, p_states = play_step_fn(*carry)
    return (state, act, random_key, index), p_states
    

(dst_state, dst_act, key, index), rollout = jax.lax.scan(scan_play_step_fn, (state, act, random_key, 0), None, length=episode_length)

# %%
states_list = []

for i in range(episode_length):
    s = jax.tree_util.tree_map(lambda x: x[i], rollout)
    states_list.append(s)


print(len(states_list))
html.save('agents.html', test_env.sys, states_list)


