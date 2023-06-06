import jax
import numpy as np
from jax import numpy as jp
from copy import deepcopy
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
from mae_envs.modules.agents import Agents
from mae_envs.modules.walls import RandomWalls, WallScenarios
from mae_envs.modules.objects import Boxes, Ramps, LidarSites
from mae_envs.modules.food import Food
from mae_envs.modules.world import FloorAttributes, WorldConstants
from mae_envs.modules.util import (uniform_placement, proximity_placement,
                                   uniform_placement_middle)


class TrackStatWrapper(MWrapper):
    '''
        Keeps track of important statistics that are indicative of hide and seek
        dynamics
    '''
    def __init__(self, env, n_boxes, n_ramps, n_food):
        super().__init__(env)
        self.n_boxes = n_boxes
        self.n_ramps = n_ramps
        self.n_food = n_food

    def reset(self, rng):
        state = self.env.reset(rng)
        if self.n_boxes > 0:
            self.box_pos_start = state.obs['box_pos']
        if self.n_ramps > 0:
            self.ramp_pos_start = state.obs['ramp_pos']
        if self.n_food > 0:
            self.total_food_eaten = jp.sum(state.obs['food_eat'])

        self.in_prep_phase = True

        return state

    def step(self, state, action):
        dst_state = self.env.step(state, action)
        info = deepcopy(dst_state.info)

        if self.n_food > 0:
            self.total_food_eaten += jp.sum(dst_state.obs['food_eat'])

        if self.in_prep_phase and dst_state.obs['prep_obs'][0, 0] == 1.0:
            # Track statistics at end of preparation phase
            self.in_prep_phase = False

            if self.n_boxes > 0:
                self.max_box_move_prep = jp.max(jp.linalg.norm(dst_state.obs['box_pos'] - self.box_pos_start, axis=-1))
                self.num_box_lock_prep = jp.sum(dst_state.obs['obj_lock'])
            if self.n_ramps > 0:
                self.max_ramp_move_prep = jp.max(jp.linalg.norm(dst_state.obs['ramp_pos'] - self.ramp_pos_start, axis=-1))
                if 'ramp_obj_lock' in dst_state.obs:
                    self.num_ramp_lock_prep = jp.sum(dst_state.obs['ramp_obj_lock'])
            if self.n_food > 0:
                self.total_food_eaten_prep = self.total_food_eaten

        if dst_state.done:
            # Track statistics at end of episode
            if self.n_boxes > 0:
                self.max_box_move = jp.max(jp.linalg.norm(dst_state.obs['box_pos'] - self.box_pos_start, axis=-1))
                self.num_box_lock = jp.sum(dst_state.obs['obj_lock'])
                info.update({
                    'max_box_move_prep': self.max_box_move_prep,
                    'max_box_move': self.max_box_move,
                    'num_box_lock_prep': self.num_box_lock_prep,
                    'num_box_lock': self.num_box_lock})

            if self.n_ramps > 0:
                self.max_ramp_move = jp.max(jp.linalg.norm(dst_state.obs['ramp_pos'] - self.ramp_pos_start, axis=-1))
                info.update({
                    'max_ramp_move_prep': self.max_ramp_move_prep,
                    'max_ramp_move': self.max_ramp_move})
                if 'ramp_obj_lock' in dst_state.obs:
                    self.num_ramp_lock = jp.sum(dst_state.obs['ramp_obj_lock'])
                    info.update({
                        'num_ramp_lock_prep': self.num_ramp_lock_prep,
                        'num_ramp_lock': self.num_ramp_lock})

            if self.n_food > 0:
                info.update({
                    'food_eaten': self.total_food_eaten,
                    'food_eaten_prep': self.total_food_eaten_prep})

        return dst_state.replace(info=info)


class HideAndSeekRewardWrapper(MWrapper):
    '''
        Establishes hide and seek dynamics (see different reward types below). Defaults to first half
            of agents being hiders and second half seekers unless underlying environment specifies
            'n_hiders' and 'n_seekers'.
        Args:
            rew_type (string): can be
                'selfish': hiders and seekers play selfishly. Seekers recieve 1.0 if they can
                    see any hider and -1.0 otherwise. Hiders recieve 1.0 if they are seen by no
                    seekers and -1.0 otherwise.
                'joint_mean': hiders and seekers recieve the mean reward of their team
                'joint_zero_sum': hiders recieve 1.0 only if all hiders are hidden and -1.0 otherwise.
                    Seekers recieve 1.0 if any seeker sees a hider.
            reward_scale (float): scales the reward by this factor
    '''
    def __init__(self, env, n_hiders, n_seekers, rew_type='selfish', reward_scale=1.0):
        super().__init__(env)
        self.n_agents = self.unwrapped.n_agents
        self.rew_type = rew_type
        self.n_hiders = n_hiders
        self.n_seekers = n_seekers
        self.reward_scale = reward_scale
        assert n_hiders + n_seekers == self.n_agents, "n_hiders + n_seekers must equal n_agents"

        self.metadata['n_hiders'] = n_hiders
        self.metadata['n_seekers'] = n_seekers

        # Agent names are used to plot agent-specific rewards on tensorboard
        self.unwrapped.agent_names = [f'hider{i}' for i in range(self.n_hiders)] + \
                                     [f'seeker{i}' for i in range(self.n_seekers)]

    def step(self, state, action):
        dst_state = self.env.step(state, action)
        obs = dst_state.obs

        this_rew = jp.ones((self.n_agents,))
        mask_aa_con = obs['mask_aa_con']

        hiders = mask_aa_con[:self.n_hiders, self.n_hiders:]
        seekers = mask_aa_con[self.n_hiders:, :self.n_hiders]

        hiders_contact = jp.any(hiders, axis=1)
        seekers_contact = jp.any(seekers, axis=0)

        hider_rewards = jp.where(hiders_contact, -1.0, 1.0)
        seeker_rewards = jp.where(seekers_contact, 1.0, -1.0)

        this_rew = jp.concatenate([hider_rewards, seeker_rewards])

        if self.rew_type == 'joint_mean':
           this_rew.at[:self.n_hiders].set(this_rew[:self.n_hiders].mean())
           this_rew.at[self.n_hiders:].set(this_rew[self.n_hiders:].mean())
        elif self.rew_type == 'joint_zero_sum':
            this_rew.at[:self.n_hiders].set(jp.min(this_rew[:self.n_hiders]))
            this_rew.at[self.n_hiders:].set(jp.max(this_rew[self.n_hiders:]))
        elif self.rew_type == 'selfish':
            pass
        else:
            assert False, f'Hide and Seek reward type {self.rew_type} is not implemented'

        this_rew = jp.multiply(this_rew, self.reward_scale)
        # this_rew = jp.add(this_rew, dst_state.reward)
        return dst_state.replace(reward=this_rew)


class MaskUnseenAction(MWrapper):
    '''
        Masks a (binary) action with some probability if agent or any of its teammates was being observed
        by opponents at any of the last n_latency time step

        Args:
            team_idx (int): Team index (e.g. 0 = hiders) of team whose actions are
                            masked
            action_key (string): key of action to be masked
    '''

    def __init__(self, env, team_idx, action_key):
        super().__init__(env)
        self.team_idx = team_idx
        self.action_key = action_key
        self.n_agents = self.unwrapped.n_agents
        self.n_hiders = self.metadata['n_hiders']

    def reset(self, rng):
        state = self.env.reset(rng)
        self.prev_obs = state.obs
        self.this_team = self.metadata['team_index'] == self.team_idx
        return state

    def step(self, state, action):
        is_caught = jp.any(self.prev_obs['mask_aa_con'][self.n_hiders:, :self.n_hiders])
        if is_caught:
            action[self.action_key][self.this_team] = 0

        dst_state = self.env.step(state, action)
        self.prev_obs = dst_state.obs
        return dst_state.replace(obs=deepcopy(self.prev_obs))


def quadrant_placement(grid, obj_size, metadata, random_state):
    '''
        Places object within the bottom right quadrant of the playing field
    '''
    grid_size = len(grid)
    qsize = metadata['quadrant_size']
    pos = np.array([random_state.randint(grid_size - qsize, grid_size - obj_size[0] - 1),
                    random_state.randint(1, qsize - obj_size[1] - 1)])
    return pos


def outside_quadrant_placement(grid, obj_size, metadata, random_state):
    '''
        Places object outside of the bottom right quadrant of the playing field
    '''
    grid_size = len(grid)
    qsize = metadata['quadrant_size']
    poses = [
        np.array([random_state.randint(1, grid_size - qsize - obj_size[0] - 1),
                  random_state.randint(1, qsize - obj_size[1] - 1)]),
        np.array([random_state.randint(1, grid_size - qsize - obj_size[0] - 1),
                  random_state.randint(qsize, grid_size - obj_size[1] - 1)]),
        np.array([random_state.randint(grid_size - qsize, grid_size - obj_size[0] - 1),
                  random_state.randint(qsize, grid_size - obj_size[1] - 1)]),
    ]
    return poses[random_state.randint(0, 3)]


# def make_env(n_substeps=15, horizon=80, deterministic_mode=False,
#              floor_size=6.0, grid_size=30, door_size=2,
#              n_hiders=1, n_seekers=1, max_n_agents=None,
#              n_boxes=2, n_ramps=1, n_elongated_boxes=0,
#              rand_num_elongated_boxes=False, n_min_boxes=None,
#              box_size=0.5, boxid_obs=False, box_only_z_rot=True,
#              rew_type='joint_zero_sum',
#              lock_box=True, grab_box=True, lock_ramp=True,
#              lock_type='any_lock_specific',
#              lock_grab_radius=0.25, lock_out_of_vision=True, grab_exclusive=False,
#              grab_out_of_vision=False, grab_selective=False,
#              box_floor_friction=0.2, other_friction=0.01, gravity=[0, 0, -50],
#              action_lims=(-0.9, 0.9), polar_obs=True,
#              scenario='quadrant', quadrant_game_hider_uniform_placement=False,
#              p_door_dropout=0.0,
#              n_rooms=4, random_room_number=True, prob_outside_walls=1.0,
#              n_lidar_per_agent=0, visualize_lidar=False, compress_lidar_scale=None,
#              hiders_together_radius=None, seekers_together_radius=None,
#              prep_fraction=0.4, prep_obs=False,
#              team_size_obs=False,
#              restrict_rect=None, penalize_objects_out=False,
#              n_food=0, food_radius=None, food_respawn_time=None, max_food_health=1,
#              food_together_radius=None, food_rew_type='selfish', eat_when_caught=False,
#              food_reward_scale=1.0, food_normal_centered=False, food_box_centered=False,
#              n_food_cluster=1):

#     grab_radius_multiplier = lock_grab_radius / box_size
#     lock_radius_multiplier = lock_grab_radius / box_size

#     env = Base(n_agents=n_hiders + n_seekers, n_substeps=n_substeps, horizon=horizon,
#                floor_size=floor_size, grid_size=grid_size,
#                action_lims=action_lims,
#                deterministic_mode=deterministic_mode)

#     if scenario == 'randomwalls':
#         env.add_module(RandomWalls(
#             grid_size=grid_size, num_rooms=n_rooms,
#             random_room_number=random_room_number, min_room_size=6,
#             door_size=door_size,
#             prob_outside_walls=prob_outside_walls, gen_door_obs=False))
#         box_placement_fn = uniform_placement
#         ramp_placement_fn = uniform_placement
#         cell_size = floor_size / grid_size

#         first_hider_placement = uniform_placement
#         if hiders_together_radius is not None:
#             htr_in_cells = np.ceil(hiders_together_radius / cell_size).astype(int)

#             env.metadata['hiders_together_radius'] = htr_in_cells

#             close_to_first_hider_placement = proximity_placement(
#                                                 "agent", 0, "hiders_together_radius")

#             agent_placement_fn = [first_hider_placement] + \
#                                  [close_to_first_hider_placement] * (n_hiders - 1)
#         else:
#             agent_placement_fn = [first_hider_placement] * n_hiders

#         first_seeker_placement = uniform_placement

#         if seekers_together_radius is not None:
#             str_in_cells = np.ceil(seekers_together_radius / cell_size).astype(int)

#             env.metadata['seekers_together_radius'] = str_in_cells

#             close_to_first_seeker_placement = proximity_placement(
#                                                 "agent", n_hiders, "seekers_together_radius")

#             agent_placement_fn += [first_seeker_placement] + \
#                                   [close_to_first_seeker_placement] * (n_seekers - 1)
#         else:
#             agent_placement_fn += [first_seeker_placement] * (n_seekers)

#     elif scenario == 'quadrant':
#         env.add_module(WallScenarios(grid_size=grid_size, door_size=door_size,
#                                      scenario=scenario, friction=other_friction,
#                                      p_door_dropout=p_door_dropout))
#         box_placement_fn = quadrant_placement
#         ramp_placement_fn = uniform_placement
#         hider_placement = uniform_placement if quadrant_game_hider_uniform_placement else quadrant_placement
#         agent_placement_fn = [hider_placement] * n_hiders + [outside_quadrant_placement] * n_seekers
#     else:
#         raise ValueError(f"Scenario {scenario} not supported.")

#     env.add_module(Agents(n_hiders + n_seekers,
#                           placement_fn=agent_placement_fn,
#                           color=[np.array((66., 235., 244., 255.)) / 255] * n_hiders + [(1., 0., 0., 1.)] * n_seekers,
#                           friction=other_friction,
#                           polar_obs=polar_obs))
#     if np.max(n_boxes) > 0:
#         env.add_module(Boxes(n_boxes=n_boxes, placement_fn=box_placement_fn,
#                              friction=box_floor_friction, polar_obs=polar_obs,
#                              n_elongated_boxes=n_elongated_boxes,
#                              boxid_obs=boxid_obs, box_only_z_rot=box_only_z_rot))
#     if n_ramps > 0:
#         env.add_module(Ramps(n_ramps=n_ramps, placement_fn=ramp_placement_fn, friction=other_friction, polar_obs=polar_obs,
#                              pad_ramp_size=(np.max(n_elongated_boxes) > 0)))
#     if n_lidar_per_agent > 0 and visualize_lidar:
#         env.add_module(LidarSites(n_agents=n_hiders + n_seekers, n_lidar_per_agent=n_lidar_per_agent))


#     if box_floor_friction is not None:
#         env.add_module(FloorAttributes(friction=box_floor_friction))
#     env.add_module(WorldConstants(gravity=gravity))
#     env.reset()
#     keys_self = ['agent_qpos_qvel', 'hider', 'prep_obs']
#     keys_mask_self = ['mask_aa_obs']
#     keys_external = ['agent_qpos_qvel']
#     keys_copy = ['you_lock', 'team_lock', 'ramp_you_lock', 'ramp_team_lock']
#     keys_mask_external = []
#     env = SplitMultiAgentActions(env)
#     if team_size_obs:
#         keys_self += ['team_size']
#     env = TeamMembership(env, np.append(np.zeros((n_hiders,)), np.ones((n_seekers,))))
#     env = AgentAgentObsMask2D(env)
#     hider_obs = np.array([[1]] * n_hiders + [[0]] * n_seekers)
#     env = AddConstantObservationsWrapper(env, new_obs={'hider': hider_obs})
#     env = HideAndSeekRewardWrapper(env, n_hiders=n_hiders, n_seekers=n_seekers,
#                                    rew_type=rew_type)

#     env = PreparationPhase(env, prep_fraction=prep_fraction)
#     env = DiscretizeActionWrapper(env, 'action_movement')
#     if np.max(n_boxes) > 0:
#         env = AgentGeomObsMask2D(env, pos_obs_key='box_pos', mask_obs_key='mask_ab_obs',
#                                  geom_idxs_obs_key='box_geom_idxs')
#         keys_external += ['mask_ab_obs', 'box_obs']
#         keys_mask_external.append('mask_ab_obs')

#     if lock_box and np.max(n_boxes) > 0:
#         env = LockObjWrapper(env, body_names=[f'moveable_box{i}' for i in range(np.max(n_boxes))],
#                              agent_idx_allowed_to_lock=np.arange(n_hiders+n_seekers),
#                              lock_type=lock_type, radius_multiplier=lock_radius_multiplier,
#                              obj_in_game_metadata_keys=["curr_n_boxes"],
#                              agent_allowed_to_lock_keys=None if lock_out_of_vision else ["mask_ab_obs"])

#     if n_ramps > 0:
#         env = AgentGeomObsMask2D(env, pos_obs_key='ramp_pos', mask_obs_key='mask_ar_obs',
#                                  geom_idxs_obs_key='ramp_geom_idxs')
#         if lock_ramp:
#             env = LockObjWrapper(env, body_names=[f'ramp{i}:ramp' for i in range(n_ramps)],
#                                  agent_idx_allowed_to_lock=np.arange(n_hiders+n_seekers),
#                                  lock_type=lock_type, ac_obs_prefix='ramp_',
#                                  radius_multiplier=lock_radius_multiplier,
#                                  obj_in_game_metadata_keys=['curr_n_ramps'],
#                                  agent_allowed_to_lock_keys=None if lock_out_of_vision else ["mask_ar_obs"])
#         keys_external += ['ramp_obs']
#         keys_mask_external.append('mask_ar_obs')


#     if prep_obs:
#         env = TrackStatWrapper(env, np.max(n_boxes), n_ramps, n_food)
#     env = SplitObservations(env, keys_self + keys_mask_self, keys_copy=keys_copy, keys_self_matrices=keys_mask_self)
#     env = SpoofEntityWrapper(env, np.max(n_boxes), ['box_obs', 'you_lock', 'team_lock', 'obj_lock'], ['mask_ab_obs'])

#     if max_n_agents is not None:
#         env = SpoofEntityWrapper(env, max_n_agents, ['agent_qpos_qvel', 'hider', 'prep_obs'], ['mask_aa_obs'])
#     env = LockAllWrapper(env, remove_object_specific_lock=True)
#     env = NoActionsInPrepPhase(env, np.arange(n_hiders, n_hiders + n_seekers))
#     env = ConcatenateObsWrapper(env, {'agent_qpos_qvel': ['agent_qpos_qvel', 'hider', 'prep_obs'],
#                                       'box_obs': ['box_obs', 'you_lock', 'team_lock', 'obj_lock'],
#                                       'ramp_obs': ['ramp_obs'] + (['ramp_you_lock', 'ramp_team_lock', 'ramp_obj_lock'] if lock_ramp else [])})
#     env = SelectKeysWrapper(env, keys_self=keys_self,
#                             keys_other=keys_external + keys_mask_self + keys_mask_external)
#     return env
