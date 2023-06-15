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
from mae_envs.modules.objects import Boxes, Ramps
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

        if self.in_prep_phase and dst_state.obs['prep_rem'][0, 0] == 1.0:
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
        d_obs = dst_state.d_obs

        this_rew = jp.ones((self.n_agents,))
        mask_aa_con = d_obs['mask_aa_con']

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


def quad_placement(grid, obj_size, metadata, random_state):
    '''
        Places object within the bottom right quad of the playing field
    '''
    grid_size = len(grid)
    qsize = metadata['quad_size']
    pos = np.array([random_state.randint(grid_size - qsize, grid_size - obj_size[0] - 1),
                    random_state.randint(1, qsize - obj_size[1] - 1)])
    return pos


def outside_quad_placement(grid, obj_size, metadata, random_state):
    '''
        Places object outside of the bottom right quad of the playing field
    '''
    grid_size = len(grid)
    qsize = metadata['quad_size']
    poses = [
        np.array([random_state.randint(1, grid_size - qsize - obj_size[0] - 1),
                  random_state.randint(1, qsize - obj_size[1] - 1)]),
        np.array([random_state.randint(1, grid_size - qsize - obj_size[0] - 1),
                  random_state.randint(qsize, grid_size - obj_size[1] - 1)]),
        np.array([random_state.randint(grid_size - qsize, grid_size - obj_size[0] - 1),
                  random_state.randint(qsize, grid_size - obj_size[1] - 1)]),
    ]
    return poses[random_state.randint(0, 3)]
