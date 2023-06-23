import gym
from gym.spaces import Discrete, MultiDiscrete, Tuple
import numpy as np
from worldgen.util.rotation import mat2quat
from maax.wrappers.util import MWrapper
from maax.util.geometry import dist_pt_to_cuboid
from copy import deepcopy
from itertools import compress



class LockObjWrapper(MWrapper):
    '''
        Allows agents to lock objects at their current position.
        Args:
            body_names (list): list of body names that the agent can lock
            radius_multiplier (float): How far away can this be activated (multiplier on box size)
            agent_idx_allowed_to_lock (np array of ints): Indicies of agents that are allowed to lock.
                Defaults to all
            lock_type (string): Options are
                any_lock: if any agent wants to lock an object it will get locked
                all_lock: all agents that are close enough must want to lock the object
                any_lock_specific: if any agent wants to lock an object it will get locked. However,
                    now the lock is agent specific, and only the agent that locked the object can unlock it.
                all_lock_team_specific: like all_lock, but only team members of the agent that
                    locked the object can unlock it.
            ac_obs_prefix (string): prefix for the action and observation keys. This is useful if using
                the lock wrapper more than once.
            obj_in_game_metadata_keys (list of string): keys in metadata with boolean array saying
                which objects are currently in the game. This is used in the event we are randomizing
                number of objects
            agent_allowed_to_lock_keys (list of string): keys in obs determining whether agent is allowed
                to lock a certain object. Each key should be a mask matrix of dim (n_agents, n_obj)

    '''
    def __init__(self, env, body_names, radius_multiplier=1.5, agent_idx_allowed_to_lock=None,
                 lock_type="any_lock", ac_obs_prefix='', obj_in_game_metadata_keys=None,
                 agent_allowed_to_lock_keys=None):
        super().__init__(env)
        self.n_agents = self.unwrapped.n_agents
        self.n_obj = len(body_names)
        self.body_names = body_names
        self.agent_idx_allowed_to_lock = np.arange(self.n_agents) if agent_idx_allowed_to_lock is None else agent_idx_allowed_to_lock
        self.lock_type = lock_type
        self.ac_obs_prefix = ac_obs_prefix
        self.obj_in_game_metadata_keys = obj_in_game_metadata_keys
        self.agent_allowed_to_lock_keys = agent_allowed_to_lock_keys
        # self.action_space.spaces[f'action_{ac_obs_prefix}glue'] = (
        #     Tuple([MultiDiscrete([2] * self.n_obj) for _ in range(self.n_agents)]))
        # self.observation_space = update_obs_space(env, {f'{ac_obs_prefix}obj_lock': (self.n_obj, 1),
        #                                                 f'{ac_obs_prefix}you_lock': (self.n_agents, self.n_obj, 1),
        #                                                 f'{ac_obs_prefix}team_lock': (self.n_agents, self.n_obj, 1)})
        self.lock_radius = radius_multiplier*self.metadata['box_size']
        self.obj_locked = np.zeros((self.n_obj,), dtype=int)

    def observation(self, obs):
        obs[f'{self.ac_obs_prefix}obj_lock'] = self.obj_locked[:, None]
        you_lock = np.arange(self.n_agents)[:, None] == self.which_locked[None, :]
        obs[f'{self.ac_obs_prefix}you_lock'] = np.expand_dims(you_lock * obs[f'{self.ac_obs_prefix}obj_lock'].T, axis=-1)
        obs[f'{self.ac_obs_prefix}team_lock'] = np.zeros((self.n_agents, self.n_obj, 1))
        for team in np.unique(self.metadata['team_index']):
            team_mask = self.metadata['team_index'] == team
            obs[f'{self.ac_obs_prefix}team_lock'][team_mask] = np.any(obs[f'{self.ac_obs_prefix}you_lock'][team_mask], 0)
        return obs

    def reset(self):
        obs = self.env.reset()
        sim = self.unwrapped.sim

        if self.obj_in_game_metadata_keys is not None:
            self.actual_body_slice = np.concatenate([self.metadata[k] for k in self.obj_in_game_metadata_keys])
        else:
            self.actual_body_slice = np.ones((len(self.body_names))).astype(np.bool)

        actual_body_names = list(compress(self.body_names, self.actual_body_slice))
        self.n_obj = len(actual_body_names)

        # Cache ids
        self.obj_body_idxs = np.array([sim.model.body_name2id(body_name) for body_name in actual_body_names])
        self.obj_jnt_idxs = [np.where(sim.model.jnt_bodyid == body_idx)[0] for body_idx in self.obj_body_idxs]
        self.obj_geom_ids = [np.where(sim.model.geom_bodyid == body_idx)[0] for body_idx in self.obj_body_idxs]
        self.agent_body_idxs = np.array([sim.model.body_name2id(f"agent{i}:particle") for i in range(self.n_agents)])
        self.agent_body_idxs = self.agent_body_idxs[self.agent_idx_allowed_to_lock]
        self.agent_geom_ids = np.array([sim.model.geom_name2id(f'agent{i}:agent') for i in range(self.n_agents)])
        self.agent_geom_ids = self.agent_geom_ids[self.agent_idx_allowed_to_lock]

        self.unlock_objs()
        self.obj_locked = np.zeros((self.n_obj,), dtype=bool)
        self.which_locked = np.zeros((self.n_obj,), dtype=int)

        if self.agent_allowed_to_lock_keys is not None:
            self.agent_allowed_to_lock_mask = np.concatenate([obs[k] for k in self.agent_allowed_to_lock_keys])
        else:
            self.agent_allowed_to_lock_mask = np.ones((self.n_agents, self.n_obj))

        return self.observation(obs)

    def lock_obj(self, action_lock):
        '''
            Implements object gluing for all agents
            Args:
                lock: (n_agent, n_obj) boolean matrix
        '''
        sim = self.unwrapped.sim
        action_lock = action_lock[self.agent_idx_allowed_to_lock]
        action_lock = action_lock[:, self.actual_body_slice]
        agent_pos = sim.data.body_xpos[self.agent_body_idxs]
        obj_pos = sim.data.body_xpos[self.obj_body_idxs]

        obj_width = sim.model.geom_size[np.concatenate(self.obj_geom_ids)]
        obj_quat = sim.data.body_xquat[self.obj_body_idxs]
        assert len(obj_width) == len(obj_quat), (
            "Number of object widths must be equal to number of quaternions for direct distance calculation method. " +
            "This might be caused by a body that contains several geoms.")
        obj_dist = dist_pt_to_cuboid(agent_pos, obj_pos, obj_width, obj_quat)

        allowed_and_desired = np.logical_and(action_lock, obj_dist <= self.lock_radius)
        allowed_and_desired = np.logical_and(allowed_and_desired, self.agent_allowed_to_lock_mask)
        allowed_and_not_desired = np.logical_and(1 - action_lock, obj_dist <= self.lock_radius)
        allowed_and_not_desired = np.logical_and(allowed_and_not_desired, self.agent_allowed_to_lock_mask)

        # objs_to_lock should _all_ be locked this round. new_objs_to_lock are objs that were not locked last round
        # objs_to_unlock are objs that no one wants to lock this round
        if self.lock_type == "any_lock":  # If any agent wants to lock, the obj becomes locked
            objs_to_lock = np.any(allowed_and_desired, axis=0)
            objs_to_unlock = np.logical_and(np.any(allowed_and_not_desired, axis=0), ~objs_to_lock)
            new_objs_to_lock = np.logical_and(objs_to_lock, ~self.obj_locked)
        elif self.lock_type == "all_lock":  # All agents that are close enough must want to lock the obj
            objs_to_unlock = np.any(allowed_and_not_desired, axis=0)
            objs_to_lock = np.logical_and(np.any(allowed_and_desired, axis=0), ~objs_to_unlock)
            new_objs_to_lock = np.logical_and(objs_to_lock, ~self.obj_locked)
        elif self.lock_type == "any_lock_specific":  # If any agent wants to lock, the obj becomes locked
            allowed_to_unlock = np.arange(self.n_agents)[:, None] == self.which_locked[None, :]  # (n_agent, n_obj)
            allowed_to_unlock = np.logical_and(allowed_to_unlock, self.obj_locked[None, :])  # Can't unlock an obj that isn't locked
            allowed_and_not_desired = np.logical_and(allowed_to_unlock[self.agent_idx_allowed_to_lock],
                                                     allowed_and_not_desired)
            objs_to_unlock = np.any(allowed_and_not_desired, axis=0)
            objs_to_lock = np.any(allowed_and_desired, axis=0)
            objs_to_relock = np.logical_and(objs_to_unlock, objs_to_lock)
            new_objs_to_lock = np.logical_and(np.logical_and(objs_to_lock, ~objs_to_relock), ~self.obj_locked)
            objs_to_unlock = np.logical_and(objs_to_unlock, ~objs_to_lock)

            for obj in np.argwhere(objs_to_relock)[:, 0]:
                self.which_locked[obj] = np.random.choice(self.agent_idx_allowed_to_lock[
                            np.argwhere(allowed_and_desired[:, obj]).flatten()])
        elif self.lock_type == "all_lock_team_specific":
            # all close agents must want to lock the object
            # only agents from the same team as the locker can unlock
            allowed_to_unlock = self.metadata['team_index'][:, None] == (
                                self.metadata['team_index'][None, self.which_locked])
            allowed_and_not_desired = np.logical_and(allowed_to_unlock[self.agent_idx_allowed_to_lock], allowed_and_not_desired)
            objs_to_unlock = np.any(allowed_and_not_desired, axis=0)
            objs_to_lock = np.logical_and(np.any(allowed_and_desired, axis=0), ~objs_to_unlock)
            new_objs_to_lock = np.logical_and(objs_to_lock, ~self.obj_locked)
        else:
            assert False, f"{self.lock_type} lock type is not implemented"

        joints_to_unlock = np.isin(sim.model.jnt_bodyid, self.obj_body_idxs[objs_to_unlock])
        joints_to_lock = np.isin(sim.model.jnt_bodyid, self.obj_body_idxs[new_objs_to_lock])

        # Turn on/off emission and joint limit
        matids_to_darken = sim.model.geom_matid[np.isin(sim.model.geom_bodyid, self.obj_body_idxs[objs_to_unlock])]
        matids_to_lighten = sim.model.geom_matid[np.isin(sim.model.geom_bodyid, self.obj_body_idxs[new_objs_to_lock])]
        matids_to_darken = matids_to_darken[matids_to_darken != -1]
        matids_to_lighten = matids_to_lighten[matids_to_lighten != -1]
        sim.model.mat_emission[matids_to_darken] = 0
        sim.model.mat_emission[matids_to_lighten] = 1
        sim.model.jnt_limited[joints_to_unlock] = 0
        sim.model.jnt_limited[joints_to_lock] = 1

        # For objs we need to newly lock, set the joint ranges to the current q of the obj.
        for obj in np.argwhere(new_objs_to_lock)[:, 0]:
            sim.model.jnt_range[self.obj_jnt_idxs[obj], :] = sim.data.q[self.obj_jnt_idxs[obj], None]
            self.which_locked[obj] = np.random.choice(self.agent_idx_allowed_to_lock[
                        np.argwhere(allowed_and_desired[:, obj]).flatten()])

        self.obj_locked = np.logical_or(np.logical_and(self.obj_locked, ~objs_to_unlock), objs_to_lock)

    def unlock_objs(self):
        sim = self.unwrapped.sim
        joints_to_unlock = np.isin(sim.model.jnt_bodyid, self.obj_body_idxs[np.arange(self.n_obj)])
        objs_to_darken = np.isin(sim.model.geom_bodyid, self.obj_body_idxs)
        sim.model.mat_emission[sim.model.geom_matid[objs_to_darken]] = 0
        sim.model.jnt_limited[joints_to_unlock] = 0

    def step(self, action):
        self.lock_obj(action[f'action_{self.ac_obs_prefix}glue'])
        obs, rew, done, info = self.env.step(action)

        if self.agent_allowed_to_lock_keys is not None:
            self.agent_allowed_to_lock_mask = np.concatenate([obs[k] for k in self.agent_allowed_to_lock_keys])
        else:
            self.agent_allowed_to_lock_mask = np.ones((self.n_agents, self.n_obj))

        return self.observation(obs), rew, done, info


class LockAllWrapper(gym.ActionWrapper):
    '''
        Allows agents to lock all boxes. This wrapper introduces an action that
        overrides box-individual gluing action to be activated.
        Args:
            remove_object_specific_lock (bool): If true, removes the agent's ability to
                selectively lock individual boxes.
    '''
    def __init__(self, env, remove_object_specific_lock=False):
        super().__init__(env)
        self.n_agents = self.unwrapped.n_agents
        self.lock_actions = [k for k in self.action_space.spaces.keys() if 'glue' in k]
        self.n_obj = {k: len(self.action_space.spaces[k].spaces[0].nvec) for k in self.lock_actions}
        self.action_space.spaces['action_glueall'] = (
            Tuple([Discrete(2) for _ in range(self.n_agents)]))
        if remove_object_specific_lock:
            for k in self.lock_actions:
                del self.action_space.spaces[k]

    def action(self, action):
        for k in self.lock_actions:
            action[k] = np.zeros((self.n_agents, self.n_obj[k]))
            action[k][action['action_glueall'] == 1, :] = 1

        return action
