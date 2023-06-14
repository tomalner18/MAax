import jax
from jax import numpy as jp
from scipy.linalg import circulant
from gym.spaces import Tuple, Box, Dict
from copy import deepcopy
from mae_envs.wrappers.util import MWrapper, RewardWrapper, ActionWrapper, ObservationWrapper


class SplitMultiAgentActions(ActionWrapper):
    '''
        Splits generated actions into a dict of tuple actions.
    '''
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = self.metadata['n_actors']
        # lows = jp.split(self.action_space.low, self.n_agents)
        # highs = jp.split(self.action_space.high, self.n_agents)
        # self.action_space = Dict({
        #     'action_movement': Tuple([Box(low=low, high=high, dtype=self.action_space.dtype)
        #                               for low, high in zip(lows, highs)])
        # })

    def action(self, state, action):
        return action['action_movement'].ravel()


class JoinMultiAgentActions(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_agents = self.metadata['n_actors']
        low = jp.concatenate([space.low for space in self.action_space.spaces])
        high = jp.concatenate([space.high for space in self.action_space.spaces])
        self.action_space = Box(low=low, high=high, dtype=self.action_space.spaces[0].dtype)

    def action(self, state, action):
        # action should be a tuple of different agent actions
        return jp.split(action, self.n_agents)


class SplitObservations(ObservationWrapper):
    """
        Split observations for each agent.
        Args:
            keys_self: list of observation names which are agent specific. E.g. this will
                    permute qpos such that each agent sees its own qpos as the first numbers
            keys_copy: list of observation names that are just passed down as is
            keys_self_matrices: list of observation names that should be (n_agent, n_agent, dim) where
                each agent has a custom observation of another agent. This is different from self_keys
                in that self_keys we assume that observations are symmetric, whereas these can represent
                unique pairwise interactions/observations
    """
    def __init__(self, env, keys_self, keys_copy=[], keys_self_matrices=[]):
        super().__init__(env)
        self.keys_self = sorted(keys_self)
        self.keys_copy = sorted(keys_copy)
        self.keys_self_matrices = sorted(keys_self_matrices)
        self.n_agents = self.metadata['n_agents']

    def observation(self, state):
        '''
        TODO: Implement observation splitting and reshaping (n_agents, n_objects, )
        '''
        # new_obs = {}
        # for k, v in obs.items():
        #     # Masks that aren't self matrices should just be copied
        #     if 'mask' in k and k not in self.keys_self_matrices:
        #         new_obs[k] = obs[k]
        #     # Circulant self matrices
        #     elif k in self.keys_self_matrices:
        #         new_obs[k] = self._process_self_matrix(obs[k])
        #     # Circulant self keys
        #     elif k in self.keys_self:
        #         new_obs[k + '_self'] = obs[k]
        #         new_obs[k] = obs[k][circulant(jp.arange(self.n_agents))]
        #         new_obs[k] = new_obs[k][:, 1:, :]  # Remove self observation
        #     elif k in self.keys_copy:
        #         new_obs[k] = obs[k]
        #     # Everything else should just get copied for each agent (e.g. external obs)
        #     else:
        #         new_obs[k] = jp.tile(v, self.n_agents).reshape([v.shape[0], self.n_agents, v.shape[1]]).transpose((1, 0, 2))

        # return new_obs
        return state.d_obs

    def _process_self_matrix(self, self_matrix):
        '''
            self_matrix will be a (n_agent, n_agent) boolean matrix. Permute each row such that the matrix is consistent with
                the circulant permutation used for self observations. E.g. this should be used for agent agent masks
        '''
        assert jp.all(self_matrix.shape[:2] == jp.array((self.n_agents, self.n_agents))), \
            f"The first two dimensions of {self_matrix} were not (n_agents, n_agents)"

        new_mat = self_matrix.copy()
        # Permute each row to the right by one more than the previous
        # E.g., [[1,2],[3,4]] -> [[1,2],[4,3]]
        idx = circulant(jp.arange(self.n_agents))
        new_mat = new_mat[jp.arange(self.n_agents)[:, None], idx]
        new_mat = new_mat[:, 1:]  # Remove self observation
        return new_mat


class SelectKeysWrapper(ObservationWrapper):
    """
        Select keys for final observations.
        Expects that all observations come in shape (n_agents, n_objects, n_dims)
        Args:
            keys_self (list): observation names that are specific to an agent
                These will be concatenated into 'observation_self' observation
            keys_other (list): observation names that should be passed through
            flatten (bool): if true, internal and external observations
    """

    def __init__(self, env, keys_self, keys_other, flatten=False):
        super().__init__(env)
        self.keys_self = sorted([k + '_self' for k in keys_self])
        self.keys_other = sorted(keys_other)
        self.flatten = flatten

        # Change observation space to look like a single agent observation space.
        # This makes constructing policies much easier
        # if flatten:
        #     size_self = sum([jp.prod(self.env.observation_space.spaces[k].shape[1:])
        #                      for k in self.keys_self + self.keys_other])
        #     self.observation_space = Dict(
        #         {'observation_self': Box(-jp.inf, jp.inf, (size_self,), jp.float32)})
        # else:
        #     size_self = sum([self.env.observation_space.spaces[k].shape[1]
        #                      for k in self.keys_self])
        #     obs_self = {'observation_self': Box(-jp.inf, jp.inf, (size_self,), jp.float32)}
        #     obs_extern = {k: Box(-jp.inf, jp.inf, v.shape[1:], jp.float32)
        #                   for k, v in self.observation_space.spaces.items()
        #                   if k in self.keys_other}
        #     obs_self.update(obs_extern)
        #     self.observation_space = Dict(obs_self)

    def observation(self, state):
        # if self.flatten:
        #     other_obs = [observation[k].reshape((observation[k].shape[0], -1))
        #                  for k in self.keys_other]
        #     obs = jp.concatenate([observation[k] for k in self.keys_self] + other_obs, axis=-1)
        #     return {'observation_self': obs}
        # else:
        #     obs = jp.concatenate([observation[k] for k in self.keys_self], -1)
        #     obs = {'observation_self': obs}
        #     other_obs = {k: v for k, v in observation.items() if k in self.keys_other}
        #     obs.update(other_obs)
        #     return obs
        return state.d_obs
