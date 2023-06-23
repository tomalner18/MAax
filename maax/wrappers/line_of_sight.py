import jax
from jax import numpy as jp
from maax.util.vision import insight, in_cone2d, caught
from maax.wrappers.util import ObservationWrapper


class AgentAgentContactMask2D(ObservationWrapper):
    """
    Adds an mask observation that states which agents are in contact with which agents.
    Args:
        distance_threshold: (float) the distance below which agents are considered in contact
    """
    def __init__(self, env, distance_threshold=3):
        super().__init__(env)
        self.distance_threshold = distance_threshold
        self.n_agents = self.unwrapped.n_agents
    
    def observation(self, state):
        # Agent to agent contact mask
        d_obs = state.d_obs
        agent_pos2d = d_obs['agent_pos'][:, :-1]
        contact_mask = caught(agent_pos2d, self.distance_threshold)

        d_obs['mask_aa_con'] = contact_mask
        return d_obs

class AgentAgentObsMask2D(ObservationWrapper):
    """ Adds an mask observation that states which agents are visible to which agents.
        Args:
            cone_angle: (float) the angle in radians btw the axis and edge of the observation cone
    """
    def __init__(self, env, cone_angle=3/8 * jp.pi):
        super().__init__(env)
        self.cone_angle = cone_angle
        self.n_agents = self.unwrapped.n_agents
        # self.observation_space = update_obs_space(env, {'mask_aa_obs': (self.n_agents, self.n_agents)})

    def observation(self, state):
        # Agent to agent obs mask
        d_obs = state.d_obs
        agent_pos2d = d_obs['agent_pos'][:, :-1]
        agent_angle = d_obs['agent_angle']
        cone_mask = in_cone2d(agent_pos2d, jp.squeeze(agent_angle, -1), self.cone_angle, agent_pos2d)
        # Make sure they are in line of sight
        for i, j in jp.argwhere(cone_mask):
            if i != j:
                cone_mask[i, j] = insight(self.unwrapped.sim,
                                          self.metadata['agent_geom_idxs'][i],
                                          self.metadata['agent_geom_idxs'][j])
        d_obs['mask_aa_obs'] = cone_mask
        return d_obs


class AgentSiteObsMask2D(ObservationWrapper):
    """ Adds an mask observation that states which sites are visible to which agents.
        Args:
            pos_obs_key: (string) the name of the site position observation of shape (n_sites, 3)
            mask_obs_key: (string) the name of the mask observation to output
            cone_angle: (float) the angle in radians btw the axis and edge of the observation cone
    """
    def __init__(self, env, pos_obs_key, mask_obs_key, cone_angle=3/8 * jp.pi):
        super().__init__(env)
        self.cone_angle = cone_angle
        self.n_agents = self.unwrapped.n_agents
        assert(self.n_agents == self.observation_space.spaces['agent_pos'].shape[0])
        self.n_objects = self.observation_space.spaces[pos_obs_key].shape[0]
        # self.observation_space = update_obs_space(env, {mask_obs_key: (self.n_agents, self.n_objects)})
        self.pos_obs_key = pos_obs_key
        self.mask_obs_key = mask_obs_key

    def observation(self, state):
        d_obs = state.d_obs
        agent_pos2d = d_obs['agent_pos'][:, :-1]
        agent_angle = d_obs['agent_angle']
        pos2d = d_obs[self.pos_obs_key][:, :2]
        cone_mask = in_cone2d(agent_pos2d, jp.squeeze(agent_angle, -1), self.cone_angle, pos2d)
        # Make sure they are in line of sight
        for i, j in jp.argwhere(cone_mask):
            agent_geom_id = self.metadata['agent_geom_idxs'][i]
            pt2 = d_obs[self.pos_obs_key][j]
            cone_mask[i, j] = insight(self.unwrapped.sim, agent_geom_id, pt2=pt2)
        d_obs[self.mask_obs_key] = cone_mask
        return d_obs


class AgentGeomObsMask2D(ObservationWrapper):
    """ Adds an mask observation that states which geoms are visible to which agents.
        Args:
            pos_obs_key: (string) the name of the site position observation of shape (n_geoms, 3)
            geom_idxs_obs_key: (string) the name of an observation that, for each object to be
                                masked, gives the Mujoco index of the geom (e.g. in sim.geom_names)
                                as an array of shape (n_geoms, 1)
            mask_obs_key: (string) the name of the mask observation to output
            cone_angle: (float) the angle in radians btw the axis and edge of the observation cone
    """
    def __init__(self, env, pos_obs_key, geom_idxs_obs_key, mask_obs_key, cone_angle=3/8 * jp.pi):
        super().__init__(env)
        self.cone_angle = cone_angle
        self.n_agents = self.unwrapped.n_agents
        # self.observation_space = update_obs_space(env, {mask_obs_key: (self.n_agents, self.n_objects)})
        self.pos_obs_key = pos_obs_key
        self.mask_obs_key = mask_obs_key
        self.geom_idxs_obs_key = geom_idxs_obs_key

    def observation(self, state):
        d_obs = state.d_obs
        agent_pos2d = d_obs['agent_pos'][:, :-1]
        agent_angle = d_obs['agent_angle']
        pos2d = d_obs[self.pos_obs_key][:, :2]
        cone_mask = in_cone2d(agent_pos2d, jp.squeeze(agent_angle, -1), self.cone_angle, pos2d)
        # Make sure they are in line of sight
        for i, j in jp.argwhere(cone_mask):
            agent_geom_id = self.metadata['agent_geom_idxs'][i]
            geom_id = d_obs[self.geom_idxs_obs_key][j, 0]
            if geom_id == -1:
                # This option is helpful if the number of geoms varies between episodes
                # If geoms don't exists this wrapper expects that the geom idx is
                # set to -1
                cone_mask[i, j] = 0
            else:
                cone_mask[i, j] = insight(self.unwrapped.sim, agent_geom_id, geom2_id=geom_id)
        d_obs[self.mask_obs_key] = cone_mask
        return d_obs
