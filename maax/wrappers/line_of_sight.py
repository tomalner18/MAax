import jax
from jax import numpy as jp
from maax.util.vision import caught
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



