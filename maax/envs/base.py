import brax
from brax.envs import Env as BEnv
from brax.envs import State

import numpy as np
import logging

import jax
import jax.numpy as jp

from mujoco_worldgen import Floor, WorldBuilder, WorldParams, Env
from mae_envs.modules.agents import Agents
from mae_envs.modules.walls import RandomWalls
from mae_envs.modules.objects import Boxes, Ramps

class Base(BEnv):
    '''
        Multi-agent Base Environment.
        Args:
            horizon (int): Number of steps agent gets to act
            n_substeps (int): Number of internal mujoco steps per outer environment step;
                essentially this is action repeat.
            n_agents (int): number of agents in the environment
            floor_size (float or (float, float)): size of the floor. If a list of 2 floats, the floorsize
                will be randomized between them on each episode
            grid_size (int): size of the grid that we'll use to place objects on the floor
            action_lims (float tuple): lower and upper limit of mujoco actions
            deterministic_mode (bool): if True, seeds are incremented rather than randomly sampled.
    '''
    def __init__(self, horizon=250, n_substeps=5, n_agents=2,
                 floor_size=6., grid_size=30,
                 action_lims=(-1.0, 1.0), deterministic_mode=False, seed=1,
                 **kwargs):
        self.n_agents = n_agents
        self.metadata = {}
        self.metadata['n_actors'] = n_agents
        self.horizon = horizon
        self.n_substeps = n_substeps
        if not isinstance(floor_size, (tuple, list, np.ndarray)):
            self.floor_size_dist = [floor_size, floor_size]
        else:
            self.floor_size_dist = floor_size
        self.grid_size = grid_size
        self.kwargs = kwargs
        self.placement_grid = np.zeros((grid_size, grid_size))
        self.modules = []

        # Required as mujoco_worldgen
        self._random_state = np.random.RandomState(seed)

    def add_module(self, module):
        self.modules.append(module)

    def _get_obs(self, sim):
        '''
            Loops through modules, calls their observation_step functions, and
                adds the result to the observation dictionary.
        '''
        obs = {}
        for module in self.modules:
            obs.update(module.observation_step(self, self.sim))
        return obs

    def _get_xml(self, seed):
        '''
            Calls build_world_step and then modify_sim_step for each module. If
            a build_world_step failed, then restarts.
        '''
        self.floor_size = np.random.uniform(self.floor_size_dist[0], self.floor_size_dist[1])
        self.metadata['floor_size'] = self.floor_size
        world_params = WorldParams(size=(self.floor_size, self.floor_size, 2.5),
                                   num_substeps=self.n_substeps)
        successful_placement = False
        failures = 0
        while not successful_placement:
            if (failures + 1) % 10 == 0:
                logging.warning(f"Failed {failures} times in creating environment")
            builder = WorldBuilder(world_params, seed)
            floor = Floor()

            builder.append(floor)

            self.placement_grid = np.zeros((self.grid_size, self.grid_size))

            successful_placement = np.all([module.build_world_step(self, floor, self.floor_size)
                                           for module in self.modules])
            failures += 1

        return builder.get_xml()

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""

    def step(self, state: State, action: jp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""

    @property
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""

    @property
    def action_size(self) -> int:
        """The size of the action vector expected by step."""

    @property
    def backend(self) -> str:
        """The physics backend that this env was instantiated with."""

