import brax
from brax.envs import Env as BEnv
# Import
from brax import base
from brax.envs import State
from brax.envs.env import PipelineEnv
from brax.io import mjcf
# Import PipelineEnv

import numpy as np
import logging

import re

import jax
import jax.numpy as jp

from mujoco_worldgen import Floor, WorldBuilder, WorldParams, Env
from mae_envs.modules.agents import Agents
from mae_envs.modules.walls import RandomWalls
from mae_envs.modules.objects import Boxes, Ramps

class Base(PipelineEnv):
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
    def __init__(
        self, 
        horizon=250, 
        n_substeps=5, 
        n_agents=2, 
        floor_size=6., 
        grid_size=30,
        action_lims=(-1.0, 1.0), 
        deterministic_mode=False, 
        seed=1,
        backend='generalized',
        **kwargs):

        sys = None

        super().__init__(sys=sys, backend=backend, **kwargs)
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
        self.q_indices = dict()
        self.qd_indices = dict()

        # Required as mujoco_worldgen
        self._random_state = np.random.RandomState(seed)

    def add_module(self, module):
        self.modules.append(module)

    def _get_obs(self, pipeline_state: base.State) -> jp.ndarray:
        '''
            Returns the environment observation.
            Loops through modules, calls their observation_step functions, and
                adds the result to the observation dictionary.
        '''
        # for module in self.modules:
        #     obs.update(module.observation_step(self, self.sim))
        # return obs
        return jp.concatenate((pipeline_state.q, pipeline_state.qd))


    def _store_joint_indices(self, init_dict):
        '''
            Stores the mapping from joint name to joint indices in the brax system
        '''
        q_index = 0
        qd_index = 0
        for (k, v) in init_dict.items():
            body, joint = k.split('_')
            b_class = re.sub('\d+', '', body)
            joint = re.sub('\d+', '', joint)
            v = jp.asarray(v)


            # Q Assignment: based on class
            if b_class in self.q_indices:
                self.q_indices[b_class] = jp.concatenate((self.q_indices[b_class], jp.arange(q_index, q_index + v.size)))

            else:
                self.q_indices[b_class] = jp.arange(q_index, q_index + v.size)
            
            # QD Assignment: based on class and joint type
            if b_class in self.qd_indices:
                if joint == "slide" or joint == "hinge":
                    self.qd_indices[b_class] = jp.concatenate((self.qd_indices[b_class], jp.arange(qd_index, qd_index + 1)))
                    qd_index += 1
                elif joint == "free":
                    self.qd_indices[b_class] = jp.concatenate((self.qd_indices[b_class], jp.arange(qd_index, qd_index + 6)))
                    qd_index += 6
            else:
                if joint == "slide" or joint == "hinge":
                    self.qd_indices[b_class] = jp.arange(qd_index, qd_index + 1)
                    qd_index += 1
                elif joint == "free":
                    self.qd_indices[b_class] = jp.arange(qd_index, qd_index + 6)
                    qd_index += 6

            q_index += v.size

    def gen_sys(self, seed):
        '''
            Generates the brax system from the random seed.
            Then populates the q and qp indices for each module.
        '''
        xml, init_dict, udd_callback = self._get_xml(seed)
        self.sys = mjcf.loads(xml)


        # init_q = jp.asarray(list(init_dict.values()))
        self.init_q = jp.hstack(list(init_dict.values()))

        # print('Init from joint positions: ', init_q)
        self.init_qd = jp.zeros(self.sys.qd_size())

        # with open("simple.xml", "w") as f:
        #     f.write(xml)

        # Store the joint indices for manipulation in observation step
        self._store_joint_indices(init_dict)

        # Cache the joint data in the modules for observation steps
        for module in self.modules:
            module.cache_step(self)


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

        pipeline_state = self.pipeline_init(self.init_q, self.init_qd)
        obs = self._get_obs(pipeline_state)
        reward, done, zero = jp.zeros(3)
        metrics = {}
        return State(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, metrics=metrics)


    def step(self, state: State, action: jp.ndarray) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # obs = self._get_obs(pipeline_state)

        return state.replace(pipeline_state=pipeline_state)

        # return state.replace(pipeline_state=pipeline_state, obs=obs)

    @property
    def dt(self) -> jp.ndarray:
        """The timestep used for each env step."""
        return self.sys.dt * self._n_frames

    @property
    def observation_size(self) -> int:
        rng = jax.random.PRNGKey(0)
        reset_state = self.unwrapped.reset(rng)
        return reset_state.obs.shape[-1]

    @property
    def action_size(self) -> int:
        return self.sys.act_size()

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def unwrapped(self):
        return self



