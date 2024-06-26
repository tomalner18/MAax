import brax
from brax.envs import Env as BEnv
# Import
from brax import base
from brax.envs.env import PipelineEnv
from brax.io import mjcf

from typing import Any, Dict, Optional

import numpy as np
import logging

import re

from flax import struct

import jax
import jax.numpy as jp

from worldgen import Floor, WorldBuilder, WorldParams
from maax.modules.agents import Agents
from maax.modules.walls import RandomWalls
from maax.modules.objects import Boxes, Ramps



@struct.dataclass
class State:
    """MAax Environment state for training and inference."""
    pipeline_state: Optional[base.State]
    obs: jp.ndarray
    d_obs: Dict[str, jp.ndarray]
    reward: jp.ndarray
    done: jp.ndarray
    step: int = 0
    metrics: Dict[str, jp.ndarray] = struct.field(default_factory=dict)
    info: Dict[str, Any] = struct.field(default_factory=dict),


class Base(PipelineEnv):
    '''
        Multi-agent Base Environment.
        Args:
            horizon (int): Number of steps agent gets to act
            n_frames (int): the number of times to step the physics pipeline for each
                environment step
            n_agents (int): number of agents in the environment
            floor_size (float or (float, float)): size of the floor. If a list of 2 floats, the floorsize
                will be randomized between them on each episode
            grid_size (int): size of the grid that we'll use to place objects on the floor
            deterministic_mode (bool): if True, seeds are incremented rather than randomly sampled.
    '''
    def __init__(
        self, 
        horizon=100, 
        n_frames=15, 
        n_agents=2, 
        floor_size=6., 
        grid_size=30,
        action_lims=(-1.0, 1.0),
        deterministic_mode=False, 
        seed=1,
        backend='generalized',
        **kwargs):

        sys = None

        super().__init__(sys=sys, backend=backend, n_frames=n_frames, **kwargs)

        self.n_agents = n_agents
        self.metadata = {}
        self.metadata['n_actors'] = n_agents
        self.metadata['n_agents'] = n_agents
        self.horizon = horizon
        self.floor_size = floor_size
        if not isinstance(floor_size, (tuple, list, np.ndarray)):
            self.floor_size_dist = [floor_size, floor_size]
        else:
            self.floor_size_dist = floor_size
        self.grid_size = grid_size

        self.placement_grid = np.zeros((grid_size, grid_size))
        self.modules = []
        self.q_indices = dict()
        self.qd_indices = dict()
        self.deterministic_mode = deterministic_mode

        # Required for worldgen
        self._random_state = np.random.RandomState(seed)

    def add_module(self, module):
        self.modules.append(module)

    def _get_d_obs(self, pipeline_state: base.State) -> jp.ndarray:
        '''
            Returns the environment observation.
            Loops through modules, calls their observation_step functions, and
                adds the result to the observation dictionary.
        '''
        d_obs = {}
        for module in self.modules:
            # obs = jp.concatenate((obs, module.observation_step(pipeline_state)))
            d_obs.update(module.observation_step(pipeline_state))
        return d_obs

    def _set_joint_ranges(self):
        '''
            Sets unlimited joint ranges for all joints in the system.
        '''
        lower_bound = jp.full(self.sys.dof.limit[0].shape, -jp.inf)
        upper_bound = jp.full(self.sys.dof.limit[1].shape, jp.inf)
        
        # Set sys.dof.limit[0] to  at all occurences of -1
        lower_bound = jp.where(self.sys.dof.limit[0] == -1, self.init_q[:lower_bound.size], lower_bound)
        upper_bound = jp.where(self.sys.dof.limit[1] == 1, self.init_q[:upper_bound.size], upper_bound)

        self.sys = self.sys.replace(dof=self.sys.dof.replace(limit=(lower_bound, upper_bound)))
        

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
        xml, self.init_dict, udd_callback = self._get_xml(seed)
        with open("simple.xml", "w") as f:
            f.write(xml)
        self.sys = mjcf.loads(xml)

        self.init_q = jp.hstack(list(self.init_dict.values()))
        self.init_qd = jp.zeros(self.sys.qd_size())

        # Store the joint indices for manipulation in observation step
        self._store_joint_indices(self.init_dict)
        
        self._set_joint_ranges()

        # Cache the joint data in the modules for observation steps
        for module in self.modules:
            module.cache_step(self)

    def _concat_obs(self, d_obs):
        '''
            Concatenates the observation dictionary into the Brax obs array
        '''
        obs = jp.concatenate([jp.ravel(v) for v in d_obs.values()])
        return obs


    def _get_xml(self, seed):
        '''
            Calls build_step and then modify_sim_step for each module. If
            a build_step failed, then restarts.
        '''
        self.floor_size = np.random.uniform(self.floor_size_dist[0], self.floor_size_dist[1])
        self.metadata['floor_size'] = self.floor_size
        world_params = WorldParams(size=(self.floor_size, self.floor_size, 2.5),
                                   num_substeps=self._n_frames)
        successful_placement = False
        failures = 0
        while not successful_placement:
            # if (failures + 1) % 10 == 0:
            #     logging.warning(f"Failed {failures} times in creating environment")
            builder = WorldBuilder(world_params, seed)
            floor = Floor()

            builder.append(floor)

            self.placement_grid = np.zeros((self.grid_size, self.grid_size))

            successful_placement = np.all([module.build_step(self, floor, self.floor_size)
                                           for module in self.modules])
            failures += 1

        return builder.get_xml()
  


    def set_info(self) -> Dict[str, Any]:
        """Sets the environment info."""
        return {'in_prep_phase': True,
                'step_count': 0}

    def reset(self, rng: jp.ndarray) -> State:
        """Resets the environment to an initial state."""

        init_q = jp.hstack(list(self.init_dict.values()))
        init_qd = jp.zeros(self.sys.qd_size())

        pipeline_state = self.pipeline_init(self.init_q, self.init_qd)
        d_obs = self._get_d_obs(pipeline_state)
        obs = self._concat_obs(d_obs)
        reward = jp.zeros(shape=(self.n_agents,))
        done, zero = jp.zeros(2)
        info = self.set_info()
        metrics = {}
        return State(pipeline_state=pipeline_state, obs=obs, d_obs=d_obs, reward=reward, done=done, metrics=metrics, info=info)


    def step(self, state: State, action: jp.ndarray) -> State:
        """Run one timestep of the environment's dynamics.
        Args:
            state: current state of the environment.
            action: action to take in the environment
        """
        pipeline_state0 = state.pipeline_state

        pipeline_state = self.pipeline_step(pipeline_state0, jp.ravel(action))

        d_obs = self._get_d_obs(pipeline_state)
        obs = self._concat_obs(d_obs)

        step = state.step + 1

        return state.replace(pipeline_state=pipeline_state, obs=obs, d_obs=d_obs, step=step)

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



