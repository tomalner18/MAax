import numpy as np
from worldgen.util.types import store_args
from worldgen.util.sim_funcs import (qpos_idxs_from_joint_prefix,
                                            qvel_idxs_from_joint_prefix)
from worldgen.transforms import set_geom_attr_transform
from worldgen.util.rotation import normalize_angles
from mae_envs.util.transforms import (add_weld_equality_constraint_transform,
                                      set_joint_damping_transform)
from mae_envs.modules import EnvModule, rejection_placement, get_size_from_xml
from worldgen import ObjFromXML

import jax
from jax import numpy as jp


class Agents(EnvModule):
    '''
        Add Agents to the environment.
        Args:
            n_agents (int): number of agents
            placement_fn (fn or list of fns): See mae_envs.modules.util:rejection_placement for
                spec. If list of functions, then it is assumed there is one function given
                per agent
            color (tuple or list of tuples): rgba for agent. If list of tuples, then it is
                assumed there is one color given per agent
            friction (float): agent friction
            damp_z (bool): if False, reduce z damping to 1
            polar_obs (bool): Give observations about rotation in polar coordinates
    '''
    @store_args
    def __init__(self, n_agents, placement_fn=None, color=None, friction=None,
                 damp_z=False, polar_obs=True):
        pass

    def build_world_step(self, env, floor, floor_size):
        env.metadata['n_agents'] = self.n_agents
        successful_placement = True

        for i in range(self.n_agents):
            env.metadata.pop(f"agent{i}_initpos", None)

        for i in range(self.n_agents):
            obj = ObjFromXML("particle", name=f"agent{i}")
            if self.friction is not None:
                obj.add_transform(set_geom_attr_transform('friction', self.friction))
            if self.color is not None:
                _color = (self.color[i]
                          if isinstance(self.color[0], (list, tuple, np.ndarray))
                          else self.color)
                obj.add_transform(set_geom_attr_transform('rgba', _color))
            if not self.damp_z:
                obj.add_transform(set_joint_damping_transform(1, 'tz'))

            if self.placement_fn is not None:
                _placement_fn = (self.placement_fn[i]
                                 if isinstance(self.placement_fn, list)
                                 else self.placement_fn)
                obj_size = get_size_from_xml(obj)
                pos, pos_grid = rejection_placement(env, _placement_fn, floor_size, obj_size)
                if pos is not None:
                    floor.append(obj, placement_xy=pos)
                    # store spawn position in metadata. This allows sampling subsequent agents
                    # close to previous agents
                    env.metadata[f"agent{i}_initpos"] = pos_grid
                else:
                    successful_placement = False
            else:
                floor.append(obj)
        return successful_placement

    def cache_step(self, env):
        # Cache q, qd idxs
        self.agent_q_idxs = env.q_indices['agent']
        self.agent_qd_idxs = env.qd_indices['agent']
        # self.agent_qpos_idxs = np.array([qpos_idxs_from_joint_prefix(sim, f'agent{i}')
        #                                  for i in range(self.n_agents)])
        # self.agent_qvel_idxs = np.array([qvel_idxs_from_joint_prefix(sim, f'agent{i}')
        #                                 for i in range(self.n_agents)])
        # env.metadata['agent_geom_idxs'] = [sim.model.geom_name2id(f'agent{i}:agent')
        #                                    for i in range(self.n_agents)]

    def observation_step(self, state):
        '''
        NOTE: Currently agents move via 2D translation. Thus they don't have a rotation
        '''
        qs = state.q.copy()
        qds = state.qd.copy()


        agent_q = qs[self.agent_q_idxs]
        agent_qd = qds[self.agent_qd_idxs]
        
        agent_q = jp.reshape(agent_q, newshape=(-1,3))
        agent_qd = jp.reshape(agent_qd, newshape=(-1,3))
        # agent_angle = agent_q[:, [-1]] - np.pi / 2  # Rotate the angle to match visual front
        agent_q_qd = jp.concatenate([agent_q, agent_qd], -1)
        # polar_angle = jp.concatenate([np.cos(agent_angle), np.sin(agent_angle)], -1)
        # if self.polar_obs:
        #     agent_q = jp.concatenate([agent_q[:, :-1], polar_angle], -1)
        # agent_angle = normalize_angles(agent_angle)
        obs = {
            'agent_qpos_qvel': agent_q_qd,
            # 'agent_angle': agent_angle,
            'agent_pos': agent_q}

        # obs = jp.concatenate(agent_q_qd, agent_angle, agent_q[:, :3])
        return obs
