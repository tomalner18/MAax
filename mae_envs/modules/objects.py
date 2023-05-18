import numpy as np
from mujoco_worldgen.util.types import store_args
from mujoco_worldgen.util.sim_funcs import (qpos_idxs_from_joint_prefix,
                                            qvel_idxs_from_joint_prefix)
from mujoco_worldgen import Geom, Material, ObjFromXML
from mujoco_worldgen.transforms import set_geom_attr_transform
from mujoco_worldgen.util.rotation import normalize_angles
from mae_envs.util.transforms import remove_hinge_axis_transform
from mae_envs.modules import EnvModule, rejection_placement, get_size_from_xml

import jax
from jax import numpy as jp


class Boxes(EnvModule):
    '''
    Add moveable boxes to the environment.
        Args:
            n_boxes (int or (int, int)): number of boxes. If tuple of ints, every episode the
                number of boxes is drawn uniformly from range(n_boxes[0], n_boxes[1] + 1)
            n_elongated_boxes (int or (int, int)): Number of elongated boxes. If tuple of ints,
                every episode the number of elongated boxes is drawn uniformly from
                range(n_elongated_boxes[0], min(curr_n_boxes, n_elongated_boxes[1]) + 1)
            placement_fn (fn or list of fns): See mae_envs.modules.util:rejection_placement for spec
                If list of functions, then it is assumed there is one function given per box
            box_size (float): box size
            box_mass (float): box mass
            friction (float): box friction
            box_only_z_rot (bool): If true, boxes can only be rotated around the z-axis
            boxid_obs (bool): If true, the id of boxes is observed
            boxsize_obs (bool): If true, the size of the boxes is observed (note that the size
                is still observed if boxsize_obs is False but there are elongated boxes)
            polar_obs (bool): Give observations about rotation in polar coordinates
            mark_box_corners (bool): If true, puts a site in the middle of each of the 4 vertical
                box edges for each box (these sites are used for calculating distances in the
                blueprint construction task).
            free (bool): If true, boxes have free joints. If false, boxes have slide joints
    '''
    @store_args
    def __init__(self, n_boxes, n_elongated_boxes=0, placement_fn=None,
                 box_size=0.5, box_mass=1.0, friction=None, box_only_z_rot=False,
                 boxid_obs=True, boxsize_obs=False, polar_obs=True, free=False):
        if type(n_boxes) not in [tuple, list, np.ndarray]:
            self.n_boxes = [n_boxes, n_boxes]
        if type(n_elongated_boxes) not in [tuple, list, np.ndarray]:
            self.n_elongated_boxes = [n_elongated_boxes, n_elongated_boxes]

    def build_world_step(self, env, floor, floor_size):
        env.metadata['box_size'] = self.box_size

        self.curr_n_boxes = env._random_state.randint(self.n_boxes[0], self.n_boxes[1] + 1)

        env.metadata['curr_n_boxes'] = np.zeros((self.n_boxes[1]))
        env.metadata['curr_n_boxes'][:self.curr_n_boxes] = 1
        env.metadata['curr_n_boxes'] = env.metadata['curr_n_boxes'].astype(bool)

        self.curr_n_elongated_boxes = env._random_state.randint(
            self.n_elongated_boxes[0], min(self.n_elongated_boxes[1], self.curr_n_boxes) + 1)

        self.box_size_array = self.box_size * np.ones((self.curr_n_boxes, 3))
        if self.curr_n_elongated_boxes > 0:
            # sample number of x-aligned boxes
            n_xaligned = env._random_state.randint(self.curr_n_elongated_boxes + 1)
            self.box_size_array[:n_xaligned, :] = self.box_size * np.array([3.3, 0.3, 1.0])
            self.box_size_array[n_xaligned:self.curr_n_elongated_boxes, :] = (self.box_size * np.array([0.3, 3.3, 1.0]))
        env.metadata['box_size_array'] = self.box_size_array

        successful_placement = True
        for i in range(self.curr_n_boxes):
            char = chr(ord('A') + i % 26)
            geom = Geom("box", self.box_size_array[i, :], name=f'moveable-box{i}', free=self.free, rgba=[1.0, 0.5, 0.8, 1.0])
            # geom.set_material(Material(texture="chars/" + char + ".png"))
            geom.add_transform(set_geom_attr_transform('mass', self.box_mass))
            if self.friction is not None:
                geom.add_transform(set_geom_attr_transform('friction', self.friction))
            # if self.box_only_z_rot:
            #     geom.add_transform(remove_hinge_axis_transform(np.array([1.0, 0.0, 0.0])))
            #     geom.add_transform(remove_hinge_axis_transform(np.array([0.0, 1.0, 0.0])))
            geom.add_transform(remove_hinge_axis_transform(np.array([1.0, 0.0, 0.0])))
            geom.add_transform(remove_hinge_axis_transform(np.array([0.0, 1.0, 0.0])))
            geom.add_transform(remove_hinge_axis_transform(np.array([0.0, 0.0, 1.0])))

            if self.placement_fn is not None:
                _placement_fn = (self.placement_fn[i]
                                 if isinstance(self.placement_fn, list)
                                 else self.placement_fn)
                pos, _ = rejection_placement(env, _placement_fn, floor_size,
                                             self.box_size_array[i, :2])
                if pos is not None:
                    floor.append(geom, placement_xy=pos)
                else:
                    successful_placement = False
            else:
                floor.append(geom)
        return successful_placement

    def cache_step(self, env):
        # Cache q, qd idxs
        self.box_q_idxs = env.q_indices['moveable-box']
        self.box_qd_idxs = env.qd_indices['moveable-box']



    def observation_step(self, state):
        qs = state.q.copy()
        qds = state.qd.copy()

        print("Box q indices: ", self.box_q_idxs)
        print("Box qd indices: ", self.box_qd_idxs)

        box_inds = jp.expand_dims(jp.arange(self.curr_n_boxes), -1)
        box_qs = qs[self.box_q_idxs]
        box_qds = qds[self.box_qd_idxs]

        print("box_qs: ", box_qs)
        print("box_qds: ", box_qds)
        box_angle = normalize_angles(box_qs[:, 3:])

        #Print shape for debug
        print("box_qs shape: ", box_qs.shape)
        print("box_qds shape: ", box_qds.shape)
        print("box_angle shape: ", box_angle.shape)
        print("box_inds shape: ", box_inds.shape)
        polar_angle = jp.concatenate([np.cos(box_angle), np.sin(box_angle)], -1)
        if self.polar_obs:
            box_qs = jp.concatenate([box_qs[:, :3], polar_angle], -1)
        box_obs = jp.concatenate([box_qs, box_qds], -1)

        if self.boxid_obs:
            box_obs = jp.concatenate([box_obs, box_inds], -1)
        if self.n_elongated_boxes[1] > 0 or self.boxsize_obs:
            box_obs = jp.concatenate([box_obs, self.box_size_array], -1)

        # obs = {'box_obs': box_obs,
        #        'box_angle': box_angle,
        #        'box_geom_idxs': box_geom_idxs,
        #        'box_pos': box_qpos[:, :3],
        #Print obs shape for debug
        print("Box obs shape: ", box_obs.shape)
        print("Box angle shape: ", box_angle.shape)
        print("Box qs shape: ", box_qs.shape)
        obs = jp.concatenate((box_obs, box_angle, box_qs[:, :3]))
        print("Box obs shape: ", obs.shape)
        return obs


class Ramps(EnvModule):
    '''
    Add moveable ramps to the environment.
        Args:
            n_ramps (int): number of ramps
            placement_fn (fn or list of fns): See mae_envs.modules.util:rejection_placement for spec
                If list of functions, then it is assumed there is one function given per ramp
            friction (float): ramp friction
            polar_obs (bool): Give observations about rotation in polar coordinates
            pad_ramp_size (bool): pads 3 rows of zeros to the ramp observation. This makes
                ramp observations match the dimensions of elongated box observations.
    '''
    @store_args
    def __init__(self, n_ramps, placement_fn=None, friction=None, polar_obs=True,
                 pad_ramp_size=False, free=False):
        pass

    def build_world_step(self, env, floor, floor_size):
        successful_placement = True

        env.metadata['curr_n_ramps'] = np.ones((self.n_ramps)).astype(bool)

        for i in range(self.n_ramps):
            char = chr(ord('A') + i % 26)
            if self.free:
                geom = ObjFromXML('ramp', name=f"ramp{i}")
            else:
                geom = ObjFromXML('ramp_slide', name=f"ramp{i}")
            geom.set_material(Material(texture="chars/" + char + ".png"))
            if self.friction is not None:
                geom.add_transform(set_geom_attr_transform('friction', self.friction))

            if self.placement_fn is not None:
                _placement_fn = (self.placement_fn[i]
                                 if isinstance(self.placement_fn, list)
                                 else self.placement_fn)
                pos, _ = rejection_placement(env, _placement_fn, floor_size, get_size_from_xml(geom))
                if pos is not None:
                    floor.append(geom, placement_xy=pos)
                else:
                    successful_placement = False
            else:
                floor.append(geom)
        return successful_placement

    def cache_step(self, env):
        # Cache q, qd indices
        self.ramp_q_idxs = env.q_indices['ramp']
        self.ramp_qd_idxs = env.qd_indices['ramp']


    def observation_step(self, state):
        qs = state.q.copy()
        qds = state.qd.copy()

        ramp_qs = qs[self.ramp_q_idxs]
        ramp_qds = qds[self.ramp_qd_idxs]
        ramp_angle = normalize_angles(ramp_qpos[:, 3:])
        polar_angle = jp.concatenate([jp.cos(ramp_angle), jp.sin(ramp_angle)], -1)
        if self.polar_obs:
            ramp_qpos = jp.concatenate([ramp_qs[:, :3], polar_angle], -1)

        ramp_obs = jp.concatenate([ramp_qs, ramp_qds], -1)
        if self.pad_ramp_size:
            ramp_obs = jp.concatenate([ramp_obs, jp.zeros((ramp_obs.shape[0], 3))], -1)

        obs = jp.concatenate((ramp_obs, ramp_angle, ramp_qpos[:, :3]))

        print("Ramp obs shape: ", obs.shape)
        return obs


class Cylinders(EnvModule):
    '''
        Add cylinders to the environment.
        Args:
            n_objects (int): Number of cylinders
            diameter (float or (float, float)): Diameter of cylinders. If tuple of floats, every
                episode the diameter is drawn uniformly from (diameter[0], diameter[1]).
                (Note that all cylinders within an episode still share the same diameter)
            height (float or (float, float)): Height of cylinders. If tuple of floats, every
                episode the height is drawn uniformly from (height[0], height[1]).
                (Note that all cylinders within an episode still share the same height)
            make_static (bool): Makes the cylinders static, preventing them from moving. Note that
                the observations (and observation keys) are different when make_static=True
            placement_fn (fn or list of fns): See mae_envs.modules.util:rejection_placement for spec
                If list of functions, then it is assumed there is one function given per cylinder
            rgba ([float, float, float, float]): Determines cylinder color.
    '''
    @store_args
    def __init__(self, n_objects, diameter, height, make_static=False,
                 placement_fn=None, rgba=[1., 1., 1., 1.]):
        if type(diameter) not in [list, np.ndarray]:
            self.diameter = [diameter, diameter]
        if type(height) not in [list, np.ndarray]:
            self.height = [height, height]

    def build_world_step(self, env, floor, floor_size):
        default_name = 'static_cylinder' if self.make_static else 'moveable_cylinder'
        diameter = env._random_state.uniform(self.diameter[0], self.diameter[1])
        height = env._random_state.uniform(self.height[0], self.height[1])
        obj_size = (diameter, height, 0)
        successful_placement = True
        for i in range(self.n_objects):
            geom = Geom('cylinder', obj_size, name=f'{default_name}{i}', rgba=self.rgba)
            if self.make_static:
                geom.mark_static()

            if self.placement_fn is not None:
                _placement_fn = (self.placement_fn[i]
                                 if isinstance(self.placement_fn, list)
                                 else self.placement_fn)
                pos, _ = rejection_placement(env, _placement_fn, floor_size, diameter * np.ones(2))
                if pos is not None:
                    floor.append(geom, placement_xy=pos)
                else:
                    successful_placement = False
            else:
                floor.append(geom)

        return successful_placement

    def cache_step(self, env):
        # Cache q, qd indices
        self.cylinder_q_idxs = env.q_indices['moveable_cylinder']
        self.cylinder_qd_idxs = env.qd_indices['moveable_cylinder']

    def observation_step(self, env, sim):
        qpos = sim.data.qpos.copy()
        qvel = sim.data.qvel.copy()

        if self.make_static:
            s_cylinder_geom_idxs = np.expand_dims(self.s_cylinder_geom_idxs, -1)
            s_cylinder_xpos = sim.data.geom_xpos[self.s_cylinder_geom_idxs]
            obs = {'static_cylinder_geom_idxs': s_cylinder_geom_idxs,
                   'static_cylinder_xpos': s_cylinder_xpos}
        else:
            m_cylinder_geom_idxs = np.expand_dims(self.m_cylinder_geom_idxs, -1)
            m_cylinder_xpos = sim.data.geom_xpos[self.m_cylinder_geom_idxs]
            m_cylinder_qpos = qpos[self.m_cylinder_qpos_idxs]
            m_cylinder_qvel = qvel[self.m_cylinder_qvel_idxs]
            mc_angle = normalize_angles(m_cylinder_qpos[:, 3:])
            polar_angle = np.concatenate([np.cos(mc_angle), np.sin(mc_angle)], -1)
            m_cylinder_qpos = np.concatenate([m_cylinder_qpos[:, :3], polar_angle], -1)
            m_cylinder_obs = np.concatenate([m_cylinder_qpos, m_cylinder_qvel], -1)
            obs = {'moveable_cylinder_geom_idxs': m_cylinder_geom_idxs,
                   'moveable_cylinder_xpos': m_cylinder_xpos,
                   'moveable_cylinder_obs': m_cylinder_obs}

        return obs


class LidarSites(EnvModule):
    '''
    Adds sites to visualize Lidar rays
        Args:
            n_agents (int): number of agents
            n_lidar_per_agent (int): number of lidar sites per agent
    '''
    @store_args
    def __init__(self, n_agents, n_lidar_per_agent):
        pass

    def build_world_step(self, env, floor, floor_size):
        for i in range(self.n_agents):
            for j in range(self.n_lidar_per_agent):
                floor.mark(f"agent{i}:lidar{j}", (0.0, 0.0, 0.0), rgba=np.zeros((4,)))
        return True

    def modify_sim_step(self, env, sim):
        # set lidar size and shape
        self.lidar_ids = np.array([[sim.model.site_name2id(f"agent{i}:lidar{j}")
                                    for j in range(self.n_lidar_per_agent)]
                                   for i in range(self.n_agents)])
        # set lidar site shape to cylinder
        sim.model.site_type[self.lidar_ids] = 5
        sim.model.site_size[self.lidar_ids, 0] = 0.02
