import numpy as np
from worldgen.util.types import store_args
from worldgen.util.sim_funcs import (q_idxs_from_joint_prefix,
                                            qd_idxs_from_joint_prefix)
from worldgen import Geom, Material, ObjFromXML
from worldgen.transforms import set_geom_attr_transform
from worldgen.util.rotation import normalize_angles
from maax.util.transforms import remove_hinge_axis_transform
from maax.modules import Module, rejection_placement, get_size_from_xml

import jax
from jax import numpy as jp


class Boxes(Module):
    '''
    Add moveable boxes to the environment.
        Args:
            n_boxes (int or (int, int)): number of boxes. If tuple of ints, every episode the
                number of boxes is drawn uniformly from range(n_boxes[0], n_boxes[1] + 1)
            n_elongated_boxes (int or (int, int)): Number of elongated boxes. If tuple of ints,
                every episode the number of elongated boxes is drawn uniformly from
                range(n_elongated_boxes[0], min(curr_n_boxes, n_elongated_boxes[1]) + 1)
            placement_fn (fn or list of fns): See maax.modules.util:rejection_placement for spec
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
                 boxid_obs=True, boxsize_obs=False, polar_obs=True, free=True):
        if type(n_boxes) not in [tuple, list, np.ndarray]:
            self.n_boxes = [n_boxes, n_boxes]
        if type(n_elongated_boxes) not in [tuple, list, np.ndarray]:
            self.n_elongated_boxes = [n_elongated_boxes, n_elongated_boxes]

    def build_step(self, env, floor, floor_size):
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
                # If Free: Ramps have 7 qs (pos, 1, rot) and 6 qds (vel, angv)
        # If not Free: Ramps have 6 qs (pos, rot) and 7 qds (vel, angv, angv2)
        qs = state.q.copy()
        qds = state.qd.copy()


        box_inds = jp.expand_dims(jp.arange(self.curr_n_boxes), -1)
        box_qs = qs[self.box_q_idxs]
        box_qds = qds[self.box_qd_idxs]

        if self.free:
            box_qs = jp.reshape(box_qs, newshape=(-1,7))
            box_qds = jp.reshape(box_qds, newshape=(-1,6))
            box_angle = normalize_angles(box_qs[:, 4:])

        else:
            box_qs = jp.reshape(box_qs, newshape=(-1,3))
            box_qds = jp.reshape(box_qds, newshape=(-1,3))
            box_angle = normalize_angles(box_qs[:, [-1]])

        polar_angle = jp.concatenate([jp.cos(box_angle), jp.sin(box_angle)], -1)
        if self.polar_obs:
            box_qs = jp.concatenate([box_qs[:, :3], polar_angle], -1)
        box_obs = jp.concatenate([box_qs, box_qds], -1)

        if self.boxid_obs:
            box_obs = jp.concatenate([box_obs, box_inds], -1)
        if self.n_elongated_boxes[1] > 0 or self.boxsize_obs:
            box_obs = jp.concatenate([box_obs, self.box_size_array], -1)



        d_obs = {'box_obs': box_obs,
        'box_angle': box_angle,
        'box_pos': box_qs[:, :3]}

        # obs = jp.concatenate((box_obs, box_angle, box_qs[:, :3]))


        return d_obs


class Ramps(Module):
    '''
    Add moveable ramps to the environment.
        Args:
            n_ramps (int): number of ramps
            placement_fn (fn or list of fns): See maax.modules.util:rejection_placement for spec
                If list of functions, then it is assumed there is one function given per ramp
            friction (float): ramp friction
            polar_obs (bool): Give observations about rotation in polar coordinates
            pad_ramp_size (bool): pads 3 rows of zeros to the ramp observation. This makes
                ramp observations match the dimensions of elongated box observations.
    '''
    @store_args
    def __init__(self, n_ramps, placement_fn=None, friction=None, polar_obs=True,
                 pad_ramp_size=False, free=True):
        pass

    def build_step(self, env, floor, floor_size):
        successful_placement = True

        env.metadata['curr_n_ramps'] = np.ones((self.n_ramps)).astype(bool)

        for i in range(self.n_ramps):
            char = chr(ord('A') + i % 26)
            if self.free:
                geom = ObjFromXML('ramp', name=f"ramp{i}")
            else:
                geom = ObjFromXML('ramp_slide', name=f"ramp{i}")
            # geom.set_material(Material(texture="chars/" + char + ".png"))
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
        # If Free: Ramps have 7 qs (pos, 1, rot) and 6 qds (vel, angv)
        # If not Free: Ramps have 6 qs (pos, rot) and 7 qds (vel, angv, angv2)
        qs = state.q.copy()
        qds = state.qd.copy()

        if self.free:

            ramp_qs = qs[self.ramp_q_idxs]
            ramp_qds = qds[self.ramp_qd_idxs]

            ramp_qs = jp.reshape(ramp_qs, newshape=(-1,7))
            ramp_qds = jp.reshape(ramp_qds, newshape=(-1,6))

            ramp_angle = normalize_angles(ramp_qs[:, 4:])

        else:

            ramp_qs = qs[self.ramp_q_idxs]
            ramp_qds = qds[self.ramp_qd_idxs]

            ramp_qs = jp.reshape(ramp_qs, newshape=(-1,3))
            ramp_qds = jp.reshape(ramp_qds, newshape=(-1,3))

            ramp_angle = normalize_angles(ramp_qs[:, [-1]])


        polar_angle = jp.concatenate([jp.cos(ramp_angle), jp.sin(ramp_angle)], -1)
        if self.polar_obs:
            ramp_qs = jp.concatenate([ramp_qs[:, :3], polar_angle], -1)

        ramp_obs = jp.concatenate([ramp_qs, ramp_qds], -1)
        if self.pad_ramp_size:
            ramp_obs = jp.concatenate([ramp_obs, jp.zeros((ramp_obs.shape[0], 3))], -1)


        d_obs = {'ramp_obs': ramp_obs,
        'ramp_angle': ramp_angle,
        'ramp_q': ramp_qs}

        # obs = jp.concatenate((ramp_obs, ramp_angle, ramp_q[:, :3]))

        return d_obs

