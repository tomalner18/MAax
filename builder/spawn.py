from .distribute import Distributer
from .build import Builder
import jax
from jax import numpy as jnp
from jax import random

class Spawner:
    def __init__(self, env_dim):
        self.distributer = Distributer()
        self.builder = Builder()
        self.env_dim = env_dim


    def spawn_objects(self, config, key, space_dim, separation, obj_type="cube", method="poisson"):
        """
        Spawns objects in the scene.
        """
        obj_max_width = 3

        if method == "poisson":
            points = self.distributer.poisson_distribute(key, obj_max_width / 2, separation, (-int(space_dim[0] / 2), int(space_dim[0] / 2)), (-int(space_dim[1] / 2), int(space_dim[1] / 2)))
            print(points)
        elif method == "unseparated":
            points = self.distributer.random_unseparated(key, 10, 1, (-int(space_dim[0] / 2), int(space_dim[0] / 2)), (-int(space_dim[1] / 2), int(space_dim[1] / 2)))
        elif method == "separated":
            points = self.distributer.random_separated(key, obj_cnt=10, obj_width=1, separation=5, width_range=(-int(space_dim[0] / 2), int(space_dim[0] / 2)), height_range=(-int(space_dim[1] / 2), int(space_dim[1] / 2)))


        objects = self.builder.build(points, obj_type)


        boundary_points = self.distributer.distribute_boundaries(self.env_dim)

        boundaries = self.builder.build(boundary_points, "box")

        # Create top wall
        b0 = config.bodies.add(name=f'boundary_0')
        box = b0.colliders.add().box
        box.halfsize.x = self.env_dim[0] / 2
        box.halfsize.y = 0.5 / 2
        box.halfsize.z = 4 / 2
        b0.mass = boundaries[0].mass

        # Create bottom wall
        b1 = config.bodies.add(name=f'boundary_1')
        box = b1.colliders.add().box
        box.halfsize.x = self.env_dim[0] / 2
        box.halfsize.y = 0.5 / 2
        box.halfsize.z = 4 / 2
        b1.mass = boundaries[1].mass

        #Create left wall
        b2 = config.bodies.add(name=f'boundary_2')
        box = b2.colliders.add().box
        box.halfsize.x = 0.5 / 2
        box.halfsize.y = self.env_dim[1] / 2
        box.halfsize.z = 4 / 2
        b2.mass = boundaries[2].mass

        # Create right wall
        b3 = config.bodies.add(name=f'boundary_3')
        box = b3.colliders.add().box
        box.halfsize.x = 0.5 / 2
        box.halfsize.y = self.env_dim[1] / 2
        box.halfsize.z = 4 / 2
        b3.mass = boundaries[3].mass

        # Randomly scale the objects
        key, *subkeys = random.split(key, len(objects) + 1)
        for i, o in enumerate(objects):
            scale = random.uniform(key=subkeys[i], minval=0.5, maxval=obj_max_width)
            o.set_half_size([o.half_size[0] * scale, o.half_size[1] * scale, o.half_size[2] * scale])

        # Add the objects to the config
        for i, object in enumerate(objects):
            body = config.bodies.add(name=f'cube_{i}')
            box = body.colliders.add().box
            box.halfsize.x = object.half_size[0]
            box.halfsize.y = object.half_size[1]
            box.halfsize.z = object.half_size[2]
            if i % 2 == 0:
                box.halfsize.x = object.half_size[0]
                box.halfsize.y = object.half_size[1]
            else:
                box.halfsize.x = 3 - object.half_size[0]
                box.halfsize.y = 0.1
                box.halfsize.z = 1.5
            body.mass = object.mass

        # Assign the positions of the objects
        default = config.defaults.add()
        key, *subkeys = random.split(key, len(objects) + 1)
        for i, object in enumerate(objects):
            qp = default.qps.add(name=f'cube_{i}')
            qp.pos.x = object.pos[0]
            qp.pos.y = object.pos[1]
            if i % 2 ==  0:
                qp.pos.z = object.half_size[2]
            else:
                qp.pos.z = 1.5
            qp.rot.z = random.uniform(key=subkeys[i], minval=0, maxval=360)

        # Assign the positions of the boundaries
        for i, boundary in enumerate(boundaries):
            qp = default.qps.add(name=f'boundary_{i}')
            qp.pos.x = boundary.pos[0]
            qp.pos.y = boundary.pos[1]
            # qp.pos.z = boundary.half_size[2] * 2
            qp.pos.z = 2

        

