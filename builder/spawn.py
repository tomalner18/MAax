from .distribute import Distributer
from .build import Builder

class Spawner:
    def __init__(self):
        self.distributer = Distributer()
        self.builder = Builder()

    def spawn_objects(self, config, key, space_dim, separation, obj_type="cube", method="poisson"):
        """
        Spawns objects in the scene.
        """

        if method == "poisson":
            points = self.distributer.poisson_distribute(key, 0.5, separation, (-10,10), (-10,10))
        elif method == "unseparated":
            points = self.distributer.random_unseparated(key, 10, 1, 10, 10)
        elif method == "separated":
            points = self.distributer.random_separated(key, 10, 1, 10, 10, 1)


        objects = self.builder.build(points, obj_type)

        # Add the objects to the config
        for i, object in enumerate(objects):
            body = config.bodies.add(name=f'cube_{i}')
            box = body.colliders.add().box
            box.halfsize.x = object.half_size[0]
            box.halfsize.y = object.half_size[1]
            box.halfsize.z = object.half_size[2]
            body.mass = object.mass

        # Assign the positions of the objects
        default = config.defaults.add()
        for i, object in enumerate(objects):
            qp = default.qps.add(name=f'cube_{i}')
            qp.pos.x = object.pos[0]
            qp.pos.y = object.pos[1]
            qp.pos.z = object.pos[2]

        

