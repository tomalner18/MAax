from .object import Object, Box, Sphere, Plane, ClippedPlane, Capsule

class Builder:
    def __init__(self):
        pass

    def build_object(self, obj_type, mass=1.0, pos=[0,0,0], rot=[0, 0, 0], vel=[0, 0, 0], ang=[0, 0, 0], interact=False):
        """Creates a new object instance

        Args:
            obj_type (str): The type of object to create.
            mass (float): The mass of the object.
            pos (list): The position of the object.
            rot (list): The rotation of the object.
            vel (list): The velocity of the object.
            ang (list): The angular velocity of the object.
            scale (list): The scale of the object.
            interact (bool): Whether the object is interactable.

        Returns:
            Object: The newly created object.
        """

        if obj_type == "box":
            return Box(mass, pos, rot, vel, ang, interact)
        elif obj_type == "sphere":
            return Sphere(mass, pos, rot, vel, ang, interact)
        elif obj_type == "plane":
            return Plane(mass, pos, rot, vel, ang, interact)
        elif obj_type == "clipped_plane":
            return ClippedPlane(mass, pos, rot, vel, ang, interact)
        elif obj_type == "capsule":
            return Capsule(mass, pos, rot, vel, ang, interact)
        else:
            raise ValueError("Invalid object type: {}".format(obj_type))



    def build(self, objects, obj_type):
        """Builds a collection of Object instances from a list of centres and object types."""

        built_objects = []
        for obj in objects:
            pos = [obj[0], obj[1], 1]
            built_objects.append(self.build_object(obj_type, pos=pos))
        return built_objects