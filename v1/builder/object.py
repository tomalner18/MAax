class Object:
    def __init__(self, mass, pos, rot, vel, ang, scale, interact=False):
        self.mass = mass
        self.pos = pos
        self.rot = rot
        self.vel = vel
        self.ang = ang
        self.scale = scale
        self.name = ""

    def spawn_width(self):
            pass
        
    def set_scale(self, scale):
            self.scale = scale
    
class Box(Object):
    """A 6-sided rectangular prism."""
    def __init__(self, mass, pos, rot, vel, ang, scale, interact=False):
        super().__init__(mass, pos, rot, vel, ang, scale, interact)
        self.type = "cube"
        self.interact = interact
        self.half_size = [0.5, 0.5, 0.5]

    def spawn_width(self):
        return max(self.half_size[0], self.half_size[1])

    def set_half_size(self, half_size):
        self.half_size = half_size

class Sphere(Object):
    """A sphere."""
    def __init__(self, mass, pos, rot, vel, ang, scale, interact=False):
        super().__init__(mass, pos, rot, vel, ang, scale, interact)
        self.type = "sphere"
        self.interact = interact
        self.radius = scale / 2

    def spawn_width(self):
        return self.radius

class Plane(Object):
    """An infinite plane."""
    def __init__(self, mass, pos, rot, vel, ang, scale, interact=False):
        super().__init__(mass, pos, rot, vel, ang, scale, interact)
        self.type = "plane"
        self.interact = interact

    def spawn_width(self):
        return 0

class ClippedPlane(Object):
    """A clipped plane"""
    def __init__(self, mass, pos, rot, vel, ang, scale, interact=False):
        super().__init__(mass, pos, rot, vel, ang, scale, interact)
        self.type = "clipped_plane"
        self.interact = interact
        self.half_size = scale / 2

    def spawn_width(self):
        return max(self.half_size[0], self.half_size[1])

class Capsule(Object):
    def __init__(self, mass, pos, rot, vel, ang, scale, interact=False):
        super().__init__(mass, pos, rot, vel, ang, scale, interact)
        self.type = "capsule"
        self.interact = interact
        self.length = 0
        self.radius = 0

    def spawn_width(self):
        return max(self.radius, self.length)
    
