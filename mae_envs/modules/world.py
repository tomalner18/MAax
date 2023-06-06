import logging
from worldgen.transforms import set_geom_attr_transform
from mae_envs.modules import Module


class FloorAttributes(Module):
    '''
        For each (key, value) in kwargs, sets the floor geom attribute key to value.
    '''
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def build_step(self, env, floor, floor_size):
        for k, v in self.kwargs.items():
            floor.add_transform(set_geom_attr_transform(k, v))
        return True


class WorldConstants(Module):
    '''
        For each (key, value) in kwargs, sets env.sys.[key] = value
    '''
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def cache_step(self, env):
        for k, v in self.kwargs.items():
            if not hasattr(env.sys, k):
                logging.warning(f"Brax system does not have attribute {k}")
            else:
                'TODO: does not work for array types'
                getattr(env.sys, k)[:] = v
