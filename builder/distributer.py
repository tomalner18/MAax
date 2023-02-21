import brax

from brax import envs
from brax import jumpy as jp
from brax.envs import to_torch
from brax.io import html
from brax.io import image
import jax
from jax import numpy as jnp
from jax import random

class Distributer():
    def 

    def random_positions(obj_cnt, key, min_x, max_x, min_y, max_y, min_z, max_z):
        """Generate random positions for objects."""
        obj_pos = 
        for i in range(obj_cnt):
            key, *subkeys = random.split(key, obj_cnt + 1)
            

    def random_separarted():
        pass

    def poisson_distributed():
        pass