import math
import jax
import numpy as np
from jax import numpy as jnp
import random

class Distributer:
    """Generates a point distribution for objects in the scene."""
    def __init__(self):
        self.d = 1

    def random_unseparated(self, key, obj_cnt, obj_width, space_width, space_height,):
        """Generate random positions for objects.

        Args:
        key (jax.random.PRNGKey): The random key to use for sampling.
        obj_cnt (int): The maximum number of objects to spawn.
        object_width (float): The half-width of each object.
        space_width (float): The width of the 2D space in which to spawn the objects.
        space_height (float): The height of the 2D space in which to spawn the objects.
        """
    
        """
        Returns:
        A list of tuples containing the (x, y) coordinates of the spawned objects.
        """
        objects = []
        for i in range(obj_cnt):
            while True:
                x = random.randint(0, space_width - obj_width)
                y = random.randint(0, space_height - obj_width)
                obj = (x, y, obj_width)
                objects.append(obj)
        return objects
            

    def random_separated(self, key, obj_cnt, obj_width, separation, space_width, space_height):
        """
        Generates up to n objects with a given separation and object size within a 2D space.

        Args:
        key (jax.random.PRNGKey): The random key to use for sampling.
        obj_cnt (int): The maximum number of objects to spawn.
        object_width (float): The half-width of each object.
        separation (float): The minimum distance between each object.
        space_width (float): The width of the 2D space in which to spawn the objects.
        space_height (float): The height of the 2D space in which to spawn the objects.

        Returns:
        A list of tuples containing the (x, y) coordinates of the spawned objects.
        """
        objects = []
        attempt_threshold = 5
        for i in range(obj_cnt):
            attempts = 0
            while attempts < attempt_threshold:
                x = random.randint(0, space_width - obj_width)
                y = random.randint(0, space_height - obj_width)
                obj = (x, y, obj_width)
                if not any(self.check_overlap(obj, other, separation) for other in objects):
                    objects.append(obj)
                attempts += 1
        return objects

    def check_overlap(self, obj1, obj2, min_separation):
        x1, y1, w1 = obj1
        x2, y2, w2 = obj2
        return abs(x1 - x2) < w1 + w2 + min_separation and abs(y1 - y2) < w1 + w2 + min_separation

    def poisson_distribute(self, key, max_width, r, width_range, height_range):
        """
        Generates a list of sample points in a 2D space, such that each point is
        at least a certain minimum distance away from any other point.

        Parameters:
        key (jax.random.PRNGKey): The random key to use for sampling.
        max_width (float): The width of the objects to be distributed.
        min_dist (float): The minimum distance between any two points.
        space_width (float): The width of the 2D space to sample in.
        space_height (float): The height of the 2D space to sample in.

        Returns:
        List of tuples: A list of sample points, where each point represents the
        center of an object that can be placed in the space without overlapping
        with any other objects.
        """

        cell_size = r / math.sqrt(2)
        grid_width = math.ceil((width_range[1] - width_range[0]) / cell_size)
        grid_height = math.ceil((height_range[1] - height_range[0]) / cell_size)
        grid = [None] * (grid_width * grid_height)
        active_list = []
        samples = []
        
        def get_cell(x, y):
            if x < 0 or x >= grid_width or y < 0 or y >= grid_height:
                return None
            return grid[x + y * grid_width]
        
        def set_cell(x, y, sample):
            grid[x + y * grid_width] = sample
            active_list.append((x, y))
        
        def get_random_point_around(sample):
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(r, 2 * r)
            x = sample[0] + distance * math.cos(angle)
            y = sample[1] + distance * math.sin(angle)
            return (x, y)
        
        def is_valid_sample(sample):
            x, y = sample
            if x < width_range[0] or x > width_range[1] or y < height_range[0] or y > height_range[1]:
                return False
            cell_x = int((x - width_range[0]) / cell_size)
            cell_y = int((y - height_range[0]) / cell_size)
            for i in range(cell_x - 2, cell_x + 3):
                for j in range(cell_y - 2, cell_y + 3):
                    if i >= 0 and i < grid_width and j >= 0 and j < grid_height:
                        cell = get_cell(i, j)
                        if cell is not None and math.sqrt((cell[0] - x)**2 + (cell[1] - y)**2) < r:
                            return False
            return True
        
        x = random.uniform(width_range[0], width_range[1])
        y = random.uniform(height_range[0], height_range[1])
        samples.append((x, y))
        set_cell(int((x - width_range[0]) / cell_size), int((y - height_range[0]) / cell_size), (x, y))
        
        while active_list:
            index = random.randint(0, len(active_list) - 1)
            sample = active_list[index]
            for i in range(30):
                new_sample = get_random_point_around(sample)
                if is_valid_sample(new_sample):
                    samples.append(new_sample)
                    set_cell(int((new_sample[0] - width_range[0]) / cell_size), int((new_sample[1] - height_range[0]) / cell_size), new_sample)
                    break
            else:
                active_list.pop(index)
        
        return samples