import math
import jax
import numpy as np
from jax import numpy as jnp
import random

class Distributer:
    """Generates a point distribution for objects in the scene."""
    def __init__(self):
        self.d = 1

    def distribute_boundaries(self, env_dim):
        """Distribute boundaries in the scene."""
        boundaries = []
        # Create top wall
        boundaries.append((0, env_dim[1] / 2))
        # Create bottom wall
        boundaries.append((0, -env_dim[1] / 2))
        # Create left wall
        boundaries.append((-env_dim[0] / 2, 0))
        # Create right wall
        boundaries.append((env_dim[0] / 2, 0))
        return boundaries

    def random_unseparated(self, key, obj_cnt, obj_width, width_range, height_range):
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
                x = random.randint(width_range[0] + obj_width, width_range[1] - obj_width)
                y = random.randint(height_range[0] + obj_width, height_range[1] - obj_width)
                obj = (x, y, obj_width)
                objects.append(obj)
        return objects
            

    def random_separated(self, key, obj_cnt, obj_width, separation, width_range, height_range):
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
        attempt_threshold = 100
        for i in range(obj_cnt):
            attempts = 0
            while attempts < attempt_threshold:
                x = random.randint(width_range[0] + obj_width, width_range[1] - obj_width)
                y = random.randint(height_range[0] + obj_width, height_range[1] - obj_width)
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

        # Calculate the cell size based on the minimum separation
        cell_size = (r + max_width) / math.sqrt(2)
        min_separation = r + max_width

        spawn_width = (width_range[0] + max_width, width_range[1] - max_width)
        spawn_height = (height_range[0] + max_width, height_range[1] - max_width)

        # Calculate the number of rows and columns in the grid
        num_rows = math.ceil((spawn_width[1] - spawn_width[0]) / cell_size)
        num_cols = math.ceil((spawn_height[1] - spawn_height[0]) / cell_size)

        # Create the grid
        grid = [[None] * num_cols for i in range(num_rows)]

        # Create a list to hold the active points
        active_points = []

        # Choose a random starting point and add it to the grid and the active points list
        x = random.uniform(spawn_width[0], spawn_width[1])
        y = random.uniform(spawn_height[0], spawn_height[1])
        grid_row = math.floor((y - spawn_height[0]) / cell_size)
        grid_col = math.floor((x - spawn_width[0]) / cell_size)
        grid[grid_row][grid_col] = (x, y)
        active_points.append((x, y))

        # Loop through the active points list until it is empty
        while len(active_points) > 0:
            # Choose a random active point
            current_point = random.choice(active_points)

            # Generate up to k points uniformly distributed between radius r and 2r from the current point
            k = 30
            found_new_point = False
            for i in range(k):
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(r, 2 * r)
                new_x = current_point[0] + radius * math.cos(angle)
                new_y = current_point[1] + radius * math.sin(angle)

                # Check if the new point is within the bounds of the sampling area
                if new_x >= spawn_width[0] and new_x <= spawn_width[1] and new_y >= spawn_height[0] and new_y <= spawn_height[1]:
                    # Check if the new point is too close to an existing point
                    grid_row = math.floor((new_y - spawn_height[0]) / cell_size)
                    grid_col = math.floor((new_x - spawn_width[0]) / cell_size)
                    if grid[grid_row][grid_col] is None:
                        is_valid = True
                        for row in range(max(0, grid_row - 2), min(num_rows, grid_row + 3)):
                            for col in range(max(0, grid_col - 2), min(num_cols, grid_col + 3)):
                                neighbor = grid[row][col]
                                if neighbor is not None:
                                    distance = math.sqrt((new_x - neighbor[0]) ** 2 + (new_y - neighbor[1]) ** 2)
                                    if distance < r:
                                        is_valid = False
                                        break
                            if not is_valid:
                                break
                        if is_valid:
                            grid[grid_row][grid_col] = (new_x, new_y)
                            active_points.append((new_x, new_y))
                            found_new_point = True
                            break

            # If no new point was found, remove the current point from the active points list
            if not found_new_point:
                active_points.remove(current_point)

        # Extract the points from the grid and return them
        points = []
        for row in range(num_rows):
            for col in range(num_cols):
                if grid[row][col] is not None:
                    points.append(grid[row][col])
        return points