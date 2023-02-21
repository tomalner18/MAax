import functools
import time

from IPython.display import HTML, Image
import gym

import brax

from brax import envs
from brax import jumpy as jp
from brax.envs import to_torch
from brax.io import html
from brax.io import image
import jax
from jax import numpy as jnp
from jax import random
import torch
v = torch.ones(1, device='cuda')  # init torch cuda before jax


seed = 42
key = random.PRNGKey(42)


def add_agent(config, agent_mass=1.0, agent_size=(0.5, 0.5, 0.5)):
    # Add the agent body
    agent = config.bodies.add(name='agent')
    cap = agent.colliders.add().capsule
    cap.radius, cap.length = 0.5, 1
    agent.mass = agent_mass
    # agent.damping = 1e-2
    # agent.friction = 0.6


# Adds joint for rolling the agent
def add_joint(config):
    joint = config.joints.add(name='rolling')
    joint.parent = 'ground'
    joint.child = 'agent'
    joint.angle = -1.57
    joint.twist = 1.0
    joint.limit.velocity = 100.0
    joint.limit.torque = 100.0
    joint.spring.stiffness = 1e5


def add_objects(config, key, object_count, cube_mass=1.0, cube_size=(0.5, 0.5, 0.5)):
    key, subkeys = random.split(key, object_count + 1)
    for i in range(object_count):
        cube = config.bodies.add(name=f'cube_{i}')
        cube.box.size = cube_size
        cube.mass = cube_mass
        cube.damping = 1e-2
        cube.friction = 0.6
        cube.position = jp.array([jax.random.uniform(key=subkeys[i], minval=-2, maxval=2),
                            jax.random.uniform(key=subkeys[i], minval=-2, maxval=2),
                            jax.random.uniform(key=subkeys[i], minval=0, maxval=4)])


def create_config():
    # Define the environment config.
    sphere_maze = brax.Config(dt=0.01, substeps=20, dynamics_mode='pbd')

    # Add the ground, a frozen (immovable) infinite plane
    ground = sphere_maze.bodies.add(name='ground')
    ground.frozen.all = True
    plane = ground.colliders.add().plane
    plane.SetInParent()  # for setting an empty oneof

    # Add the agent body
    add_agent(sphere_maze)


    # Add the cubes.
    add_objects(sphere_maze, key, object_count=5)

    return sphere_maze

def create_environment(config):
    gravity = jnp.array([0., 0., -9.81])
    env = envs.create_env(gravity=gravity.tolist(), config=config, sys=envs.SysType.PHYSICS)
    return env

# Set the actions
def set_action(env, action):
    torque = jnp.array([0., 0., action[0]])
    env.physics.forces['agent', 'rolling'].max_torque = jnp.abs(torque)
    env.physics.forces['agent', 'rolling'].torque = torque

# Create the info dictionary
def info(env):
    return {}

# Create the observation dictionary
def obs(env):
    pos = env.physics.bodies.pos
    vel = env.physics.bodies.vel
    return {
        'position': take(pos, 'agent'),
        'velocity': take(vel, 'agent'),
    }

def gen_vis(config):
    # Create a visualization
    vis_config = html.Config.from_config(config)
    vis_config.side_length = 15
    vis = html.Visualization(vis_config)
    return vis

if __name__ == 'main':
    config = create_config()
    env = create_environment(config)
    vis = gen_vis()

    # Reset the environment
    state = env.reset(rng=jax.random.PRNGKey(0))

    # Run the environment loop
    for i in range(500):
        print(i)
        action = jnp.array([jnp.sin(i / 10)])
        set_action(env, action)
        state = env.step(state, action)
        vis.add_frame(env.physics)
    vis.render()
