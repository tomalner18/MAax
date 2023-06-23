# MAax
Procedural Environment Generation for Accelerated Multi-Agent Reinforcement Learning


### Installation
This repository depends on the [mujoco-worldgen](https://github.com/openai/mujoco-worldgen) package. We provide a fork of this repository and thank the authors for their contribution.
Install the necessary dependencies:
```
pip install -r worldgen/requirements.txt
pip install -e worldgen/
pip install -e maax/
```
### Use



Environment creation begins from the `Base` class. you can then add desired modules to introduce objects to the environment (`Walls`, `Boxes`, `Ramps`).

For adding environmental mechanics, we recommend implementing a new `Wrapper` inheriting from our custom Wrapper class designed for Multi-agent environments in Brax.

An example of environment creation can be found in the `example.ipynb` notebook. This notebook is currently parameterised to create a standard Hide-and_seek environment
with 2 hiders, 2 seekers, 2 boxes, 1 ramp and a small room sectioned by walls.