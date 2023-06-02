import jax
from jax import numpy as jp
from copy import deepcopy
from mae_envs.wrappers.util import MWrapper


class PreparationPhase(MWrapper):
    '''
        Rewards are switched off during preparation.

        Args: prep_fraction (float): Fraction of total time that is preparation time
    '''
    def __init__(self, env, prep_fraction=.2):
        super().__init__(env)
        self.prep_fraction = prep_fraction
        self.prep_time = self.prep_fraction * self.unwrapped.horizon
        self.n_agents = self.metadata['n_agents']
        self.step_counter = 0

    def reset(self, rng):
        self.step_counter = 0
        self.in_prep_phase = True
        state = self.env.reset(rng)
        obs = self.observation(state.obs)
        return state.replace(obs=obs)

    def reward(self, reward):
        reward = jax.lax.cond(
            self.in_prep_phase,
            lambda _: jp.zeros_like(reward),
            lambda _: reward,
            operand=None
        )

        print('Reward: ', reward)
        return reward

    def observation(self, obs):
        obs['prep_obs'] = (jp.ones((self.n_agents, 1)) *
                           jp.minimum(1.0, self.step_counter / (self.prep_time + 1e-5)))

        return obs

    def step(self, state, action):
        dst_state = self.env.step(state, action)
        rew = self.reward(dst_state.reward)
        self.step_counter += 1
        print('Step counter: ', self.step_counter)
        print('Prep time: ', self.prep_time)

        self.in_prep_phase = jax.lax.cond(
            self.step_counter < self.prep_time,
            lambda _: True,
            lambda _: False,
            operand=None
        )
        print('In prep phase: ', self.in_prep_phase)

        info = dst_state.info

        info['in_prep_phase'] = self.in_prep_phase

        print(info)

        obs = self.observation(dst_state.obs)

        return dst_state.replace(obs=obs, reward=rew, info=info)


class NoActionsInPrepPhase(MWrapper):
    '''Disabled actions for indexed agents  during preparation phase.'''

    def __init__(self, env, agent_idxs):
        super().__init__(env)
        self.agent_idxs =jp.array(agent_idxs)
        print(agent_idxs)

    def reset(self, rng):
        state = self.env.reset(rng)
        self.in_prep_phase = True
        return state

    def step(self, state, action):
        dst_state = self.env.step(state, self.action(action))
        self.in_prep_phase = state.info['in_prep_phase'].astype(bool)
        return dst_state

    def action(self, action):
        print('Action before: ', action)
        # print('In prep phase: ', self.in_prep_phase)

        zero_ac = 0.0

        ac = jax.lax.cond(
            self.in_prep_phase,
            lambda: action.at[self.agent_idxs].set(jp.zeros(action.shape[1])),
            lambda: action
        )

        print('Action after: ', ac)
        return ac


class MaskPrepPhaseAction(MWrapper):
    '''
        Masks a (binary) action during preparation phase
    '''
    def __init__(self, env, action_key):
        super().__init__(env)
        self.action_key = action_key

    def reset(self, rng):
        state = self.env.reset(rng)
        self.in_prep_phase = True
        return state

    def step(self, state, action):
        action[self.action_key] = (action[self.action_key] * (1 - self.in_prep_phase)).astype(bool)

        dst_state = self.env.step(state, action)
        self.in_prep_phase = state.info['in_prep_phase'].astype(bool)
        return state
