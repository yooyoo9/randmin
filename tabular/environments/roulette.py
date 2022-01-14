import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class RouletteEnv(gym.Env):
    """
    Simple roulette environment
    The roulette wheel has s spots. If the bet is 0 and a 0 comes up, you win a reward of s-2.
    If any other number comes up you get a reward of -1.
    For non-zero bets, if the parity of your bet matches the parity of the spin, you win 1.
    Otherwise you receive a reward of -1.
    The last action (s+1) stops the rollout for a return of 0 (walking away)
    """

    def __init__(self, max_steps=500):
        self.name = "Roulette"
        self.n = 157
        self.action_space = spaces.Discrete(self.n)
        self.observation_space = spaces.Discrete(1)
        self.max_step = max_steps
        self.steps = 0
        self.seed()

        expected_val = 35 / 38 - 37 / 38
        self.q_opt = expected_val * np.ones(
            (self.observation_space.n, self.action_space.n)
        )
        self.q_opt[0, -1] = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        val = self.np_random.random()
        self.steps += 1
        if action == self.n - 1:
            # observation, reward, done, info
            return 0, 0, True, {}
        elif action < 38:
            r = 35 if val < 1 / 38 else -1
        elif action < 99:
            r = 17 if val < 1 / 19 else -1
        elif action < 111:
            r = 11 if val < 1 / (12 + 2 / 3) else -1
        elif action < 133:
            r = 8 if val < 1 / (9 + 1 / 2) else -1
        elif action < 144:
            r = 5 if val < 1 / (6 + 1 / 3) else -1
        elif action < 150:
            r = 2 if val < 1 / (3 + 1 / 6) else -1
        elif action < 156:
            r = 1 if val < 1 / (2 + 1 / 9) else -1
        return 0, r, self.steps == self.max_step, {}

    def reset(self):
        self.steps = 0
        return 0

    def get_result(self, q, discount):
        return np.mean((q - self.q_opt) ** 2)
