import numpy as np
import gym
from gym import spaces
from gym.utils import seeding


class GridworldEnv(gym.Env):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.
    For example, a 4x4 grid looks as follows:
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T
    x is your position and T are the two terminal states.
    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    def __init__(self, n, max_steps, var1, var2, seed=0):
        self.name = "Gridworld_" + str(var1) + "_" + str(var2)
        self.n = n
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(n ** 2)
        self.max_step = max_steps
        self.steps = 0
        self.var1 = var1
        self.var2 = var2
        self.seed(seed)
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        self.steps += 1

        x = nx = self.state // self.n
        y = ny = self.state % self.n

        if action == 0:  # UP
            ny = min(self.n - 1, y + 1)
        elif action == 1:  # DOWN
            ny = max(0, y - 1)
        elif action == 2:  # LEFT
            nx = max(0, x - 1)
        else:
            nx = min(self.n - 1, x + 1)

        if nx == self.n - 1 and ny == self.n - 1:
            done = True
            r = self.np_random.normal(0, self.var1)
        else:
            done = False
            r = self.np_random.normal(-1, self.var2)

        done = done or self.steps == self.max_step
        ns = nx * self.n + ny
        self.state = ns
        return ns, r, done, {}

    def reset(self):
        self.steps = 0
        self.state = 0
        return 0

    def get_result(self, q, discount):
        opt = -(1 - discount ** (2 * self.n - 3)) / (1 - discount)
        return (opt - np.max(q[0])) ** 2
