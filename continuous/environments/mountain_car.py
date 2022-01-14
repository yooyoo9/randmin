import math

import numpy as np

from gym.envs.classic_control.mountain_car import MountainCarEnv


class RandMountainCarEnv(MountainCarEnv):
    def __init__(self, goal_velocity=0, variance=0):
        super().__init__(goal_velocity)
        self.name = "MountainCar" + str(variance)
        self.variance = variance
        self.state = None

    @staticmethod
    def done(steps):
        return steps < 250

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = np.random.normal(-1, self.variance) if not done else 0
        done = done

        self.state = np.array([position, velocity])
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return self.state
