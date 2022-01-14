from numpy import pi
import numpy as np

from gym.envs.classic_control.acrobot import AcrobotEnv, wrap, bound, rk4


class RandAcrobotEnv(AcrobotEnv):
    def __init__(self, variance=0):
        super().__init__()
        self.name = "Acrobot" + str(variance)
        self.variance = variance

    @staticmethod
    def done(steps):
        return steps < 195

    def step(self, a):
        s = self.state
        torque = self.AVAIL_TORQUE[a]

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(
                -self.torque_noise_max, self.torque_noise_max
            )

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])

        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._terminal()
        reward = np.random.normal(-1.0, self.variance) if not terminal else 0.0
        return self._get_ob(), reward, terminal, {}
