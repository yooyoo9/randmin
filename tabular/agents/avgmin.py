import numpy as np

from .agent import Agent


class AvgMinQlearning(Agent):
    def __init__(self, env, beta=0.5, discount=0.9, learning_rate=0.01, epsilon=0.1):
        super().__init__(env, discount, learning_rate, epsilon)
        self.name = "AvgMin" + str(beta)
        self.beta = beta
        self.q = np.random.uniform(low=-1, high=1, size=(self.n_states, self.n_actions))
        self.old_values = np.random.uniform(
            low=-1, high=1, size=(self.n_states, self.n_actions)
        )

    def choose_best_action(self, state):
        return np.argmax(self.old_values[state])

    def calculate_diff(self):
        return self.env.get_result(self.q, self.discount)

    def update(self, state, action, r, ns):
        q_estimate = np.max(self.old_values[ns])
        td_target = r + self.discount * q_estimate
        td_delta = td_target - self.q[state, action]
        self.q[state, action] += self.lr * td_delta
        if self.q[state, action] <= self.old_values[state, action]:
            self.old_values[state, action] = self.q[state, action]
        else:
            self.old_values[state, action] *= 1 - self.beta
            self.old_values[state, action] += self.beta * self.q[state, action]
