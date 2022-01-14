import numpy as np

from .agent import Agent


class DQlearning(Agent):
    def __init__(self, env, discount=0.9, learning_rate=0.01, epsilon=0.1):
        super().__init__(env, discount, learning_rate, epsilon)
        self.name = "DQ"
        self.q1 = np.random.uniform(
            low=-1, high=1, size=(self.n_states, self.n_actions)
        )
        self.q2 = np.random.uniform(
            low=-1, high=1, size=(self.n_states, self.n_actions)
        )

    def choose_best_action(self, state):
        action = np.argmax(self.q1[state] + self.q2[state])
        return action

    def calculate_diff(self):
        return self.env.get_result((self.q1 + self.q2) / 2, self.discount)

    def update(self, state, action, r, ns):
        if np.random.random() < 0.5:
            na = np.argmax(self.q1[ns])
            td_target = r + self.discount * self.q2[ns, na]
            td_delta = td_target - self.q1[state, action]
            self.q1[state, action] += self.lr * td_delta
        else:
            na = np.argmax(self.q2[ns])
            td_target = r + self.discount * self.q1[ns, na]
            td_delta = td_target - self.q2[state, action]
            self.q2[state, action] += self.lr * td_delta
