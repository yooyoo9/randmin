import numpy as np

from .agent import Agent


class Qlearning(Agent):
    def __init__(self, env, discount=0.9, learning_rate=0.01, epsilon=0.1):
        super().__init__(env, discount, learning_rate, epsilon)
        self.name = "Q"
        self.q = np.random.uniform(low=-1, high=1, size=(self.n_states, self.n_actions))
        self.state_action_visitation = np.ones((self.n_states, self.n_actions))

    def choose_best_action(self, state):
        return np.argmax(self.q[state])

    def get_epsilon(self, state):
        return 1 / np.sum(self.state_action_visitation[state]) ** 0.5

    def calculate_diff(self):
        return self.env.get_result(self.q, self.discount)
        return diff

    def update(self, state, action, r, ns):
        self.state_action_visitation[state, action] += 1
        na = np.argmax(self.q[ns])
        lr = 1 / self.state_action_visitation[state, action] ** 0.8
        td_target = r + self.discount * self.q[ns, na]
        td_delta = td_target - self.q[state, action]
        self.q[state, action] += lr * td_delta
