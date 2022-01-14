import numpy as np

from .agent import Agent


class MaxMinQlearning(Agent):
    def __init__(self, env, n=2, discount=0.9, learning_rate=0.01, epsilon=0.1):
        super().__init__(env, discount, learning_rate, epsilon)
        self.name = "MaxMin" + str(n)
        self.n = n
        self.q_tables = [
            np.random.uniform(low=-1, high=1, size=(self.n_states, self.n_actions))
            for _ in range(n)
        ]
        self.nq_tables = [np.ones((self.n_states, self.n_actions)) for _ in range(n)]
        self.q_min = None
        self.get_q_min()

    def get_q_min(self):
        self.q_min = np.min(np.array(self.q_tables), axis=0)

    def choose_best_action(self, state):
        return np.argmax(self.q_min[state])

    def calculate_diff(self):
        return self.env.get_result(self.q_min, self.discount)

    def update(self, state, action, r, ns):
        idx = np.random.randint(self.n)
        self.nq_tables[idx][state, action] += 1
        na = np.argmax(self.q_min[ns])
        td_target = r + self.discount * self.q_tables[idx][ns, na]
        td_delta = td_target - self.q_tables[idx][state, action]
        self.q_tables[idx][state, action] += self.lr * td_delta
        self.get_q_min()
