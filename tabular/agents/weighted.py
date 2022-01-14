import numpy as np

from .agent import Agent


class WDQlearning(Agent):
    def __init__(self, env, c=1, discount=0.9, learning_rate=0.01, epsilon=0.1):
        super().__init__(env, discount, learning_rate, epsilon)
        self.name = "WDQ" + str(c)
        self.c = c
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
            amax = np.argmax(self.q1[ns])
            amin = np.argmin(self.q1[ns])
            beta = abs(self.q2[ns, amax] - self.q2[ns, amin])
            beta = beta / (self.c + beta)
            td_target = r + self.discount * (
                beta * self.q1[ns, amax] + (1 - beta) * self.q2[ns, amax]
            )
            td_delta = td_target - self.q1[state, action]
            self.q1[state, action] += self.lr * td_delta
        else:
            amax = np.argmax(self.q2[ns])
            amin = np.argmin(self.q2[ns])
            beta = abs(self.q1[ns, amax] - self.q1[ns, amin])
            beta = beta / (self.c + beta)
            td_target = r + self.discount * self.discount * (
                beta * self.q2[ns, amax] + (1 - beta) * self.q1[ns, amax]
            )
            td_delta = td_target - self.q2[state, action]
            self.q2[state, action] += self.lr * td_delta
