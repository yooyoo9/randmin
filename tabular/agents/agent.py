from collections import deque

import numpy as np


class Agent:
    def __init__(self, env, discount=1.0, learning_rate=0.01, epsilon=0.1):
        self.env = env
        self.discount = discount
        self.lr = learning_rate
        self.epsilon = epsilon
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n

        # Initialize variable to track rewards
        self.steps = 500
        self.rewards = -200

    @staticmethod
    def get_state_action_value(state, action, q_table):
        return q_table[state, action]

    def choose_best_action(self, state):
        raise NotImplementedError

    def policy(self, state):
        """
        Implements epsilon-greedy policy
        :param state: the state for which to determine the best action
        :return: the action from the given state
        """
        if np.random.random() <= self.epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action = self.choose_best_action(state)
        return action

    def update(self, state, action, r, ns):
        raise NotImplementedError

    def calculate_diff(self):
        raise NotImplementedError

    def train(self, n_episodes, n_avg, thres, render, verbose):
        diff = None
        rewards = []
        steps = []
        running_reward = deque(maxlen=n_avg)
        running_step = deque(maxlen=n_avg)
        for i_episode in range(n_episodes):
            done = False
            total_reward, reward = 0, 0
            state = self.env.reset()
            nb_steps = 0

            while not done:
                nb_steps += 1
                if render and i_episode >= (n_episodes - 1):
                    self.env.render()
                action = self.policy(state)
                ns, r, done, _ = self.env.step(action)
                self.update(state, action, r, ns)
                total_reward += r
                state = ns
            running_reward.append(total_reward)
            running_step.append(nb_steps)
            rewards.append(np.mean(running_reward))
            steps.append(np.mean(running_step))

            if (i_episode + 1) % n_avg == 0 and verbose:
                print(
                    "Episode {} Average Reward: {} Steps: {}".format(
                        i_episode + 1, rewards[-1], steps[-1]
                    )
                )

            if thres != -1:
                diff = self.calculate_diff()
                if (i_episode + 1) % 500 == 0 and verbose:
                    print("MSE to Qopt {}".format(diff))
                if nb_steps > n_avg and diff < thres:
                    if verbose:
                        print(
                            "Algorithm converged within {} steps. MSE: {}".format(
                                i_episode + 1, diff
                            )
                        )
        self.env.close()
        return rewards, steps, diff
