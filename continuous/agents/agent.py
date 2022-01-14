from collections import deque

import torch
import torch.nn.functional as F
import numpy as np
import random

from .replay_buffer import ReplayBuffer


class Agent:
    def __init__(
        self, env, action_dim, seed, eps, eps_end, eps_decay, rbs, bs, gamma, tnuf
    ):
        self.env = env
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_net_update_freq = tnuf
        self.batch_size = bs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.eps = eps
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.memory = ReplayBuffer(action_dim, rbs, bs, seed)

    def select_best_action(self, state):
        raise NotImplementedError

    def select_action(self, state):
        if random.random() > self.eps:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            return self.select_best_action(state)
        else:
            return random.choice(np.arange(self.action_dim))

    def save_transition(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def update_net_targets(self):
        raise NotImplementedError

    def end_episode(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        self.update_net_targets()

    def compute_q_targets(self, next_states, idx):
        raise NotImplementedError

    def compute_q(self, states, actions, idx):
        return self.net_local[idx](states).gather(1, actions)

    def update_idx(self, idx):
        states, actions, rewards, next_states, dones = self.memory.sample()
        q_targets_next = self.compute_q_targets(next_states, idx)
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
        q_expected = self.compute_q(states, actions, idx)
        loss = F.mse_loss(q_expected, q_targets)

        self.optimizer[idx].zero_grad()
        loss.backward()
        self.optimizer[idx].step()

    def update(self):
        raise NotImplementedError

    def train(self, n_episodes, max_steps, n_interval, verbose):
        # Train for a number of frames
        rewards = []
        steps = []
        running_reward = deque(maxlen=n_interval)
        running_step = deque(maxlen=n_interval)
        for i_episode in range(n_episodes):
            state = self.env.reset()
            cur_step = 0
            cur_reward = 0
            while max_steps == -1 or cur_step < max_steps:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.save_transition(state, action, reward, next_state, done)
                if self.batch_size <= len(self.memory):
                    self.update()
                state = next_state
                cur_step += 1
                cur_reward += reward
                if done:
                    break
            self.end_episode()
            running_reward.append(cur_reward)
            running_step.append(cur_step)
            rewards.append(np.mean(running_reward))
            steps.append(np.mean(running_step))
            if verbose and i_episode % 10 == 0:
                print(
                    "Episode {}\tReward: {:.2f}\tStep: {:.2f}".format(
                        i_episode, rewards[-1], steps[-1]
                    )
                )
        return rewards, steps
