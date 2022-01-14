import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from .QNetwork import QNetwork
from .agent import Agent


class RandMin(Agent):
    def __init__(
        self,
        env,
        state_dim,
        hidden_dim,
        action_dim,
        seed,
        bs1,
        lr=5e-4,
        eps=1.0,
        eps_end=0.01,
        eps_decay=1e3,
        rbs=100000,
        bs=64,
        gamma=0.99,
        tnuf=4,
    ):
        super().__init__(
            env, action_dim, seed, eps, eps_end, eps_decay, rbs, bs, gamma, tnuf
        )
        self.name = "RandMin" + str(bs1)
        self.batch_size1 = bs1
        self.net_local = [
            QNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        ] * 2
        self.net_target = [
            QNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        ] * 2
        self.optimizer = [
            optim.Adam(self.net_local[i].parameters(), lr=lr) for i in range(2)
        ]
        self.update_net_targets()
        for i in range(2):
            self.net_target[i].eval()

    def select_best_action(self, state):
        with torch.no_grad():
            self.net_local[1].eval()
            action_values = self.net_local[1](state)
            self.net_local[1].train()
        return np.argmax(action_values.cpu().data.numpy())

    def compute_q_targets(self, next_states, idx):
        with torch.no_grad():
            return self.net_target[1](next_states).detach().max(1)[0].unsqueeze(1)

    def update(self):
        states, actions, rewards, next_states, dones = self.memory.sample()

        q_targets_next = self.compute_q_targets(next_states, 0)
        q_targets = rewards + self.gamma * q_targets_next * (1 - dones)
        q_expected = self.compute_q(states, actions, 0)
        loss = F.mse_loss(q_expected, q_targets)

        self.optimizer[0].zero_grad()
        loss.backward()
        self.optimizer[0].step()

        diff = (q_expected - q_targets).cpu().data.numpy()
        idx_smaller = np.nonzero(diff <= 0)[0]
        idx_random = np.random.choice(
            np.arange(self.batch_size), self.batch_size1, replace=False
        )
        idx = np.union1d(idx_smaller, idx_random)
        q_targets_next = q_targets_next[idx]
        q_targets = rewards[idx] + self.gamma * q_targets_next * (1 - dones[idx])
        q_expected = self.compute_q(states[idx], actions[idx], 1)
        loss = F.mse_loss(q_expected, q_targets)

        self.optimizer[1].zero_grad()
        loss.backward()
        self.optimizer[1].step()

    def update_net_targets(self):
        for i in range(2):
            self.net_target[i].load_state_dict(self.net_local[i].state_dict())
