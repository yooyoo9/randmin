import torch
import torch.optim as optim
import numpy as np

from .QNetwork import QNetwork
from .agent import Agent


class AveragedDQN(Agent):
    def __init__(
        self,
        env,
        state_dim,
        hidden_dim,
        action_dim,
        seed,
        n,
        lr=5e-4,
        eps=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        rbs=100000,
        bs=64,
        gamma=0.99,
        tnuf=4,
    ):
        super().__init__(
            env, action_dim, seed, eps, eps_end, eps_decay, rbs, bs, gamma, tnuf
        )
        self.name = "Averaged" + str(n)
        self.n = n
        self.net_local = [QNetwork(state_dim, hidden_dim, action_dim).to(self.device)]
        self.net_target = [
            QNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        ] * self.n
        self.optimizer = [optim.Adam(self.net_local[0].parameters(), lr=lr)]
        self.idx = 0
        for i in range(self.n):
            self.net_target[i].eval()

    def get_q_averaged(self, states):
        with torch.no_grad():
            q_sum = self.net_local[0](states).clone()
            for i in range(1, self.n):
                q_sum += self.net_target[i](states)
            q = q_sum / self.n
            return q

    def select_best_action(self, state):
        q_avg = self.get_q_averaged(state)
        return np.argmax(q_avg.cpu().data.numpy())

    def compute_q_targets(self, next_states, idx):
        q_avg = self.get_q_averaged(next_states)
        return q_avg.detach().max(1)[0].unsqueeze(1)

    def update(self):
        self.update_idx(0)

    def update_net_targets(self):
        self.net_target[self.idx].load_state_dict(self.net_local[0].state_dict())
        self.idx += 1
        self.idx %= self.n
