import torch
import torch.optim as optim
import numpy as np

from .QNetwork import QNetwork
from .agent import Agent


class MaxminDQN(Agent):
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
        self.name = "Maxmin" + str(n)
        self.n = n
        self.net_local = [
            QNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        ] * self.n
        self.net_target = [
            QNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        ] * self.n
        self.optimizer = [
            optim.Adam(self.net_local[i].parameters(), lr=lr) for i in range(self.n)
        ]
        for i in range(self.n):
            self.net_target[i].eval()

    def select_best_action(self, state):
        with torch.no_grad():
            for i in range(self.n):
                self.net_local[i].eval()
                if i == 0:
                    q_min = self.net_local[i](state).clone()
                else:
                    q_min = torch.min(q_min, self.net_local[i](state))
                self.net_local[i].train()
        return np.argmax(q_min.cpu().data.numpy())

    def compute_q_targets(self, next_states, idx):
        for i in range(self.n):
            if i == 0:
                q_min = self.net_target[i](next_states).clone()
            else:
                q_min = torch.min(q_min, self.net_target[i](next_states))
        return q_min.detach().max(1)[0].unsqueeze(1)

    def update(self):
        cur_idx = np.random.randint(self.n)
        self.update_idx(cur_idx)

    def update_net_targets(self):
        for i in range(self.n):
            self.net_target[i].load_state_dict(self.net_local[i].state_dict())
