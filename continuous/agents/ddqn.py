import numpy as np
import torch
from torch import optim

from .QNetwork import QNetwork
from .agent import Agent


class DDQN(Agent):
    def __init__(
        self,
        env,
        state_dim,
        hidden_dim,
        action_dim,
        seed,
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
        self.name = "DDQN"

        self.net_local = [
            QNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        ] * 2
        self.net_target = [
            QNetwork(state_dim, hidden_dim, action_dim).to(self.device)
        ] * 2
        self.optimizer = [
            optim.Adam(self.net_local[i].parameters(), lr=lr) for i in range(2)
        ]
        for idx in range(2):
            self.net_target[idx].eval()

    def select_best_action(self, state):
        with torch.no_grad():
            for i in range(2):
                self.net_local[i].eval()
                if i == 0:
                    action_values = self.net_local[i](state)
                else:
                    action_values += self.net_local[i](state)
                self.net_local[i].train()
        return np.argmax(action_values.cpu().data.numpy())

    def compute_q_targets(self, next_states, idx):
        with torch.no_grad():
            best_actions = self.net_target[idx](next_states).argmax(1).unsqueeze(1)
            return self.net_target[1 - idx](next_states).gather(1, best_actions)

    def update(self):
        cur_idx = np.random.randint(2)
        self.update_idx(cur_idx)

    def update_net_targets(self):
        for i in range(2):
            self.net_target[i].load_state_dict(self.net_local[i].state_dict())
