import torch
import numpy as np
from torch import optim

from .QNetwork import QNetwork
from .agent import Agent


class DQN(Agent):
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
        self.name = "DQN"

        self.net_local = [QNetwork(state_dim, hidden_dim, action_dim).to(self.device)]
        self.optimizer = [optim.Adam(self.net_local[0].parameters(), lr=lr)]

        self.net_target = [QNetwork(state_dim, hidden_dim, action_dim).to(self.device)]
        self.net_target[0].load_state_dict(self.net_local[0].state_dict())
        self.net_target[0].eval()

    def select_best_action(self, state):
        self.net_local[0].eval()
        with torch.no_grad():
            action_values = self.net_local[0](state)
        self.net_local[0].train()
        return np.argmax(action_values.cpu().data.numpy())

    def compute_q_targets(self, next_states, idx):
        with torch.no_grad():
            return self.net_target[0](next_states).detach().max(1)[0].unsqueeze(1)

    def update(self):
        self.update_idx(0)

    def update_net_targets(self):
        self.net_target[0].load_state_dict(self.net_local[0].state_dict())
