# RL_agent.py
import torch
import torch.nn as nn
import torch.distributions as dist

class StockNetwork(nn.Module):
    def __init__(self, Q):
        super().__init__()
        self.Q = Q
        self.net = nn.Sequential(
            nn.Linear(5, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # Output: shares (raw), stopping probability
        )
        self.log_std = nn.Parameter(torch.zeros(1))  # Learnable log std

    def forward(self, x):
        x = self.net(x)
        shares = x[:, 0]  # Raw shares component
        stop_prob = torch.sigmoid(x[:, 1])  # Stopping probability
        return shares, stop_prob

    def normalize(self, t, S_t, A_t, q_t, N_t, N, S0):
        return torch.cat([
            (t / N).unsqueeze(-1),
            (S_t / S0).unsqueeze(-1),
            (A_t / S0).unsqueeze(-1),
            (q_t / self.Q).unsqueeze(-1),
            (N_t / (S0 * self.Q)).unsqueeze(-1)
        ], dim=-1)