import torch
import torch.nn as nn

class StockNetwork(nn.Module):
    def __init__(self, Q):
        super().__init__()
        self.Q = Q
        self.net = nn.Sequential(
            nn.Linear(5, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Linear(512, 2)  # Ensure this outputs 2 features
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        # Ensure we handle both outputs correctly
        x = torch.stack([
            x[:, 0],  # Shares component (raw)
            self.sigmoid(x[:, 1])  # Stopping probability
        ], dim=1)
        return x

    def normalize_state(self, state, N, S0):
        t_tensor, S_t, A_t, q_t, N_t = state
        return torch.cat([
            (t_tensor / N).unsqueeze(-1),
            (S_t / S0).unsqueeze(-1),
            (A_t / S0).unsqueeze(-1),  # Changed normalization
            (q_t / self.Q).unsqueeze(-1),
            (N_t / (S0 * self.Q)).unsqueeze(-1)
        ], dim=-1)