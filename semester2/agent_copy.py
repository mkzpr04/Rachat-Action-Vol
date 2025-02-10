import torch 
import torch.nn as nn
import numpy as np 

class StockNetwork(nn.Module):
    def __init__(self):
        super().__init__() 
        # 5 Hidden layers as specified
        self.hidden1 = nn.Linear(5, 128)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(128, 128)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(128, 128)
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(128, 128)
        self.act4 = nn.ReLU()
        self.hidden5 = nn.Linear(128, 128)
        self.act5 = nn.ReLU()
        
        # Single output layer with 2 units
        self.final_layer = nn.Linear(128, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))
        output = self.final_layer(x)
        # Split and apply sigmoid only to bell parameter
        mean = output[:, 0].unsqueeze(-1)  # Keep original dimension
        bell_param = self.sigmoid(output[:, 1]).unsqueeze(-1)
        return torch.cat([mean, bell_param], dim=1)
    
    # Keep the rest of the methods unchanged
    @staticmethod
    def normalize(state, days, goal, S0):
        t, S_n, A_n, q_n, total_spent = state
        return torch.vstack([
            torch.full_like(S_n, t / days),
            S_n / S0,
            A_n / S0,
            q_n / goal,
            total_spent / (S0 * goal)
        ]).T

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)