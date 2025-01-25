import torch
import torch.nn as nn
import numpy as np

class StockNetwork(nn.Module):
    def __init__(self,Q):
        super().__init__()
        self.hidden1 = nn.Linear(5, 512)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(512, 512)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(512, 512)
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(512, 512)
        self.act4 = nn.ReLU()
        self.hidden5 = nn.Linear(512, 512)
        self.act5 = nn.ReLU()
        self.output_layer = nn.Linear(512, 2)
        self.act_output = nn.Sigmoid()
        self.scale = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        
    def forward(self, x):
        input = x
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))
        x = self.output_layer(x)
        
        #x[:,0] = torch.minimum(torch.maximum((x[:, 0]+input[:, 0]) * self.Q, torch.tensor(0.0, dtype=torch.float32)),self.Q)
        x[:,0] = self.Q * torch.minimum((1 + x[:, 0]) )

        #x[:,0] = self.Q* torch.minimum( (1+ x[:, 0])*(input[:,0]+1)/self.N, torch.tensor(1))-input[:,3]
        x[:,1] = self.act_output(x[:,1])
        return x

    def sample_action(self, state, goal, days):
        mean, bell = self.forward(state)
        std = (goal / days) * 0.05
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.tensor(0.0)
        total_stock_target = mean + std * torch.randn_like(mean) # mu + sigma * N(0,1)
        log_density = -0.5 * torch.log(2 * torch.tensor(np.pi) * (std *std)) - ((total_stock_target - mean) *(total_stock_target - mean)) / (2 * (std *std)) # vraisemblance de la première action mais il mnanque la proba de sonner la cloche

        #prob = 0.5 * (1 + torch.erf((total_stock_target - mean) / (std * torch.sqrt(torch.tensor(2.0))))) 

        # Calcul de la vraisemblance d'avoir sonné la cloche
        #bell_density = torch.log(1-bell.float())

        #log_density += torch.log(bell_density)

        return total_stock_target, bell, log_density
    
    @staticmethod
    def normalize(state, days, goal, S0):
        t, S_n, A_n, q_n, total_spent = state  # tenseurs
        return torch.vstack([
            torch.full_like(S_n, t / days),  # t normalisé
            S_n / S0,                        # Prix S_n normalisé
            A_n / S0,                        # A_n normalisé
            q_n / goal,                      # Nombre d'actions normalisé
            total_spent / (S0 * goal)        # Total dépensé normalisé
        ])


    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        

