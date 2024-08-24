import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm

class StockNetwork(nn.Module):
    def __init__(self):
        super().__init__() 
        self.hidden1 = nn.Linear(5, 512)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(512, 512)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(512, 512)
        self.act3 = nn.ReLU()
        self.mean_output = nn.Linear(512, 1)
        self.bell_output = nn.Linear(512, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        mean = self.mean_output(x)
        bell = self.act_output(self.bell_output(x))
        return mean, bell

    def sample_action(self, state, goal, days):
        mean, bell = self.forward(state)
        std = (goal / days) * 0.05

        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.tensor(0.0)

        total_stock_target = mean + std * torch.randn_like(mean)
        u = np.random.uniform(0, 1)
        bell = 1 if u < bell.item() else 0
        log_density = -0.5 * torch.log(2 * torch.tensor(np.pi) * (std *std)) - ((total_stock_target - mean) *(total_stock_target - mean)) / (2 * (std *std)) # vraisemblance de la première action mais il mnanque la proba de sonner la cloche
        if log_density.dim() > 1:
            log_density = log_density.sum()

        prob = 0.5 * (1 + torch.erf((total_stock_target - mean) / (std * torch.sqrt(torch.tensor(2.0))))) 

        # Calcul de la vraisemblance d'avoir sonné la cloche
        bell_prob = bell * prob + (1 - bell) * (1 - prob)
        log_density += torch.log(bell_prob)

        return total_stock_target, bell, log_density, prob
    
    @staticmethod
    def normalize_state(state, days, goal, S0):
        t, S_n, A_n, total_stocks, total_spent = state
        adjusted_A_n = A_n #S_n - A_n
        return np.array([
            t / days, 
            S_n / 100,  
            adjusted_A_n / 100, 
            total_stocks / goal, 
            total_spent / (goal * S0)
        ])

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def prendre_decision(n, S_tensor, A_tensor, q, net):
            # Normalisation des données et appel du modèle
            normalized_input = net.normalize(n + 1, S_tensor[n], A_tensor[n], q[n])
            out = net(normalized_input)
        
            # Sortie du modèle
            nombre_actions = out[0]  # Nombre d'actions à avoir dans le portefeuille à la fin de la journée (n+1)
            prob_sonner_cloche = out[1]  # Probabilité associée à "sonner la cloche"
        
            return nombre_actions, prob_sonner_cloche
