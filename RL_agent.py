import torch 
import torch.nn as nn
import numpy as np 

class StockNetwork(nn.Module):
    def __init__(self):
        super().__init__() 
        self.hidden1 = nn.Linear(5, 128)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(128, 128)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(128, 128)
        self.act3 = nn.ReLU()
        self.mean_output = nn.Linear(128, 1)
        self.bell_output = nn.Linear(128, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
            x = self.act1(self.hidden1(x))
            x = self.act2(self.hidden2(x))
            x = self.act3(self.hidden3(x))
            mean = self.mean_output(x)
            bell_param = self.act_output(self.bell_output(x))
            return mean, bell_param
    
    def sample_action(self, state, goal, days):
        mean, bell_param = self.forward(state)
        std = (goal / days) * 0.05
        # Loi normale de paramètre mean,std
        # Loi de bernoulli de paramètre bell_param
        
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.tensor(0.0)
    
        total_stock_target = mean + std * torch.randn_like(mean) # mu + sigma * N(0,1)
        u = np.random.uniform(0, 1, size=bell_param.shape)
        u = torch.tensor(u, dtype=torch.float32)
        bell = (u < bell_param).float()

        pdf_total_stock_target=(1 / (torch.sqrt(2 * np.pi) * std)) * torch.exp(-0.5 * ((total_stock_target - mean) / std) ** 2)
        pdf_bell=bell_param if bell==1 else 1-bell_param
        
        likelihood=pdf_bell*pdf_total_stock_target  #produit car hypothèse d'indépendance
        
        log_density=torch.log(likelihood) #en réalité c'est la log_likelihood mais en 1D c'est la meme chose donc on la nomme comme ca pour enlever toute ambiguité lorque l'on va utiliser la policy gradient method 

    
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
        ]).T


    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
        
        