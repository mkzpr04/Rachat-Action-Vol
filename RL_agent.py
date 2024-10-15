import torch 
import torch.nn as nn
import numpy as np 

class StockNetwork(nn.Module):
    def __init__(self, goal):
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
        self.mean_output = nn.Linear(512, 1)
        self.bell_output = nn.Linear(512, 1)
        self.act_output = nn.Sigmoid()
        self.Q = goal
        
    def forward(self, x):
        input = x
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))
        mean = torch.minimum( torch.maximum(self.mean_output(x).squeeze(1) * (self.Q- input[:, 3]), torch.tensor(0.0)),
                              torch.tensor(self.Q))
        bell_param = self.act_output(self.bell_output(x)).squeeze(1)
        return mean, bell_param
    
    def sample_action(self, state, goal, days):
        mean, bell_param = self.forward(state)
        mean = torch.tensor(mean, dtype=torch.float32, requires_grad=True)
        mean = mean.squeeze()
        std = (goal / days) * 0.05
        std = torch.tensor(std, dtype=torch.float32, requires_grad=True)
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    
        total_stock_target = mean + std * torch.randn_like(mean) # mu + sigma * N(0,1)
        total_stock_target = total_stock_target.squeeze()
        u = np.random.uniform(0, 1, size=bell_param.shape)
        u = torch.tensor(u, dtype=torch.float32, requires_grad=True)
        bell = (u < bell_param).float().squeeze()
        bell_param = bell_param.squeeze()


        two_pi = torch.tensor(2 * np.pi, dtype=torch.float32, requires_grad=True)
        pdf_total_stock_target=(1 / (torch.sqrt(two_pi) *std))* torch.exp(-0.5 * ((total_stock_target - mean) / std) ** 2)
        pdf_bell = torch.where(bell == 1, bell_param, 1 - bell_param) 
    
        
        likelihood=pdf_bell*pdf_total_stock_target 
        
        log_density=torch.log(likelihood) 
    

        """
        on peut aussi faire:
        log_stock_purchase = -0.5 * ((stock_purchase - mean) / std) ** 2 - torch.log(std * torch.sqrt(2 * torch.pi))
        log_bell = bell * torch.log(bell_param) + (1 - bell) * torch.log(1 - bell_param)
        log_density = log_bell + log_stock_purchase
        """
    
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
        
        
        