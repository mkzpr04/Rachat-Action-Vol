import torch 
import torch.nn as nn
import numpy as np 

class StockNetwork(nn.Module):
    def __init__(self, Q):
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
        self.Q = torch.tensor(Q, dtype=torch.float32)
        self.scale = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        
    def forward(self, x):
        input = x
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))
        x = self.output_layer(x)
        
        x[:,0] = torch.minimum(torch.maximum((x[:, 0]+input[:, 0]) * self.Q, torch.tensor(0.0, dtype=torch.float32)),
                              self.Q)
        #x[:,0] = self.Q* torch.minimum( (1+ x[:, 0])*(input[:,0]+1)/self.N, torch.tensor(1))-input[:,3]
        x[:,1] = self.act_output(x[:,1])

        """
        # Utilisation de self.scale pour ajuster la sortie de mean_output
        mean_output = x[:, 0]
        adjusted_mean_output = mean_output * self.scale
        
        x[:, 0] = torch.minimum(
            torch.maximum((adjusted_mean_output + input[:, 0]) * self.Q, torch.tensor(0.0)),
            torch.tensor(self.Q)
        )
        
        # Utilisation de self.scale pour ajuster la sortie de bell_output (si nécessaire)
        bell_output = x[:, 1]
        adjusted_bell_output = bell_output * self.scale
        
        x[:, 1] = torch.sigmoid(adjusted_bell_output)"""
        return x.T
    
    def sample_action(self, state, Q, N):
        x = self.forward(state)
        mean = x[:,0]

        mean = mean.float()
        std = (Q / N) * 0.05
        std = torch.tensor(std, dtype=torch.float32, device=mean.device)
    
        total_stock_target = mean + std * torch.randn_like(mean) # mu + sigma * N(0,1)
        u = np.random.uniform(0, 1, size=mean.shape)
        u = torch.tensor(u, dtype=torch.float32, device=mean.device)
        #bell = (u < x[:,1]).float()

        pdf_total_stock_target =  torch.exp(-0.5 * ((total_stock_target - mean) / std) ** 2) / ((np.sqrt(2*np.pi) * std))  
        #pdf_bell = torch.where(bell == 1, x[:,1], 1 - x[:,1]) 
    
        density=pdf_total_stock_target 

        log_density=torch.log(density) 
    
        return total_stock_target, x[:,1], log_density

    
    @staticmethod
    def normalize(state, N, Q, S0):
        t, S_n, A_n, q_n, total_spent = state  # tenseurs
        return torch.vstack([
            torch.full_like(S_n, t / N),  # t normalisé
            S_n / S0,                        # Prix S_n normalisé
            A_n / S0,                        # A_n normalisé
            q_n / Q,                      # Nombre d'actions normalisé
            total_spent / (S0 * Q)        # Total dépensé normalisé
        ]).T
    
    @staticmethod
    def normalize1(state, N, Q, S0):
        n, S_n, A_n, q_n, total_spent = state  # tenseurs
        return torch.vstack([
            torch.full_like(S_n, (n / N)-1/2),  # t normalisé
            (S_n-S0 )/ S0,                        # Prix S_n normalisé
            (A_n-S_n) / S0,                        # A_n normalisé
            (q_n / Q) -1/2,                      # Nombre d'actions normalisé
            total_spent / (S0 * Q)        # Total dépensé normalisé
        ]).T

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
        
        