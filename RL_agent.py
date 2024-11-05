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
        self.Q = Q
        self.scale = nn.Parameter(torch.tensor(0.0, requires_grad=True))
        
    def forward(self, x):
        input = x
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act4(self.hidden4(x))
        x = self.act5(self.hidden5(x))
        x = self.output_layer(x)
        
        x[:,0] = torch.minimum( torch.maximum( (x[:, 0]+input[:, 0]) * self.Q, torch.tensor(0.0)),
                              torch.tensor(self.Q))
        #x[:,0] = self.Q* torch.minimum( (1+ x[:, 0])*(input[:,0]+1)/self.N, torch.tensor(1))-input[:,3]
        x[:,1] = self.act_output(x[:, 1])

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
        mean, bell_param = self.forward(state)

        mean = torch.tensor(mean, dtype=torch.float32, requires_grad=True)
        std = (Q / N) * 0.05
        std = torch.tensor(std, dtype=torch.float32, requires_grad=True)
    
        total_stock_target = mean + std * torch.randn_like(mean) # mu + sigma * N(0,1)
        u = np.random.uniform(0, 1, size=bell_param.shape)
        u = torch.tensor(u, dtype=torch.float32, requires_grad=True)
        bell = (u < bell_param).float()


        two_pi = torch.tensor(2 * np.pi, dtype=torch.float32, requires_grad=True)
        pdf_total_stock_target=(1 / (torch.sqrt(two_pi) *std))* torch.exp(-0.5 * ((total_stock_target - mean) / std) ** 2)
        pdf_bell = torch.where(bell == 1, bell_param, 1 - bell_param) 
    
        density=pdf_bell*pdf_total_stock_target 

        log_density=torch.log(density) 
    

        """
        on peut aussi faire:
        log_stock_purchase = -0.5 * ((stock_purchase - mean) / std) ** 2 - torch.log(std * torch.sqrt(2 * torch.pi))
        log_bell = bell * torch.log(bell_param) + (1 - bell) * torch.log(1 - bell_param)
        log_density = log_bell + log_stock_purchase
        """
    
        return total_stock_target, bell, log_density

    
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
        
        
        