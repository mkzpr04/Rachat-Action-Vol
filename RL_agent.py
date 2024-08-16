import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm

class StockNetwork(nn.Module):
    def __init__(self):
        super().__init__() 
        self.hidden1 = nn.Linear(5, 256)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(256, 256)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(256, 256)
        self.act3 = nn.ReLU()
        self.mean_output = nn.Linear(256, 1)
        self.bell_output = nn.Linear(256, 1)
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

        # Vérification de la validité de 'mean'
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.tensor(0.0)

        total_stock_target = mean + std * torch.randn_like(mean) # générer une normale avec numpy
        u = np.random.uniform(0, 1)
        bell = 1 if u < bell.item() else 0
        log_density = -0.5 * torch.log(2 * torch.tensor(np.pi) * (std *std)) - ((total_stock_target - mean) *(total_stock_target - mean)) / (2 * (std *std))
        prob = 0.5 * (1 + torch.erf((total_stock_target - mean) / (std * torch.sqrt(torch.tensor(2.0)))))

        return total_stock_target, bell, log_density, prob
    
    @staticmethod
    def normalize_state(state, days, goal, S0):
        t, S_n, A_n, total_stocks, total_spent = state
        
        # Calcul de la moyenne des prix jusqu'au jour t
        mean_price_t = A_n
        adjusted_S_n = S_n
        # Normalisation
        return np.array([
            t / days, 
            adjusted_S_n / 100,  
            A_n / 100, 
            total_stocks / goal, 
            total_spent / (goal * S0)
        ])

    def train_model(self, env, num_episodes, simulate_episode, S0, V0, mu, kappa, theta, sigma, rho, days, goal):
        optimizer = optim.Adam(self.parameters(), lr=0.01)

        for episode in tqdm(range(num_episodes)):
           states, actions, densities, episode_payoff, _, probabilities = simulate_episode(self, env, S0, V0, mu, kappa, theta, sigma, rho, days, goal)
           optimizer.zero_grad()
           loss = 0.0
           for density in densities:
               loss = loss - density    #log_density
           loss*= episode_payoff

           if episode % 100 == 0:
               print(f"Episode {episode}: Episode_payoff {episode_payoff}, Loss {loss}")
           loss = torch.tensor(loss, requires_grad=True)
           loss.backward()
           optimizer.step()
        
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
