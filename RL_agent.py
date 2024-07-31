import torch
import torch.nn as nn
import numpy as np

class StockNetwork(nn.Module):
    def __init__(self):
        """
        Neural network
        """
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
        """
        Forward pass
        return the mean and bell predicted
        """
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        mean = self.mean_output(x)
        bell = self.act_output(self.bell_output(x))
        
        return mean, bell

    def sample_action(self, state, goal, days):
        """
        Sample an action from the state mean and std
        return the action sampled and the bell
        """
        mean, bell = self.forward(state)
        std = (goal / days) * 0.05 # fixed standard deviation to add some randomness to the actions
        dist = torch.distributions.Normal(mean, std)
        action_sampled = dist.sample() 
        bell = torch.bernoulli(bell) # u loi uniforme de (0,1), si u < p alors bell = 1 sinon bell = 0
        log_prob = dist.log_prob(action_sampled) # a la main
        # 1/sqrt(2pi)*(1/std)*exp(-0.5*((x-mean)/std)^2)  vérifier que la moyenne à un gradient
        
        return action_sampled, bell, log_prob

    @staticmethod
    def simulate_price_heston(S0, V0, mu, kappa, theta, sigma, rho, days):
        dt = 1 / days
        prices = [S0]
        volatilities = [V0]
        S = S0
        V = V0
        for _ in range(days):
            dW_S = np.random.normal(0, np.sqrt(dt))
            dW_V = rho * dW_S + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt))
            V = np.maximum(V + kappa * (theta - V) * dt + sigma * np.sqrt(V) * dW_V, 0)  
            S = S * np.exp((mu - 0.5 * V) * dt + np.sqrt(V * dt) * dW_S)
            prices.append(S)
            volatilities.append(V)
        return prices, volatilities

    @staticmethod
    def simulate_price(S_n, X, sigma):
        prices = [S_n]
        for x in X:
            S_n = S_n + sigma * x
            prices.append(S_n)
        return prices
    
    @staticmethod
    def normalize_state(state, days, goal, S0):
        t, S_n, A_n, total_stocks, total_spent = state
        return np.array([t / days, S_n / 100, A_n / 100, total_stocks / goal, total_spent / (goal * S0)])

    @staticmethod
    def payoff(A_n, total_spent):
        return 100 * A_n - total_spent
