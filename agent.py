import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from tqdm import tqdm

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
        mean.retain_grad() # Ensure that the mean has a gradient
        
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
        u = np.random.uniform(0,1)
        if u < bell:
            bell = 1
        else:
            bell = 0

        log_density = -0.5*np.log(2*np.pi*std**2)-((action_sampled-mean)**2)/(2*std**2)

        prob = dist.cdf(action_sampled)


        return action_sampled, bell, log_density, prob
    
    def train_model(self, env, num_episodes, simulate_episode, S0, V0, mu, kappa, theta, sigma, rho, days, goal):
        optimizer = optim.Adam(self.parameters(), lr=0.01)

        for episode in tqdm(range(num_episodes)):
            states, actions, log_probs, episode_payoff, _, probabilities = simulate_episode(self, env, S0, V0, mu, kappa, theta, sigma, rho, days, goal)
            optimizer.zero_grad()
            loss = 0.0

            for log_prob in log_probs:
                loss = loss - log_prob  # log_density

            loss = loss * episode_payoff

            if episode % 100 == 0:
                print(f"Episode {episode}: Episode_payoff {episode_payoff}, Loss {loss}")
            loss = torch.tensor(loss, requires_grad=True)
            loss.backward()
            optimizer.step()
    
    

    
        

