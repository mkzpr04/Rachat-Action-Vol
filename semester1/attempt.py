import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Bernoulli

class StockNetwork(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, 1)  # Mean of Gaussian policy
        self.fc_bell = nn.Linear(hidden_dim, 1)  # Bell probability
        self.log_std = nn.Parameter(torch.zeros(1))  # Learnable log std
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        bell_logit = self.fc_bell(x)
        return mean, bell_logit, self.log_std

    def sample_action(self, state, remaining_stocks, day, current_q):
        mean, bell_logit, log_std = self.forward(state)
        
        # Gaussian policy for stock purchase
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        v_n = dist.rsample()
        v_n = torch.clamp(v_n, min=0.0, max=remaining_stocks)  # Action clipping
        log_prob_v = dist.log_prob(v_n)
        
        # Bernoulli policy for bell (only considered when conditions are met)
        bell_prob = torch.sigmoid(bell_logit)
        valid_bell = (day >= 20) & (current_q >= 100)
        bell = torch.where(valid_bell, Bernoulli(bell_prob).sample(), torch.zeros_like(bell_prob))
        log_prob_bell = torch.where(valid_bell, 
                                  bell * torch.log(bell_prob + 1e-8) + (1 - bell) * torch.log(1 - bell_prob + 1e-8),
                                  torch.zeros_like(bell_prob))
        
        return v_n.detach(), bell.detach(), log_prob_v + log_prob_bell

    @staticmethod
    def normalize(state, days, goal, S0):
        t, S_n, A_n, q_n, total_spent = state
        return torch.stack([
            t / days,
            S_n / S0,
            A_n / S0,
            q_n / goal,
            total_spent / (S0 * goal)
        ], dim=-1)
    



def simulate_episode(model, S0, sigma, days, goal, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q_n = torch.zeros((days+1, batch_size), device=device)
    total_spent = torch.zeros((days+1, batch_size), device=device)
    log_probs = []
    
    # Generate price path
    X = torch.randn((days, batch_size), device=device) * sigma
    S_n = torch.cat([S0*torch.ones(1, batch_size, device=device), S0 + torch.cumsum(X, dim=0)])
    A_n = torch.cumsum(S_n[1:], dim=0) / torch.arange(1, days+1, device=device).unsqueeze(-1)
    A_n = torch.cat([S0*torch.ones(1, batch_size, device=device), A_n])

    for t in range(days):
        current_q = q_n[t]
        remaining = goal - current_q
        
        # Early exit if goal already met
        if (remaining <= 0).all():
            break
            
        state = model.normalize(
            (torch.tensor(t, device=device), 
             S_n[t], 
             A_n[t], 
             current_q, 
             total_spent[t]),
            days, goal, S0
        )
        
        # Sample action
        v_n, bell, log_prob = model.sample_action(state, remaining, t, current_q)
        v_n = torch.clamp(v_n, max=remaining)
        
        # Update states
        q_n[t+1] = current_q + v_n
        total_spent[t+1] = total_spent[t] + v_n * S_n[t+1]
        log_probs.append(log_prob)
        
        # Check bell condition
        if (bell > 0.5).any() and t >= 20:
            payoff = goal * A_n[t+1] - total_spent[t+1]
            return payoff, log_probs, t
    
    # Final payoff if not stopped early
    payoff = goal * A_n[-1] - total_spent[-1]
    return payoff, log_probs, days

def train_model(model, num_episodes, S0, sigma, days, goal, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for episode in range(num_episodes):
        payoff, log_probs, _ = simulate_episode(model, S0, sigma, days, goal)
        loss = -torch.cat(log_probs).sum() * payoff.mean()
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Payoff: {payoff.mean().item()}")



# Parameters
S0 = 100
sigma = 0.6
days = 60
goal = 100

# Initialize and train
model = StockNetwork()
train_model(model, num_episodes=1000, S0=S0, sigma=sigma, days=days, goal=goal)