# simulation.py
import numpy as np
import torch
from RL_agent import StockNetwork
import pandas as pd
import matplotlib.pyplot as plt

def simulate_price(S0, sigma, N, batch_size):
    X = torch.randn((N, batch_size))
    prices = torch.zeros((N+1, batch_size))
    prices[0] = S0
    for t in range(1, N+1):
        prices[t] = prices[t-1] + sigma * X[t-1]
    return prices

class TradingEnvironment:
    def __init__(self, S0, sigma, Q, N, stop_day):
        self.S0 = S0
        self.sigma = sigma
        self.Q = Q
        self.N = N
        self.stop_day = stop_day

    def reset(self, batch_size):
        self.prices = simulate_price(self.S0, self.sigma, self.N, batch_size)
        self.q = torch.zeros((self.N+1, batch_size))
        self.total_spent = torch.zeros((self.N+1, batch_size))
        self.actions = torch.zeros((self.N, batch_size))
        self.stopping_probs = torch.zeros((self.N, batch_size))
        self.A = torch.full((self.N+1, batch_size), self.S0)
        return self._get_state(0)

    def _get_state(self, t):
        batch_size = self.prices.shape[1]
        return (
            torch.full((batch_size,), t, dtype=torch.float32),  # t
            self.prices[t].clone(),
            self.A[t].clone(),
            self.q[t].clone(),
            self.total_spent[t].clone()
        )

    def step(self, t, action, batch_size):
        # Convert constants to tensors with proper device
        device = action.device
        min_shares = torch.tensor(0.0, device=device)
        Q_tensor = torch.tensor(self.Q, dtype=torch.float32, device=device)
        
        # Calculate maximum shares with tensor operations
        max_shares = (Q_tensor - self.q[t]).clamp(min=0)
        
        # Clamp actions with tensor boundaries
        shares_to_buy = torch.clamp(
            action[:, 0],
            min=min_shares,
            max=max_shares
        )
        
        stop_prob = action[:, 1]
        
        # Update portfolio state
        self.q[t+1] = self.q[t] + shares_to_buy
        self.total_spent[t+1] = self.total_spent[t] + shares_to_buy * self.prices[t]
        
        # Update running average price
        self.A[t+1] = (self.A[t] * (t+1) + self.prices[t]) / (t+2)
        
        # Determine early stopping
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        if t >= self.stop_day - 1:
            stop_decision = (stop_prob > 0.5) & (self.q[t+1] >= Q_tensor)
            done = stop_decision | (t == self.N-1)
            
            # Complete remaining purchases if stopped
            remaining = Q_tensor - self.q[t+1]
            self.q[t+1] += remaining
            self.total_spent[t+1] += remaining * self.prices[t]
            
        return self._get_state(t+1), done


def reinforce_train(model, env, episodes=1000, batch_size=64, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_payoff = -float('inf')
    device = next(model.parameters()).device  # Get model device
    
    for episode in range(episodes):
        # Environment reset with proper device
        states = env.reset(batch_size)
        log_probs = []
        episode_payoffs = []
        
        # Adaptive exploration schedule
        noise_scale = max(0.1, 1.5 * (1 - episode/300))
        
        for t in range(env.N):
            # State normalization
            state_normalized = model.normalize_state(
                (states[0].to(device), 
                 states[1].to(device),
                 states[2].to(device),
                 states[3].to(device),
                 states[4].to(device)),
                env.N, env.S0
            )
            
            # Model prediction
            action_params = model(state_normalized)
            assert action_params.shape[1] == 2, "Model must output 2 parameters"
            
            # Action components
            shares_mean = torch.sigmoid(action_params[:, 0])  # [0, 1] scaled
            stop_probs = torch.sigmoid(action_params[:, 1])    # Already [0, 1]
            
            # Shares calculation with exploration
            max_shares = (env.Q - env.q[t].to(device)).clamp(min=0)
            shares_to_buy = shares_mean * max_shares
            noise = torch.randn_like(shares_to_buy) * noise_scale
            shares_to_buy = (shares_to_buy + noise).clamp(min=0, max=max_shares)
            
            # Probability distribution
            dist = torch.distributions.Normal(shares_mean, noise_scale + 1e-5)
            log_prob = dist.log_prob(shares_to_buy)
            log_probs.append(log_prob)
            
            # Create action tensor
            action = torch.stack([
                shares_to_buy.detach(),
                stop_probs.detach()
            ], dim=1)
            
            # Environment step
            states, done = env.step(t, action.cpu(), batch_size)  # Move to CPU for env
            states = tuple(s.to(device) for s in states)  # Move states back to device
            
            if done.any(): break

        # Calculate payoff
        with torch.no_grad():
            final_q = env.q[t+1].to(device)
            final_A = env.A[t+1].to(device)
            final_spent = env.total_spent[t+1].to(device)
            payoff = env.Q * final_A - final_spent
            episode_payoffs.append(payoff)
            
        # Policy optimization
        optimizer.zero_grad()
        policy_loss = -(torch.stack(log_probs).sum(0) * payoff).mean()
        
        if not torch.isnan(policy_loss):
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        
        # Save best model
        current_payoff = torch.stack(episode_payoffs).mean().item()
        if current_payoff > best_payoff:
            best_payoff = current_payoff
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Progress monitoring
        if episode % 20 == 0:
            print(f"Episode {episode:4d} | Avg Payoff: {current_payoff:7.2f}€ | "
                  f"Max Shares: {shares_to_buy.max().item():4.1f}")

    return model

def evaluate(model, env, seed=42, model_type='rl'):
    torch.manual_seed(seed)
    device = next(model.parameters()).device  # Get model device
    states = env.reset(1)
    actions = []
    stop_day = None
    
    # Convert environment tensors to model device
    states = tuple(s.to(device) if isinstance(s, torch.Tensor) else s 
                  for s in states)
    
    for t in range(env.N):
        # State normalization
        if model_type == 'rl':
            state_norm = model.normalize_state(
                (states[0].to(device),
                 states[1].to(device),
                 states[2].to(device),
                 states[3].to(device),
                 states[4].to(device)),
                env.N, env.S0
            )
        else:
            state_norm = model.normalize(
                t+1, 
                states[1].to(device),
                states[2].to(device),
                states[3].to(device)
            )
        
        # Model prediction
        with torch.no_grad():
            action = model(state_norm)
            assert action.shape[1] == 2, "Invalid action dimensions"
        
        # Extract components
        shares = action[0, 0].item()
        stop_prob = action[0, 1].item()
        actions.append(shares)
        
        # Format action for environment
        env_action = torch.tensor([[shares, stop_prob]], device='cpu')
        
        # Environment step
        states, done = env.step(t, env_action, 1)
        states = tuple(s.to(device) if isinstance(s, torch.Tensor) else s 
                      for s in states)
        
        # Check stopping condition
        if done.item() and t >= env.stop_day-1:
            stop_day = t
            break
    
    # Final calculations
    if stop_day is None:
        stop_day = env.N - 1
    
    final_price = env.A[stop_day].item()
    total_spent = env.total_spent[stop_day].item()
    payoff = env.Q * final_price - total_spent
    
    # Generate report
    print(f"\n{' RL Model ' if model_type == 'rl' else ' Legacy Model ':-^40}")
    print(f" Stopping Day: {stop_day + 1}")
    print(f" Final Average Price: {final_price:.2f}€")
    print(f" Total Spent: {total_spent:.2f}€")
    print(f" Payoff: {payoff:.2f}€")
    print(f" Shares Bought: {sum(actions):.1f}/20")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(env.prices[:stop_day+1, 0], label='Stock Price')
    plt.plot(env.A[:stop_day+1, 0], label='Average Price')
    plt.bar(range(stop_day+1), actions[:stop_day+1], 
           alpha=0.3, label='Daily Shares')
    plt.axvline(stop_day, color='r', linestyle='--', 
               label='Stopping Day')
    plt.title(f"Trading Simulation - {model_type.upper()} Model")
    plt.xlabel("Day")
    plt.ylabel("Value (€)")
    plt.legend()
    plt.show()
    
    return payoff


if __name__ == "__main__":
    # Initialize environment and model
    env = TradingEnvironment(S0=100, sigma=0.6, Q=20, N=63, stop_day=22)
    model = StockNetwork(Q=20).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training phase
    trained_model = reinforce_train(model, env, episodes=800)
    
    # Evaluation
    evaluate(trained_model, env, model_type='rl')
    
    # Legacy model evaluation
    from nn import Net
    legacy_model = Net().to('cuda' if torch.cuda.is_available() else 'cpu')
    legacy_model.load_model("trained_model.pt")
    evaluate(legacy_model, env, model_type='legacy')