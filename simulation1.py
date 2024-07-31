import numpy as np
import torch
from RL_agent import StockNetwork
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
 
def simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal):
    total_stocks = 0
    total_spent = 0
    prices, volatilities = model.simulate_price_heston(S0, V0, mu, kappa, theta, sigma, rho, days)
    actions = []
    states = []
    densities = []
    probabilities = []
    done = False
    episode_payoff = 0

    for t in range(days + 1):
        A_n = S0 if t == 0 else np.mean(prices[1:t+1])
        if t == days:
            action = 0
            episode_payoff = model.payoff(A_n, total_spent) - (goal - total_stocks) * prices[t]
        if total_stocks >= goal and t >= 19:
            with torch.no_grad():
                state = model.normalize_state((t, prices[t], A_n, total_stocks, total_spent), days, goal, S0)
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action, bell, density, prob = model.sample_action1(state_tensor, goal, days)
                ring_bell = bell.item() >= 0.5
                if ring_bell:
                    done = True
                    episode_payoff = model.payoff(A_n, total_spent) - (goal - total_stocks) * prices[t]
        else:
            ring_bell = False

        if not done:
            state = model.normalize_state((t, prices[t], A_n, total_stocks, total_spent), days, goal, S0)
            state_tensor = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                action, bell, density, prob = model.sample_action1(state_tensor, goal, days)
                action = action.item()
                v_n = action * (goal - total_stocks)
                densities.append(density)
                probabilities.append(prob)
        else:
            v_n = 0

        if t < days:
            total_spent += v_n * prices[t + 1]
            total_stocks += v_n
            actions.append(v_n)
            states.append(state)
        else:
            actions.append(0)
            states.append(model.normalize_state((t, prices[t], A_n, total_stocks, total_spent), days, goal, S0))

        if done:
            break

    return states, actions, densities, episode_payoff, prices, probabilities

def train(model, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal):
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for episode in tqdm(range(num_episodes)):
        states, actions, densities, episode_payoff, _, probabilities = simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal)
        optimizer.zero_grad()
        loss = 0.0

        for density in densities:
            loss = loss - density    #log_density

        loss = loss * episode_payoff
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Episode_payoff {episode_payoff}, Loss {loss}")
        loss = torch.tensor(loss, requires_grad=True)
        loss.backward()
        optimizer.step()

def evaluate_policy(model, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal):
    total_spent_list = []
    total_stocks_list = []
    A_n_list = []
    payoff_list = []
    final_day_list = []
    actions_list = []
    for _ in range(num_episodes):
        states, actions, densities, episode_payoff, prices, probabilities = simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal)
        final_day = len(actions) - 1
        total_spent = sum([a * prices[t] for t, a in enumerate(actions)])
        total_stocks = sum(actions)
        A_n = np.mean(prices[1:final_day + 1])
        episode_payoff_value = model.payoff(A_n, total_spent)
        total_spent_list.append(total_spent)
        total_stocks_list.append(total_stocks)
        A_n_list.append(A_n)
        payoff_list.append(episode_payoff_value)
        final_day_list.append(final_day)
        actions_list.append(actions)
    avg_total_spent = np.mean(total_spent_list)
    avg_total_stocks = np.mean(total_stocks_list)
    avg_A_n = np.mean(A_n_list)
    avg_episode_payoff_value = np.mean(payoff_list)
    avg_final_day = np.mean(final_day_list)
    return avg_total_spent, avg_total_stocks, avg_A_n, avg_episode_payoff_value, avg_final_day, actions_list

def display_optimal_plan(actions, prices):
    print("\nProgramme d'achat optimal:")
    for t, action in enumerate(actions):
        print(f"Jour {t + 1}: Achat de {action} actions")
    print(f"Prix des actions: {prices[:len(actions)]}")
    print(f"Actions cumulées: {np.cumsum([a for a in actions])}")

def export_csv(states, actions, densities, probabilities, episode_payoff, prices, filename):
    actions_len = len(actions)
    payoff_list = [episode_payoff] * actions_len
    A_n_list = [state[2] * 100 for state in states]
    total_stocks_list = [state[3] * 100 for state in states]
    total_spent_list = [state[4] * 10000 for state in states]
    veriftotalspent_list = []

    verifTotalStock = [sum(actions[:t]) for t in range(actions_len)]
    veriftotalspent_list = [0] * actions_len
    cumulative_sum = 0
    for t in range(actions_len - 1):
        cumulative_sum += prices[t+1] * actions[t]
        veriftotalspent_list[t+1] = cumulative_sum
    data = {
        "Day": list(range(0, actions_len)),
        "Prices(S_n)": prices,
        "A_n": A_n_list,
        "Total_Stocks": total_stocks_list,
        "Total_Spent": total_spent_list,
        "actions": actions,
        "density": densities,
        "Probabilité action": [prob.item() for prob in probabilities],
        "payoff": payoff_list,
        "verifTotalStock": verifTotalStock,
        "VerifTotalSpent": veriftotalspent_list
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

# Initialisation du modèle et des paramètres
model = StockNetwork()
S0 = 100
V0 = 0.04  # Initial volatility
mu = 0.1   # Expected return
kappa = 2.0
theta = 0.04
sigma = 0.1
rho = -0.7
days = 60
goal = 100

# Entraînement du modèle
train(model, num_episodes=500, S0=S0, V0=V0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho, days=days, goal=goal)

# Évaluation de la politique
num_episodes = 50
avg_total_spent, avg_total_stocks, avg_A_n, avg_episode_payoff_value, avg_final_day, actions_list = evaluate_policy(model, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal)

# Simulation d'un épisode et exportation des résultats
states, actions, densities, episode_payoff, prices, probabilities = simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal)
export_csv(states, actions, densities, probabilities, episode_payoff, prices, "episode1.csv")