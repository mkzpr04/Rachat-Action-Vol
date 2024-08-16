import numpy as np
from RL_agent import StockNetwork
from nn import Net
import torch

class StockEnvironment:

    @staticmethod
    def payoff(A_n, total_spent):
        return 100 * A_n - total_spent
    
    @staticmethod
    def simulate_price(S_n, X, sigma):
        prices = [S_n]
        for x in X:
            S_n = S_n + sigma * x
            prices.append(S_n)
        return prices

    @staticmethod
    def simulate_price_heston(S0, V0, mu, kappa, theta, sigma, rho, days):
        dt = 1 / days
        prices = [S0]
        volatilities = [V0]
        S = S0
        V = V0
        for _ in range(days):
            dW_S = np.random.normal(0, np.sqrt(dt))
            dW_V = rho * dW_S + np.sqrt(1 - rho*rho) * np.random.normal(0, np.sqrt(dt))
            V = np.maximum(V + kappa * (theta - V) * dt + sigma * np.sqrt(V) * dW_V, 0)
            S = S * np.exp((mu - 0.5 * V) * dt + np.sqrt(V * dt) * dW_S)
            prices.append(S)
            volatilities.append(V)
        return prices, volatilities
    
    def execute_step(self, t, days, prices, total_stocks, total_spent, goal, S0, model, done):
        A_n = S0 if t == 0 else np.mean(prices[1:t+1])
        episode_payoff = 0
        log_densities, probabilities, actions, states = [], [], [], []

        if t == days:
            # méthode 2 : action = goal - total_stocks
            episode_payoff = self.payoff(A_n, total_spent) - (goal - total_stocks) * prices[t]
        elif total_stocks >= goal and t >= 19:
            with torch.no_grad():
                state = StockNetwork.normalize_state((t, prices[t], A_n, total_stocks, total_spent), days, goal, S0)
                state_tensor = torch.tensor(state, dtype=torch.float32)
                total_stock_target, bell, log_density, prob = model.sample_action(state_tensor, goal, days)
                if bell.item() >= 0.5:
                    done = True
                    episode_payoff = self.payoff(A_n, total_spent) - (goal - total_stocks) * prices[t]
        else:
            if not done:
                state = StockNetwork.normalize_state((t, prices[t], A_n, total_stocks, total_spent), days, goal, S0)
                state_tensor = torch.tensor(state, dtype=torch.float32)

                with torch.no_grad():
                    total_stock_target, bell, log_density, prob = model.sample_action(state_tensor, goal, days)
                    v_n = total_stock_target.item() * (goal - total_stocks)
                    log_densities.append(log_density)
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
                states.append(StockNetwork.normalize_state((t, prices[t], A_n, total_stocks, total_spent), days, goal, S0))

        return states, actions, log_densities, episode_payoff, done, prices, probabilities

    def prendre_decision(n, S_tensor, A_tensor, q, net):
        # Normalisation des données et appel du modèle
        normalized_input = net.normalize(n + 1, S_tensor[n], A_tensor[n], q[n])
        out = net(normalized_input)
    
        # Sortie du modèle
        nombre_actions = out[0]  # Nombre d'actions à avoir dans le portefeuille à la fin de la journée (n+1)
        prob_sonner_cloche = out[1]  # Probabilité associée à "sonner la cloche"
    
        return nombre_actions, prob_sonner_cloche