import numpy as np

class StockEnvironment:

    @staticmethod
    def normalize_state(state, days, goal, S0):
        t, S_n, A_n, total_stocks, total_spent = state
        return np.array([t / days, S_n / 100, A_n / 100, total_stocks / goal, total_spent / (goal * S0)])

    @staticmethod
    def payoff(A_n, total_spent):
        return 100 * A_n - total_spent

    @staticmethod
    def simulate_price_heston(self, S0, V0, mu, kappa, theta, sigma, rho, days):
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