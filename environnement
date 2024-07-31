import numpy as np

class StockEnvironment:

    @staticmethod
    def normalize_state(state, days, goal, S0):
        t, S_n, A_n, total_stocks, total_spent = state
        return np.array([t / days, S_n / 100, A_n / 100, total_stocks / goal, total_spent / (goal * S0)])

    @staticmethod
    def payoff(A_n, total_spent):
        return 100 * A_n - total_spent
