import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ASRNet(nn.Module):
    def __init__(self):
        super(ASRNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),  # Entrée : [n/N, S/S0, A/S0, q/Q]
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.output_v = nn.Linear(64, 1)  # Nombre d'actions
        self.output_p = nn.Linear(64, 1)  # Probabilité d'arrêt

    def forward(self, x):
        x = self.fc(x)
        v = self.output_v(x)
        p = torch.sigmoid(self.output_p(x))
        return v, p


def simulate_episode(model, price_path, Q, S0, min_stop, max_days):
    n_days = len(price_path)
    S, A = S0, S0
    q, x = 0, 0
    rewards = []
    stop_day = None

    for n in range(n_days):
        if n >= max_days:
            break

        # État en tant que tenseur PyTorch
        input_state = torch.tensor([n / max_days, S / S0, A / S0, q / Q], dtype=torch.float32, requires_grad=True)
        v, p = model(input_state)

        # Contraindre les actions restantes
        remaining_actions = Q - q
        if n == n_days - 1:  # Dernier jour
            v = remaining_actions
        else:
            v = torch.clamp(v, torch.tensor(0), remaining_actions)

        # Empêcher l'arrêt prématuré
        stop = torch.bernoulli(p).item() if q == Q and n >= min_stop else 0

        # Mise à jour des variables
        q += v
        x += v * S
        A = ((A * n) + S) / (n + 1)

        # Récompense si on s'arrête
        if stop:
            payoff = Q * A - x
            rewards.append(payoff)
            stop_day = n
            break

        # Mise à jour du prix
        S = price_path[n]

    # Si non arrêté correctement
    if stop_day is None or q < Q:
        payoff = -x
        rewards.append(payoff)

    return rewards, stop_day


def train_model(model, optimizer, price_simulator, epochs, batch_size, Q, S0, sigma, min_stop, max_days):
    for epoch in range(epochs):
        total_rewards = []

        for _ in range(batch_size):
            # Simuler un chemin de prix
            price_path = price_simulator(S0, sigma, max_days)
            rewards, _ = simulate_episode(model, price_path, Q, S0, min_stop, max_days)
            total_rewards.extend(rewards)

        # Convertir en tenseur
        rewards = torch.tensor(total_rewards, dtype=torch.float32, requires_grad=True)
        loss = -torch.mean(rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

def simulate_price(S0, sigma, N):
    return np.cumsum(np.random.normal(0, sigma, N)) + S0



# Paramètres
N = 63
S0 = 45
sigma = 0.6
Q = 20
min_stop = 22

# Initialisation
model = ASRNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Entraînement
train_model(
    model, optimizer, simulate_price, epochs=10, batch_size=32,
    Q=Q, S0=S0, sigma=sigma, min_stop=min_stop, max_days=N
)
def evaluate_policy(model, price_simulator, num_simulations, Q, S0, sigma, min_stop, max_days):
    total_actions = []
    total_payoffs = []
    stopping_days = []

    for _ in range(num_simulations):
        price_path = price_simulator(S0, sigma, max_days)
        rewards, stop_day = simulate_episode(model, price_path, Q, S0, min_stop, max_days)

        total_payoffs.append(rewards[0].detach().numpy())  # Payoff final
        stopping_days.append(stop_day.detach().numpy())  # Jour d'arrêt

    # Résultats globaux
    avg_actions = np.mean([np.sum(actions) for actions in total_actions])
    avg_payoff = np.mean(total_payoffs)
    avg_stopping_day = np.mean(stopping_days)

    print(f"Politique évaluée sur {num_simulations} simulations :")
    print(f"- Moyenne des actions totales achetées : {avg_actions:.2f}")
    print(f"- Moyenne des payoffs : {avg_payoff:.2f}")
    print(f"- Moyenne du jour d'arrêt : {avg_stopping_day:.2f}")

    return total_actions, total_payoffs, stopping_days



# Évaluer la politique apprise
evaluate_policy(
    model,
    simulate_price,
    num_simulations=100,
    Q=20,
    S0=45,
    sigma=0.6,
    min_stop=20,
    max_days=63
)
