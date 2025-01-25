# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 23:50:26 2024

@author: bartb
"""

import numpy as np
import torch
from RL_agent import StockNetwork
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from nn import Net

# Set the batch size here
BATCH_SIZE = 2  # You can change this value to any integer

def simulate_price(X, sigma, S0):
    # Simulation des prix
    S_n = np.zeros((X.shape[0] + 1, X.shape[1]))
    S_n[0, :] = S0
    S_n[1:] = S0 + np.cumsum(sigma * X, axis=0)
    return S_n

def payoff(A_n, total_spent):
    return 20 * A_n - total_spent

def simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal, flag, batch_size=BATCH_SIZE):
    q_n = torch.zeros((days+1, batch_size), dtype=torch.float32)
    q_n[days, :] = goal
    A_n = torch.zeros((days+1, batch_size), dtype=torch.float32)
    actions = torch.zeros((days+1, batch_size), dtype=torch.float32)
    bell_signals = torch.zeros((days+1, batch_size), dtype=torch.float32)
    total_spent = torch.zeros((days+1, batch_size), dtype=torch.float32)
    log_densities = torch.zeros((days+1, batch_size), dtype=torch.float32)
    probabilities = torch.zeros((days+1, batch_size), dtype=torch.float32)
    episode_payoff = torch.full((batch_size,), float('nan'), dtype=torch.float32)

    if not flag:  # lorsqu'on évalue le model
        np.random.seed(0)
    X = np.random.normal(0, 1, (days, batch_size))
    S_n = simulate_price(X, sigma, S0)
    S_n = torch.tensor(S_n, dtype=torch.float32)

    for t in range(days):
        A_n[t, :] = torch.mean(S_n[1:t+1, :], axis=0) if t > 0 else S0
        if isinstance(model, Net):
            state = model.normalize(t+1, S_n[t, :], A_n[t, :], q_n[t, :])
        else:
            state = model.normalize((t, S_n[t, :], A_n[t, :], q_n[t, :], total_spent[t, :]), days, goal, S0)
        state_tensor = torch.tensor(state, dtype=torch.float32)

        log_density = None
        prob = 0

        with torch.no_grad():
            if flag:
                total_stock_target, bell, log_density, prob = model.sample_action(state_tensor, goal, days)
            else:
                if isinstance(model, Net):
                    etat = model.forward(state_tensor)
                    total_stock_target = etat[0]
                    bell = etat[1]
                else:
                    total_stock_target, bell = model.forward(state_tensor)

        # MAJ des états
        q_n[t+1, :] = total_stock_target if t < days-1 else goal
        v_n = q_n[t+1, :] - q_n[t, :]
        total_spent[t+1, :] = total_spent[t, :] + v_n * S_n[t+1, :]
        log_densities[t, :] = log_density if log_density is not None else 0
        probabilities[t, :] = prob
        actions[t, :] = v_n
        bell_signals[t, :] = bell
        condition = ((bell_signals[t, :] >= 0.5) & (t >= 21) & (q_n[t, :] >= goal)) | (t+1 >= days)
        if condition.any():
            bell_signals[t, :] = bell_signals[t, :] + 1
            q_n[t+1:, condition] = q_n[t, condition].unsqueeze(0) #ici importtant de verif
            not_assigned = torch.isnan(episode_payoff)
            episode_payoff[not_assigned] = payoff(A_n[t, not_assigned], total_spent[t, not_assigned])
            A_n[days, condition] = torch.mean(S_n[1:days + 1, :], axis=0)[condition]

    condition = q_n[days, :] < goal
    if condition.any():
        A_n[days, condition] = torch.mean(S_n[1:days + 1, :], axis=0)[condition]
        final_adjustment = goal - q_n[days, condition]
        total_spent[days, condition] += final_adjustment * S_n[days, condition]
        actions[-1, condition] += final_adjustment
        q_n[days, condition] = goal
        episode_payoff = payoff(A_n[days, :], total_spent[days, :])

    return S_n, A_n, q_n, total_spent, actions, log_densities, probabilities, bell_signals, episode_payoff

def train_model(model, simulate_episode, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal, batch_size=BATCH_SIZE):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for episode in tqdm(range(num_episodes)):
        results = simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal, flag=True, batch_size=batch_size)
        S_n, A_n, q_n, total_spent, actions, log_densities, probabilities, bell_signals, episode_payoff = results

        optimizer.zero_grad()
        loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        loss = loss - torch.sum(log_densities)
        loss *= torch.mean(episode_payoff)

        if episode % 50 == 0:
            print(f"Episode {episode}: Average Episode Payoff {np.mean(episode_payoff.detach().numpy())}, Loss {loss.item()}")

        loss.backward()
        optimizer.step()

def evaluate_policy(model, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal, batch_size=BATCH_SIZE):
    total_spent_list = []
    total_stocks_list = []
    A_n_list = []
    payoff_list = []
    final_day_list = []
    actions_list = []

    for _ in range(num_episodes):
        results = simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal, flag=False, batch_size=batch_size)
        S_n, A_n, q_n, total_spent, actions, log_densities, probabilities, bell_signals, episode_payoff = results
        final_day_indices = (bell_signals > 1).nonzero(as_tuple=False)
        for batch_idx in range(batch_size):
            final_day = final_day_indices[final_day_indices[:, 1] == batch_idx][0][0].item() if (final_day_indices[:, 1] == batch_idx).any() else days
            total_spent_single = total_spent[final_day, batch_idx]
            total_stocks = q_n[final_day, batch_idx]
            total_spent_list.append(total_spent_single.item())
            total_stocks_list.append(total_stocks.item())
            A_n_list.append(A_n[final_day, batch_idx].item())
            payoff_list.append(episode_payoff[batch_idx].item())
            final_day_list.append(final_day)
            actions_list.append(actions[:, batch_idx].detach().numpy())

    avg_total_spent = np.mean(total_spent_list)
    avg_total_stocks = np.mean(total_stocks_list)
    avg_A_n = np.mean(A_n_list)
    avg_payoff = np.mean(payoff_list)
    avg_final_day = np.mean(final_day_list)
    max_len_actions = max([len(a) for a in actions_list])
    padded_actions = np.array([np.pad(a, (0, max_len_actions - len(a)), 'constant', constant_values=0) for a in actions_list])
    avg_actions = np.mean(padded_actions, axis=0)

    return avg_total_spent, avg_total_stocks, avg_A_n, avg_payoff, avg_final_day, avg_actions

def display_optimal_plan(actions, S_n):
    print("\nProgramme d'achat optimal:")
    for t, action in enumerate(actions):
        print(f"Jour {t + 1}: Achat de {action} actions")
    print(f"Prix des actions: {S_n[:len(actions)]}")
    print(f"Actions cumulées: {np.cumsum(actions)}")

def export_csv(actions, episode_payoff, S_n, A_n, q_n, total_spent, filename, sample_index=0):
    actions_sample = actions[:, sample_index]
    S_n_sample = S_n[:, sample_index]
    A_n_sample = A_n[:, sample_index]
    q_n_sample = q_n[:, sample_index]
    total_spent_sample = total_spent[:, sample_index]
    episode_payoff_sample = episode_payoff[sample_index]

    actions_len = len(actions_sample)

    if len(A_n_sample) != actions_len or len(q_n_sample) != actions_len or len(total_spent_sample) != actions_len:
        raise ValueError("Les longueurs des listes ne correspondent pas.")

    payoff_list = [0] * (actions_len - 1) + [episode_payoff_sample.item()]
    data = {
        "Day": list(range(actions_len)),
        "Prices(S_n)": S_n_sample.numpy(),
        "A_n": A_n_sample.numpy(),
        "Total_Stocks": q_n_sample.numpy(),
        "Total_Spent": total_spent_sample.numpy(),
        "actions": actions_sample.numpy(),
        "payoff": payoff_list,
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def plot_episode(S_n, A_n, q_n, cloche_n, sample_index=0):
    S_n_sample = S_n[:, sample_index]
    A_n_sample = A_n[:, sample_index]
    q_n_sample = q_n[:, sample_index]
    cloche_n_sample = cloche_n[:, sample_index]

    plt.figure(figsize=(14, 7))

    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.set_xlabel('Jour')
    ax1.set_ylabel("S_n et A_n en euro (€)", color='black')
    ax1.plot(S_n_sample, label="S_n (Prix de l'action au jour n)", color="blue")
    ax1.plot(A_n_sample, label="A_n (Prix moyen des actions aux jours n)", color="green")
    ax1.tick_params(axis='y', labelcolor='black')
    verification = True
    for i, cloche_value in enumerate(cloche_n_sample):
        if float(cloche_value.item()) >= 1 and verification:
            ax1.axvline(x=i, color="purple", linestyle='--', label="cloche_n >= 1" if i == 0 else "")
            verification = False
    ax2 = ax1.twinx()
    ax2.set_ylabel('q_n en valeur réelle', color='red')
    ax2.plot(q_n_sample, label="q_n (Quantité totale d'actions au jour n)", color="red", linestyle='-')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.tight_layout()
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title(f"Évolution de S_n, A_n, q_n et cloche_n au fil du temps (Sample {sample_index})")
    plt.grid(True)
    plt.show()

def get_user_choice(prompt, valid_choices):
    """ Function to get a validated user choice from a list of valid choices """
    while True:
        choice = input(prompt).strip().lower()
        if choice in valid_choices:
            return choice
        print(f"Choix invalide. Veuillez entrer une des options suivantes : {', '.join(valid_choices)}")

# Initialisation du modèle et des paramètres

model_name = input("Quel modèle voulez-vous utiliser ? (par exemple : Net, StockNetwork, etc.) : ").strip()

try:
    # Importation dynamique de la classe de modèle
    ModelClass = globals()[model_name]
    model = ModelClass()  # Instanciation du modèle
except KeyError:
    raise ValueError(f"Le modèle '{model_name}' n'est pas reconnu. Assurez-vous que le nom du modèle est correct.")

# Paramètres du modèle
S0 = 45
sigma = 0.6
days = 63
goal = 20

V0 = 0.04  # Volatilité initiale
mu = 0.1   # Rendement attendu
kappa = 2.0
theta = 0.04
rho = -0.7

# Choix de l'utilisateur pour charger ou entraîner le modèle
choice = get_user_choice("Voulez-vous charger un modèle existant (c) ou entraîner un nouveau modèle (e) ? (c/e) : ", ['c', 'e'])

if choice == 'c':
    model_path = input("Entrez le chemin du modèle à charger : ").strip()
    try:
        model.load_model(model_path)
        print(f"Modèle chargé depuis {model_path}")

        continue_training = get_user_choice("Souhaitez-vous continuer l'entraînement du modèle ? (o/n) : ", ['o', 'n'])
        if continue_training == 'o':
            num_episodes = int(input("Entrez le nombre d'épisodes supplémentaires pour l'entraînement : "))
            train_model(model, simulate_episode=simulate_episode, num_episodes=num_episodes, S0=S0, V0=V0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho, days=days, goal=goal, batch_size=BATCH_SIZE)

            save_path = input("Entrez le chemin pour sauvegarder le modèle : ").strip()
            model.save_model(save_path)
            print(f"Modèle sauvegardé à {save_path}")

    except Exception as e:
        print(f"Erreur lors du chargement ou de l'entraînement du modèle : {e}")

elif choice == 'e':
    # Entraînement d'un nouveau modèle
    num_episodes = int(input("Entrez le nombre d'épisodes pour l'entraînement : "))
    train_model(model, simulate_episode=simulate_episode, num_episodes=num_episodes, S0=S0, V0=V0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho, days=days, goal=goal, batch_size=BATCH_SIZE)

    save_path = input("Entrez le chemin pour sauvegarder le modèle : ").strip()
    try:
        model.save_model(save_path)
        print(f"Modèle sauvegardé à {save_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")

# Évaluation de la politique
num_episodes = 100
avg_total_spent, avg_total_stocks, avg_A_n, avg_payoff, avg_final_day, avg_actions = evaluate_policy(
    model, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal, batch_size=BATCH_SIZE
)
print(f"\nRésultats de l'évaluation de la politique sur {num_episodes} épisodes :")
print(f"Total dépensé en moyenne: {avg_total_spent}")
print(f"Total d'actions en moyenne: {avg_total_stocks}")
print(f"Prix moyen des actions: {avg_A_n}")
print(f"Payoff moyen: {avg_payoff}")
print(f"Jour final moyen: {avg_final_day}")
print(f"Actions moyennes par jour: {avg_actions}")

# Simulation d'un épisode et exportation des résultats
S_n, A_n, q_n, total_spent, actions, log_densities, probabilities, cloche_n, episode_payoff = simulate_episode(
    model, S0, V0, mu, kappa, theta, sigma, rho, days, goal, flag=False, batch_size=BATCH_SIZE
)

# Exportation des résultats pour un échantillon spécifique
sample_index = int(input(f"Entrez l'indice du sample à exporter (0 à {BATCH_SIZE - 1}) : "))
export_csv(actions, episode_payoff, S_n, A_n, q_n, total_spent, "episode_sans_heston.csv", sample_index=sample_index)

# Tracé des résultats pour le même échantillon
plot_episode(S_n, A_n, q_n, cloche_n, sample_index=sample_index)
