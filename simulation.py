import numpy as np
import torch
from RL_agent import StockNetwork 
from nn import Net
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)


def payoff(A_n, total_spent):
    return 100 * A_n - total_spent

def simulate_price(S_n, X, sigma):
        prices = [S_n]
        for x in X:
            S_n = S_n + sigma * x
            prices.append(S_n)
        return prices

def simulate_price_heston(S0, V0, mu, kappa, theta, sigma, rho, days, nb_paths=2):
    dt = 1 / days
    prices = np.zeros((days + 1, nb_paths))
    volatilities = np.zeros((days + 1, nb_paths))
    
    prices[0, :] = S0
    volatilities[0, :] = V0

    for t in range(1, days + 1):
        dW_S = np.random.normal(0, np.sqrt(dt), nb_paths)
        dW_V = rho * dW_S + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), nb_paths)
        volatilities[t, :] = np.maximum(volatilities[t-1, :] + kappa * (theta - volatilities[t-1, :]) * dt + sigma * np.sqrt(volatilities[t-1, :]) * dW_V, 0)
        prices[t, :] = prices[t-1, :] * np.exp((mu - 0.5 * volatilities[t, :]) * dt + np.sqrt(volatilities[t, :] * dt) * dW_S)

    return prices, volatilities


def execute_step( t, days, prices, total_stocks, total_spent, goal, S0, model, done):
        A_n = S0 if t == 0 else np.mean(prices[1:t+1])
        episode_payoff = 0
        log_densities, probabilities, actions, states = [], [], [], []

        if t == days:
            action = 0 # méthode 2 : action = goal - total_stocks
            episode_payoff = payoff(A_n, total_spent) - (goal - total_stocks) * prices[t]
        elif total_stocks >= goal and t >= 19:
            with torch.no_grad():
                state = StockNetwork.normalize_state((t, prices[t], A_n, total_stocks, total_spent), days, goal, S0)
                state_tensor = torch.tensor(state, dtype=torch.float32)
                total_stock_target, bell, log_density, prob = model.sample_action(state_tensor, goal, days)
                if bell.item() >= 0.5:
                    done = True
                    episode_payoff = payoff(A_n, total_spent) - (goal - total_stocks) * prices[t]
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

def simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal, nb_paths=2):
    total_stocks = np.zeros(nb_paths)
    total_spent = np.zeros(nb_paths)
    prices, volatilities = simulate_price_heston(S0, V0, mu, kappa, theta, sigma, rho, days, nb_paths)
    actions, states, log_densities, probabilities = [], [], [], []
    done, episode_payoff = np.zeros(nb_paths), np.zeros(nb_paths)

    for t in range(days + 1):
        S_tensor_step = torch.tensor(prices[t], dtype=torch.float32)
        A_tensor_step = torch.tensor(np.mean(prices[1:t + 1], axis=0) if t > 0 else np.full(nb_paths, S0), dtype=torch.float32)
        q_tensor_step = torch.tensor(total_stocks, dtype=torch.float32)

        # Normaliser l'état actuel pour Net et obtenir les actions et probabilités
        normalized_state = model.normalize_state(t + 1, S_tensor_step, A_tensor_step, q_tensor_step)
        out = model(normalized_state)
        step_action = out[0]
        step_probability = out[1]

        actions.append(step_action.numpy())
        states.append(normalized_state.numpy())
        log_densities.append(torch.tensor([0.0]).numpy())  # Placeholder, à ajuster si besoin
        probabilities.append(step_probability.numpy())
        
        if t < days:
            total_spent += step_action.numpy() * prices[t + 1]
            total_stocks += step_action.numpy()

        if np.any(done):
            break

    return states, actions, log_densities, episode_payoff, prices, probabilities


def evaluate_policy(model, env, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal):
    total_spent_list = []
    total_stocks_list = []
    A_n_list = []
    payoff_list = []
    final_day_list = []
    actions_list = []

    for _ in range(num_episodes):
        states, actions, log_densities, episode_payoff, prices, probabilities = simulate_episode(
            model, env, S0, V0, mu, kappa, theta, sigma, rho, days, goal
        )
        final_day = len(actions) - 1
        total_spent = sum([a * prices[t] for t, a in enumerate(actions)])
        total_stocks = sum(actions)
        A_n = np.mean(prices[1:final_day + 1])
        episode_payoff_value = env.payoff(A_n, total_spent)
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
    print(f"Actions cumulées: {np.cumsum(actions)}")



def export_csv(states, actions, densities, probabilities, episode_payoff, prices, filename):
    actions_len = len(actions)
    
    # Vérification que les longueurs des listes sont cohérentes
    if len(states) != actions_len or len(densities) != actions_len or len(probabilities) != actions_len:
        raise ValueError("Les longueurs des listes ne correspondent pas.")
    
    # Calcul des listes nécessaires pour le DataFrame
    payoff_list = [episode_payoff] * actions_len
    A_n_list = [state[2] * 100 for state in states]
    total_stocks_list = [state[3] * 100 for state in states]
    total_spent_list = [state[4] * 10000 for state in states]
    veriftotalspent_list = []
    
    # Calcul des valeurs vérifiées
    verifTotalStock = [sum(actions[:t]) for t in range(actions_len)]
    veriftotalspent_list = [0] * actions_len
    cumulative_sum = 0
    for t in range(actions_len - 1):
        cumulative_sum += prices[t + 1] * actions[t]
        veriftotalspent_list[t + 1] = cumulative_sum

    # Création du DataFrame
    data = {
        "Day": list(range(actions_len)),
        "Prices(S_n)": prices[:actions_len],
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
def plot_episode(S_n, A_n, q_n, cloche_n):
    plt.figure(figsize=(14, 7))

    fig, ax1 = plt.subplots(figsize=(14, 7))

    ax1.set_xlabel('Jour')
    ax1.set_ylabel("S_n et A_n en euro (€)", color='black')
    ax1.plot(S_n, label="S_n (Prix de l'action au jour n)", color="blue")
    ax1.plot(A_n, label="A_n (Prix moyen des actions aux jours n)", color="green")
    ax1.tick_params(axis='y', labelcolor='black')

    for i, cloche_value in enumerate(cloche_n):
        if cloche_value == 1:
            ax1.axvline(x=i, color="purple", linestyle='--', label="cloche_n = 1" if i == 0 else "")

    ax2 = ax1.twinx()  
    ax2.set_ylabel('q_n en valeur reel', color='red')
    ax2.plot(q_n, label="q_n (Quantité totale d'actions au jour n)", color="red", linestyle='-')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.tight_layout()
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    plt.title("Évolution de S_n, A_n, q_n et cloche_n au fil du temps")
    plt.grid(True)
    plt.show()
def get_user_choice():
    while True:
        choice = input("Voulez-vous entraîner le modèle (e) ou charger un modèle existant (c) ? (e/c) : ").strip().lower()
        if choice in ['e', 'c']:
            return choice
        print("Choix invalide. Veuillez entrer 'e' pour entraîner ou 'c' pour charger un modèle.")

# Initialisation du modèle et des paramètres

model_name = input("Quel modèle voulez-vous utiliser ? (par exemple : Net, StockNetwork, etc.) : ").strip()

try:
    # Importation dynamique de la classe de modèle
    ModelClass = globals()[model_name]
    model = ModelClass()  # Instanciation du modèle
except KeyError:
    raise ValueError(f"Le modèle '{model_name}' n'est pas reconnu. Assurez-vous que le nom du modèle est correct.")

S0 = 100
V0 = 0.04  # Volatilité initiale
mu = 0.1   # Rendement attendu
kappa = 2.0
theta = 0.04
sigma = 0.1
rho = -0.7
days = 60
goal = 100


# charger un modèle existant ou en entraîner un nouveau
choice = input("Voulez-vous charger un modèle existant (c) ou entraîner un nouveau modèle (e) ? (c/e) : ").strip().lower()

if choice == 'c':
    model_path = input("Entrez le chemin du modèle à charger : ").strip()
    model.load_model(model_path)
    print(f"Modèle chargé depuis {model_path}")

    continue_training = input("Souhaitez-vous continuer l'entraînement du modèle ? (o/n) : ").strip().lower()
    if continue_training == 'o':
        num_episodes = int(input("Entrez le nombre d'épisodes supplémentaires pour l'entraînement : "))
        model.train_model( num_episodes=num_episodes, simulate_episode=simulate_episode, S0=S0, V0=V0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho, days=days, goal=goal)
        
        save_path = input("Entrez le chemin pour sauvegarder le modèle : ").strip()
        model.save_model(save_path)
        print(f"Modèle sauvegardé à {save_path}")

elif choice == 'e':
    # Entraînement d'un nouveau modèle
    num_episodes = int(input("Entrez le nombre d'épisodes pour l'entraînement : "))
    model.train_model( num_episodes=num_episodes, simulate_episode=simulate_episode, S0=S0, V0=V0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho, days=days, goal=goal)
    
    save_path = input("Entrez le chemin pour sauvegarder le modèle : ").strip()
    model.save_model(save_path)
    print(f"Modèle sauvegardé à {save_path}")

# Évaluation de la politique
num_episodes = 50
avg_total_spent, avg_total_stocks, avg_A_n, avg_episode_payoff_value, avg_final_day, actions_list = evaluate_policy(model,  num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal)

print(f"\nRésultats de l'évaluation de la politique sur {num_episodes} épisodes :")
print(f"Total dépensé en moyenne: {avg_total_spent}")
print(f"Total d'actions en moyenne: {avg_total_stocks}")
print(f"Prix moyen des actions: {avg_A_n}")
print(f"Payoff moyen: {avg_episode_payoff_value}")
print(f"Jour final moyen: {avg_final_day}")

# Simulation d'un épisode et exportation des résultats
states, actions, densities, episode_payoff, prices, probabilities = simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal)
export_csv(states, actions, densities, probabilities, episode_payoff, prices, "episode_sans_heston.csv")
bell=0
S_n = prices
A_n = [state[2]*100 for state in states]  
q_n = [state[3] * 100 for state in states] 
cloche_n = [1 if bell==1 else 0 for state in states]  

# Tracé des résultats
plot_episode(S_n, A_n, q_n, cloche_n)
