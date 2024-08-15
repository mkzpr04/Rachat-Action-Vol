import numpy as np
import torch
from RL_agent import StockNetwork 
from environnement import StockEnvironment  
from nn import Net
import torch.optim as optim
from tqdm import tqdm
import pandas as pd

np.random.seed(0)
torch.manual_seed(0)

def simulate_episode(model, env, S0, V0, mu, kappa, theta, sigma, rho, days, goal):
    total_stocks = 0
    total_spent = 0
    X = np.random.normal(0, 1, days)
    prices = env.simulate_price(S0, X, sigma)
    actions, states, log_densities, probabilities = [], [], [], []
    done, episode_payoff = False, 0

    for t in range(days+1):
        step_states, step_actions, step_log_densities, episode_payoff, done, step_prices, step_probabilities = env.execute_step(
            t, days, prices, total_stocks, total_spent, goal, S0, model, done)
        
        states.extend(step_states)
        actions.extend(step_actions)
        log_densities.extend(step_log_densities)
        probabilities.extend(step_probabilities)
        
        if t < days:
            total_spent += step_actions[0] * prices[t+1]
            total_stocks += step_actions[0]

        if done:
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
        states, actions, log_densities, episode_payoff, prices, probabilities = simulate_episode(model, env, S0, V0, mu, kappa, theta, sigma, rho, days, goal)
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

# Initialisation de l'environnement
env = StockEnvironment()

# charger un modèle existant ou en entraîner un nouveau
choice = input("Voulez-vous charger un modèle existant (c) ou entraîner un nouveau modèle (e) ? (c/e) : ").strip().lower()

if choice == 'c':
    model_path = input("Entrez le chemin du modèle à charger : ").strip()
    model.load_model(model_path)
    print(f"Modèle chargé depuis {model_path}")

    continue_training = input("Souhaitez-vous continuer l'entraînement du modèle ? (o/n) : ").strip().lower()
    if continue_training == 'o':
        num_episodes = int(input("Entrez le nombre d'épisodes supplémentaires pour l'entraînement : "))
        model.train_model(env, num_episodes=num_episodes, simulate_episode=simulate_episode, S0=S0, V0=V0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho, days=days, goal=goal)
        
        save_path = input("Entrez le chemin pour sauvegarder le modèle : ").strip()
        model.save_model(save_path)
        print(f"Modèle sauvegardé à {save_path}")

elif choice == 'e':
    # Entraînement d'un nouveau modèle
    num_episodes = int(input("Entrez le nombre d'épisodes pour l'entraînement : "))
    model.train_model(env, num_episodes=num_episodes, simulate_episode=simulate_episode, S0=S0, V0=V0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho, days=days, goal=goal)
    
    save_path = input("Entrez le chemin pour sauvegarder le modèle : ").strip()
    model.save_model(save_path)
    print(f"Modèle sauvegardé à {save_path}")

# Évaluation de la politique
num_episodes = 50
avg_total_spent, avg_total_stocks, avg_A_n, avg_episode_payoff_value, avg_final_day, actions_list = evaluate_policy(model, env, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal)

print(f"\nRésultats de l'évaluation de la politique sur {num_episodes} épisodes :")
print(f"Total dépensé en moyenne: {avg_total_spent}")
print(f"Total d'actions en moyenne: {avg_total_stocks}")
print(f"Prix moyen des actions: {avg_A_n}")
print(f"Payoff moyen: {avg_episode_payoff_value}")
print(f"Jour final moyen: {avg_final_day}")

# Simulation d'un épisode et exportation des résultats
states, actions, densities, episode_payoff, prices, probabilities = simulate_episode(model, env, S0, V0, mu, kappa, theta, sigma, rho, days, goal)
export_csv(states, actions, densities, probabilities, episode_payoff, prices, "episode_sans_heston.csv")
