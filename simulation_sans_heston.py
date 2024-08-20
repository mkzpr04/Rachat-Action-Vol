import numpy as np
import torch
from RL_agent import StockNetwork 
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt 



def simulate_price(S_n, X, sigma):
    prices = [S_n]
    for x in X:
        S_n = S_n + sigma * x
        prices.append(S_n)
    return prices

def payoff(A_n, total_spent):
    return 100 * A_n - total_spent

def simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal):
    np.random.seed(0)
    q_n, A_n, actions, bell_signals, total_spent, X = np.zeros(days+1), np.zeros(days+1), np.zeros(days+1), np.zeros(days+1), 0, np.random.normal(0, 1, days)
    prices = simulate_price(S0, X, sigma)
    states, actions, log_densities, probabilities = [], [], [], []
    
    for t in range(days):
        A_n[t] = np.mean(prices[1:t+1]) if t > 0 else S0  
        state = StockNetwork.normalize_state((t, prices[t], A_n[t], q_n[t], total_spent), days, goal, S0)
        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            total_stock_target, bell, log_density, prob = model.sample_action(state_tensor, goal, days)
            q_n[t+1] = total_stock_target.item() * (goal - q_n[t]) if t < days - 1 else goal
            v_n = q_n[t+1] - q_n[t] 
            total_spent += v_n * prices[t]
            log_densities.append(log_density)
            probabilities.append(torch.exp(-prob).item())
            actions.append(v_n) # todo passer en array
            bell_signals[t] = bell

            if (bell >= 0.5 and t >= 19 and q_n[t+1]>= goal):
                return prices, A_n, q_n, total_spent, actions, probabilities, bell_signals, payoff(np.mean(prices), total_spent)
    # t = days
    if q_n[days] < goal:
        final_adjustment = goal - q_n[days]
        total_spent += final_adjustment * prices[days]  
        actions[-1] += final_adjustment 
        q_n[days] = goal

    return prices, A_n, q_n, total_spent, actions, log_densities, probabilities, bell_signals, payoff(np.mean(prices), total_spent)


        
"""
        A_n = S0 if t == 0 else np.mean(prices[1:t+1])
        if t == days:
            action = 0
            episode_payoff = payoff(A_n, total_spent) 
        if total_stocks >= goal and t >= 19:
            with torch.no_grad():
                state = StockNetwork.normalize_state((t, prices[t], A_n, total_stocks, total_spent), days, goal, S0)
                state_tensor = torch.tensor(state, dtype=torch.float32)
                action, bell, log_prob = model.sample_action(state_tensor, goal, days)
                ring_bell = bell.item() >= 0.5
                if ring_bell:
                    done = True
                    episode_payoff = payoff(A_n, total_spent) 
        else:
            ring_bell = False

        if not done:
            state = StockNetwork.normalize_state((t, prices[t], A_n, total_stocks, total_spent), days, goal, S0)
            state_tensor = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                action, bell, log_density, prob = model.sample_action(state_tensor, goal, days)
                action = action.item()
                v_n = action * (goal - total_stocks)
                log_densities.append(log_density)
                probabilities.append(torch.exp(prob).item())
        else:
            v_n = 0

        if t < days:
            total_spent += v_n * prices[t+1]
            total_stocks += v_n
            actions.append(v_n)
            states.append(state)
        else:
            actions.append(0)
            states.append(StockNetwork.normalize_state((t, prices[t], A_n, total_stocks, total_spent), days, goal, S0))

        if done:
            break
        

    return states, actions, log_densities, episode_payoff, prices, probabilities
""" 
def evaluate_policy(model, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal):
    np.random.seed(0)
    total_spent_list = []
    total_stocks_list = []
    A_n_list = []
    payoff_list = []
    final_day_list = []
    actions_list = []
   
    for _ in range(num_episodes):
        prices, A_n, q_n, total_spent, actions, log_densities, probabilities, bell_signals, episode_payoff = simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal)
        final_day = len(actions) - 1
        total_spent = sum([a * prices[t] for t, a in enumerate(actions)])# non plus total_spent = np.sum(np.array(actions) * np.array(prices[:len(actions)])) 
        total_stocks = np.sum(actions)
        total_spent_list.append(total_spent)
        total_stocks_list.append(total_stocks)
        A_n_list.append(A_n)
        payoff_list.append(episode_payoff)
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

"""
def export_csv_2(states, actions, log_densities, probabilities, episode_payoff, prices, q_n, A_n, filename):
    actions_len = len(actions)
    
    if len(states) != actions_len or len(log_densities) != actions_len or len(probabilities) != actions_len or len(q_n) != actions_len + 1:
        raise ValueError("Les longueurs des listes ne correspondent pas.")
    
    # Calcul des listes nécessaires pour le DataFrame
    payoff_list = np.full(actions_len, episode_payoff)
    total_stocks_list = q_n[:-1] * 100  # Exclure le dernier élément car q_n a une longueur de actions_len + 1
    total_spent_list = np.array([state[4] * 10000 for state in states])
    veriftotalspent_list = np.zeros(actions_len)
    
    # Calcul des valeurs vérifiées
    verifTotalStock = q_n[:-1]
    cumulative_sum = 0
    for t in range(actions_len):
        cumulative_sum += prices[t] * actions[t]
        veriftotalspent_list[t] = cumulative_sum

    # Création du DataFrame
    data = {
        "Day": np.arange(actions_len),
        "Prices(S_n)": prices[:actions_len],
        "A_n": A_n[:actions_len] * 100,
        "Total_Stocks": total_stocks_list,
        "Total_Spent": total_spent_list,
        "Actions": actions,
        "Log Density": log_densities,
        "Probability": probabilities,
        "Payoff": payoff_list,
        "VerifTotalStock": verifTotalStock * 100,
        "VerifTotalSpent": veriftotalspent_list
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
"""
"""
def export_csv(states, actions, log_densities, probabilities, episode_payoff, prices, filename):
    actions_len = len(actions)
    
    if len(states) != actions_len or len(log_densities) != actions_len or len(probabilities) != actions_len:
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
        "log density": log_densities,
        "Probabilité action": [prob for prob in probabilities],
        "payoff": payoff_list,
        "verifTotalStock": verifTotalStock,
        "VerifTotalSpent": veriftotalspent_list
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
"""
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
    ax2.set_ylabel('q_n en valeur réelle', color='red')
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
        model.train_model(num_episodes=num_episodes, simulate_episode=simulate_episode, S0=S0, V0=V0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho, days=days, goal=goal)
        
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
avg_total_spent, avg_total_stocks, avg_A_n, avg_episode_payoff_value, avg_final_day, actions_list = evaluate_policy(model, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal)

print(f"\nRésultats de l'évaluation de la politique sur {num_episodes} épisodes :")
print(f"Total dépensé en moyenne: {avg_total_spent}")
print(f"Total d'actions en moyenne: {avg_total_stocks}")
print(f"Prix moyen des actions: {avg_A_n}")
print(f"Payoff moyen: {avg_episode_payoff_value}")
print(f"Jour final moyen: {avg_final_day}")

# Simulation d'un épisode et exportation des résultats
prices, A_n, q_n, total_spent, actions, log_densities, probabilities, cloche_n, episode_payoff = simulate_episode(model,  S0, V0, mu, kappa, theta, sigma, rho, days, goal)
#export_csv(states, actions, log_densities, probabilities, episode_payoff, prices, "episode_sans_heston.csv")
S_n = prices

# Tracé des résultats
plot_episode(S_n, A_n, q_n, cloche_n)



    
