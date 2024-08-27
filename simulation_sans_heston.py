import numpy as np
import torch
from RL_agent import StockNetwork 
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt 
from nn import Net


def simulate_price(X, sigma):
    # S_n initialisé avec la même forme que X mais avec une ligne supplémentaire pour S0
    S_n = np.zeros((X.shape[0] + 1, X.shape[1]))
    S_n[0, :] = S0
    S_n[1:] = S0 + np.cumsum(sigma * X, axis=0)
    return S_n

def payoff(A_n, total_spent):
    return 100 * A_n - total_spent

def simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal, flag, batch_size=2):
    # Initialisation des arrays pour les batchs
    q_n = np.zeros((days + 1, batch_size))
    q_n[days, :] = goal + 1 # pour considérer que l'épisode termine avant t = days
    A_n = np.zeros((days + 1, batch_size))
    actions = np.zeros((days + 1, batch_size))
    bell_signals = np.zeros((days + 1, batch_size))
    total_spent = np.zeros((days + 1, batch_size))
    log_densities = np.zeros((days + 1, batch_size), dtype=object) # pour stocker des tenseurs
    probabilities = np.zeros((days + 1, batch_size))
    episode_payoff = np.zeros(batch_size)

    X = np.random.normal(0, 1, (days, batch_size))
    S_n = simulate_price(X, sigma)

    for t in range(days):
        A_n[t, :] = np.mean(S_n[1:t+1, :], axis=0) if t > 0 else S0
        state = StockNetwork.normalize((t, S_n[t, :], A_n[t, :], q_n[t, :], total_spent[t, :]), days, goal, S0) # todo verif
        state_tensor = torch.tensor(state, dtype=torch.float32)

        log_density = None
        prob = 0

        with torch.no_grad():
            if flag:
                total_stock_target, bell, log_density, prob = model.sample_action(state_tensor, goal, days)
                # Conversion to scalar
                total_stock_target = total_stock_target.item()
                prob = prob.item()
            else:
                np.random.seed(0)
                total_stock_target, bell = model.forward(state_tensor)
                total_stock_target = total_stock_target.item()
                bell = bell.item()

                q_n[t+1, :] = total_stock_target * (goal - q_n[t, :] ) if t < days - 1 else goal 
                v_n = q_n[t+1, :] - q_n[t, :]
                total_spent[t+1, :] = total_spent[t, :] + v_n * S_n[t+1, :]
                log_densities[t, :] = log_density
                probabilities[t, :] = np.exp(-prob)
                actions[t, :] = v_n
                bell_signals[t, :] = bell

                condition = (bell_signals[t, :] >= 0.5) & (t >= 19) & (q_n[t + 1, :] >= goal)
                if np.any(condition):
                    q_n[t+1:, condition] = q_n[t, condition]  # Maintenir q_n constant après la cloche
                    not_assigned = np.isnan(episode_payoff[condition])
                    if np.any(not_assigned): # Si le payoff n'est pas assigné alors on le calcule
                        episode_payoff[condition] = payoff(A_n[t, condition], total_spent[t, condition])
    
    condition = q_n[days, :] < goal
    if np.any(condition):
        A_n[days, condition] = np.mean(S_n[1:days + 1, :], axis=0)[condition]
        final_adjustment = goal - q_n[days, condition]
        total_spent[days, condition] += final_adjustment * S_n[days, condition]
        actions[-1, condition] += final_adjustment
        q_n[days, condition] = goal
    
    episode_payoff = payoff(A_n[days, :], total_spent[days, :])

    return S_n, A_n, q_n, total_spent, actions, log_densities, probabilities, bell_signals, episode_payoff

def train_model(model, simulate_episode, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal, batch_size=2):
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for episode in tqdm(range(num_episodes)):
        results = simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal, flag=True, batch_size=batch_size)
        S, A_n, q_n, total_spent, actions, log_densities, probabilities, bell_signals, episode_payoff = results

        optimizer.zero_grad()
        loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        loss = loss - log_densities.sum()

        # moyenne pour chaque b des days et multiplier par la moyenne des b
        episode_payoff_tensor = torch.tensor(episode_payoff, dtype=torch.float32, requires_grad=True)
        mean_days_per_b = torch.mean(episode_payoff_tensor, dim=0)
        mean_b = torch.mean(mean_days_per_b)
        loss *= mean_b


        if episode % 50 == 0:
            print(f"Episode {episode}: Average Episode Payoff {np.mean(episode_payoff)}, Loss {loss}")

        loss.backward()
        optimizer.step()

def evaluate_policy(model, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal, batch_size=2):
    total_spent_list = []
    total_stocks_list = []
    A_n_list = []
    payoff_list = []
    final_day_list = []
    actions_list = []
   
    for _ in range(num_episodes):
        # Simuler un épisode avec batch_size
        S_n, A_n, q_n, total_spent, actions, log_densities, probabilities, bell_signals, episode_payoff = simulate_episode(
            model, S0, V0, mu, kappa, theta, sigma, rho, days, goal, flag=False, batch_size=batch_size
        )
        
        # Prendre la première trajectoire (batch 0) pour l'évaluation
        final_day = len(actions[:, 0]) - 1
        total_spent_single = total_spent[final_day, 0]
        total_stocks = q_n[final_day, 0]
        total_spent_list.append(total_spent_single) 
        total_stocks_list.append(total_stocks)
        A_n_list.append(A_n[final_day, 0])
        payoff_list.append(episode_payoff[0])  # Prendre la moyenne du premier batch
        final_day_list.append(final_day)
        actions_list.append(actions[:, 0])

    avg_total_spent = np.mean(total_spent_list)
    avg_total_stocks = np.mean(total_stocks_list)
    avg_A_n = np.mean(A_n_list)
    avg_episode_payoff_value = np.mean(payoff_list)
    avg_final_day = np.mean(final_day_list)
    #  On retourne la moyenne des achats d'action pour chaque jour
    avg_actions = np.mean(actions_list, axis=0)

    return avg_total_spent, avg_total_stocks, avg_A_n, avg_episode_payoff_value, avg_final_day, avg_actions


def display_optimal_plan(actions, S_n):
    print("\nProgramme d'achat optimal:")
    for t, action in enumerate(actions):
        print(f"Jour {t + 1}: Achat de {action} actions")
    print(f"Prix des actions: {S_n[:len(actions)]}")
    print(f"Actions cumulées: {np.cumsum(actions)}")

def export_csv(actions, episode_payoff, S_n, A_n, q_n, total_spent, filename):
    actions_len = len(actions)
    
    if len(A_n) != actions_len or len(log_densities) != actions_len or len(probabilities) != actions_len or len(q_n) != actions_len or len(total_spent) != actions_len:
        raise ValueError("Les longueurs des listes ne correspondent pas.")
    
    payoff_list = [0] * (actions_len - 1) + [episode_payoff]  # Payoff nul jusqu'au dernier jour
    data = {
        "Day": list(range(actions_len)),
        "Prices(S_n)": S_n,  
        "A_n": A_n,
        "Total_Stocks": q_n,
        "Total_Spent": total_spent,
        "actions": actions, 
        "payoff": payoff_list,
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
    ax2.set_ylabel('q_n en valeur réelle', color='red')
    ax2.plot(q_n, label="q_n (Quantité totale d'actions au jour n)", color="red", linestyle='-')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.tight_layout()
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    plt.title("Évolution de S_n, A_n, q_n et cloche_n au fil du temps")
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
S0 = 100
V0 = 0.04  # Volatilité initiale
mu = 0.1   # Rendement attendu
kappa = 2.0
theta = 0.04
sigma = 2.0
rho = -0.7
days = 60
goal = 100

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
            train_model(model, simulate_episode=simulate_episode, num_episodes=num_episodes, S0=S0, V0=V0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho, days=days, goal=goal)
            
            save_path = input("Entrez le chemin pour sauvegarder le modèle : ").strip()
            model.save_model(save_path)
            print(f"Modèle sauvegardé à {save_path}")

    except Exception as e:
        print(f"Erreur lors du chargement ou de l'entraînement du modèle : {e}")

elif choice == 'e':
    # Entraînement d'un nouveau modèle
    num_episodes = int(input("Entrez le nombre d'épisodes pour l'entraînement : "))
    train_model(model, simulate_episode=simulate_episode, num_episodes=num_episodes, S0=S0, V0=V0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho, days=days, goal=goal)
    
    save_path = input("Entrez le chemin pour sauvegarder le modèle : ").strip()
    try:
        model.save_model(save_path)
        print(f"Modèle sauvegardé à {save_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")



# Évaluation de la politique
num_episodes = 50
avg_total_spent, avg_total_stocks, avg_A_n, avg_episode_payoff_value, avg_final_day, actions_list = evaluate_policy(
    model, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal, batch_size=2
)

print(f"\nRésultats de l'évaluation de la politique sur {num_episodes} épisodes :")
print(f"Total dépensé en moyenne: {avg_total_spent}")
print(f"Total d'actions en moyenne: {avg_total_stocks}")
print(f"Prix moyen des actions: {avg_A_n}")
print(f"Payoff moyen: {avg_episode_payoff_value}")
print(f"Jour final moyen: {avg_final_day}")
print(f"Actions moyennes par jour: {actions_list}")

# Simulation d'un épisode et exportation des résultats
S_n, A_n, q_n, total_spent, actions, log_densities, probabilities, cloche_n, episode_payoff = simulate_episode(
    model, S0, V0, mu, kappa, theta, sigma, rho, days, goal, flag=False, batch_size=2
)

# Exportation des résultats du premier batch
export_csv(actions[:, 0], episode_payoff[0], S_n[:,0], A_n[:, 0], q_n[:, 0], total_spent[:, 0], "episode_sans_heston.csv")

# Tracé des résultats pour le premier batch
plot_episode(S_n[:, 0], A_n[:, 0], q_n[:, 0], cloche_n[:, 0])