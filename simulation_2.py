import numpy as np
import torch
from RL_agent import StockNetwork
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from nn import Net


def simulate_price(S_0, X, sigma):
    S_n = torch.zeros((X.shape[0] + 1, X.shape[1]), dtype=X.dtype)
    S_n[0, :] = S_0
    S_n[1:] = S_0 + torch.cumsum(sigma * X, dim=0)
    return S_n

def calculate_condition1(bell_signals, q_n, t, jour_cloche, goal, days, batch_index):
    return ((bell_signals[t, batch_index] >= 0.5) & (t >= jour_cloche-1) & (q_n[t, batch_index] >= goal)) | (t+1 >= days) 

def calculate_condition(bell_signals, q_n, t, jour_cloche, goal, days):
    return ((bell_signals[t, :] >= 0.5) & (t >= jour_cloche-1) & (q_n[t, :] >= goal)) 

def payoff(q_n, A_n, total_spent,n):
    return q_n[n,:] * A_n[n,:] - total_spent[n,:]

def expected_payoff(A_n, total_spent, bell_n, q_n, N):
    """
    Calcule l'espérance des payoffs en utilisant la formule :
    E[ somme_{n=1}^{N} prod_{k=1}^{n-1} (1 - p_k) * p_n * PnL_n ]

    Arguments :
    - A_n : moyenne des prix des actions (Tensor)
    - total_spent : montant total dépensé jusqu'à l'étape n (Tensor)
    - bell_n : signaux d'arrêt (Tensor de probabilités entre 0 et 1)
    - goal : objectif d'actions à atteindre
    - N : nombre total d'étapes

    Retourne :
    - payoff_values : le payoff attendu pour chaque simulation dans le batch
    """
    payoff_values = torch.zeros(bell_n.shape[1], dtype=torch.float32, device=bell_n.device)

    for n in range(1, N+1):
        # Produit des termes (1 - p_k) pour k = 1 à n-1
        product = torch.ones(bell_n.shape[1], dtype=torch.float32, device=bell_n.device)
        for k in range(1, n):
            product *= (1 - bell_n[k, :])  # Produit de (1 - p_k)

        # Contribution du terme courant (p_n * PnL_n)
        p_n = bell_n[n, :]
        pnl_n = payoff(q_n, A_n, total_spent, n)  # Calcul du payoff à l'étape n

        # Mise à jour des valeurs de payoff
        payoff_values += product * p_n * pnl_n

    return payoff_values

def simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal, train, batch_size=2):
    A_n = torch.zeros((days+1, batch_size), dtype=torch.float32)
    q_n = torch.zeros((days+1, batch_size), dtype=torch.float32)
    v_n = torch.zeros((days+1, batch_size), dtype=torch.float32)
    X_n = torch.zeros((days+1, batch_size), dtype=torch.float32)
    bell_n = torch.zeros((days+1, batch_size), dtype=torch.float32)
    somme_density = 0.0

    if not train:  # lorsqu'on évalue le model
        np.random.seed(0)
        #torch.seed()
    
    X_ = torch.normal(0, 1, (days, batch_size), dtype=torch.float32)
    S_n = simulate_price(S0, X_, sigma)

    for t in range(days+1):
        A_n[t, :] = torch.mean(S_n[1:t+1, :], axis=0) if t > 0 else S0
        
    for t in range(days):
        if isinstance(model, Net):
            state = model.normalize(t+1, S_n[t, :], A_n[t, :], q_n[t, :])  
        else: 
            
            state = model.normalize1((t, S_n[t, :], A_n[t, :], q_n[t, :], X_n[t, :]), days, goal, S0)
        if train:
            total_stock_target, bell, log_density = model.sample_action(state, goal, days)
            #print(total_stock_target, bell, log_density)
            somme_density += log_density
        else:
            if isinstance(model, Net):
                etat = model.forward(state)
                total_stock_target = etat[0]
                bell = etat[1]
            else:
                etat = model.forward(state)
                total_stock_target = etat[0]
                bell = etat[1]

        bell_n[t, :] = bell
        condition = calculate_condition(bell_n, q_n, t, 22, goal, days)
        not_condition=~condition
        q_n[t+1, condition], q_n[t+1, not_condition] = goal, total_stock_target[not_condition]
        q_n[t+1,(q_n[t]==goal)]=goal

        v_n[t+1,:] = q_n[t+1, :] - q_n[t, :]
        X_n[t+1,:] = X_n[t, :] + v_n[t+1,:] * S_n[t+1,:]
        
    condition = q_n[days, :] < goal
    if condition.any():
        final_adjustment = goal - q_n[days, condition]
        X_n[days, condition] = X_n[days-1, condition] + final_adjustment * S_n[days, condition]
        v_n[-1, condition] = final_adjustment
        q_n[days, condition] = goal
    bell_n[days,:]=1
    episode_payoff = expected_payoff(A_n, X_n, bell_n, q_n, days)
    return S_n, A_n, q_n, X_n, v_n, somme_density, bell_n, episode_payoff

def train_model(model, simulate_episode, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, days, goal, batch_size=2, lr=0.05, save_path="model.pt"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for episode in range(num_episodes):
        # Simuler un épisode
        S_n, A_n, q_n, total_spent, actions, somme, bell_signals, episode_payoff = simulate_episode(
            model, S0, V0, mu, kappa, theta, sigma, rho, days, goal, train=True, batch_size=batch_size
        )
        
        # Calcul de la perte et rétropropagation
        optimizer.zero_grad()
        loss = -(somme * episode_payoff).mean()
        loss.backward()
        optimizer.step()
        
        # Afficher les statistiques pour chaque épisode
        #print(f"Episode {episode + 1}/{num_episodes}: Loss = {loss.item():.4f}, Total Spent = {total_spent:.2f}, A_n = {A_n:.2f}, Payoff = {episode_payoff:.2f}")
    
    # Sauvegarder le modèle
    torch.save(model.state_dict(), save_path)
    print(f"Modèle sauvegardé à {save_path}")


                
def evaluate_single_episode(model, S0, V0, mu, kappa, theta, sigma, rho, N, Q, batch_size=2):
    results = simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal, train=False, batch_size=2)
    S_n, A_n, q_n, total_spent, actions, log_densities, bell_signals, episode_payoff = results
    final_day = None
    for t in range(N):
        condition = calculate_condition(bell_signals, q_n, t, jour_cloche, Q, N)
        if (condition & (bell_signals >= 1)).any():
            final_day = t
            break
    
    if final_day is None:
        final_day = N #si aucune condition remplie on prend le dernier jour

    total_spent_0 = total_spent[final_day, 0]
    total_stocks_0 = q_n[final_day, 0]
    A_n_0 = A_n[final_day, 0]
    payoff_0 = episode_payoff[0]
    actions_0 = actions[:, 0]

    return total_spent_0, total_stocks_0, A_n_0, payoff_0, final_day, actions_0


def display_optimal_plan(actions, S_n):
    print("\nProgramme d'achat optimal:")
    for t, action in enumerate(actions):
        print(f"Jour {t + 1}: Achat de {action} actions")
    print(f"Prix des actions: {S_n[:len(actions)]}")
    print(f"Actions cumulées: {np.cumsum(actions)}")

def export_csv(actions, episode_payoff, S_n, A_n, q_n, total_spent, filename):
    episode_payoff_detached = episode_payoff.item() if episode_payoff.dim() == 0 else episode_payoff[0].detach().numpy()
    
    # Convertir les scalaires en listes ou tableaux pour Pandas
    actions_detached = actions[0].detach().numpy() if actions.dim() > 0 else [actions.item()]
    S_n_detached = S_n[0].detach().numpy() if S_n.dim() > 0 else [S_n.item()]
    A_n_detached = A_n[0].detach().numpy() if A_n.dim() > 0 else [A_n.item()]
    q_n_detached = q_n[0].detach().numpy() if q_n.dim() > 0 else [q_n.item()]
    total_spent_detached = total_spent[0].detach().numpy() if total_spent.dim() > 0 else [total_spent.item()]

    # Si vous avez plusieurs lignes, vous devrez peut-être inclure un index
    data = {
        'actions': actions_detached,
        'episode_payoff': episode_payoff_detached,
        'S_n': S_n_detached,
        'A_n': A_n_detached,
        'q_n': q_n_detached,
        'total_spent': total_spent_detached
    }
    # Créer un index pour le DataFrame si vous utilisez des scalaires
    index = [0]  # Si vous avez seulement une ligne de données
    df = pd.DataFrame(data, index=index)
    df.to_csv(filename, index=False)

 
def plot_episode(S_n, A_n, q_n, bell_signals, days, jour_cloche, goal):
    S_n_detached = S_n.detach().numpy() if isinstance(S_n, torch.Tensor) else S_n
    A_n_detached = A_n.detach().numpy() if isinstance(A_n, torch.Tensor) else A_n
    q_n_detached = q_n.detach().numpy() if isinstance(q_n, torch.Tensor) else q_n

    # Créez la figure avec un axe principal
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.set_xlabel('Jour')
    ax1.set_ylabel("S_n et A_n en euro (€)", color='black')
    
    # Tracez S_n et A_n sur l'axe principal
    ax1.plot(S_n_detached, label="S_n (Prix de l'action au jour n)", color="blue")
    ax1.plot(A_n_detached, label="A_n (Prix moyen des actions aux jours n)", color="green")
    ax1.tick_params(axis='y', labelcolor='black')

    # Recherche du jour où la cloche sonne
    final_day_1 = None
    final_day_2 = None
    for t in range(days):
        condition0 = calculate_condition1(bell_signals, q_n, t, jour_cloche, goal, days, 0)
        if (condition0 & (bell_signals >= 1)).any():
            final_day_1 = t
            break
    for t in range(days):
        condition1 = calculate_condition1(bell_signals, q_n, t, jour_cloche, goal, days, 1)
        if (condition1 & (bell_signals >= 1)).any():
            final_day_2 = t
            break


        # Ajouter des lignes verticales pour les jours de cloche
    if final_day_1 is not None:
        ax1.axvline(x=final_day_1, color="red", linestyle='--', label="cloche_n = 1 (batch 1)" if final_day_1 == 0 else "")
    if final_day_2 is not None:
        ax1.axvline(x=final_day_2, color="purple", linestyle='--', label="cloche_n = 1 (batch 2)" if final_day_2 == 0 else "")

    # Créez un axe secondaire pour q_n
    ax2 = ax1.twinx()  
    ax2.set_ylabel('q_n en valeur réelle', color='red')
    
    # Tracez q_n sur l'axe secondaire
    ax2.plot(q_n_detached[:, 0], label="q_n (Quantité totale d'actions au jour n, batch 1)", color="red", linestyle='-')
    ax2.plot(q_n_detached[:, 1], label="q_n (Quantité totale d'actions au jour n, batch 2)", color="purple", linestyle='-')
    ax2.tick_params(axis='y', labelcolor='red')

    # Mise en forme du graphique
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    
    # Finalisation de la présentation du graphique
    fig.tight_layout()
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
 # Paramètres du modèle
S0 = 45
sigma = 0.6
days = 63
goal = 20
jour_cloche = 22
V0 = 0.04  # Volatilité initiale
mu = 0.1   # Rendement attendu
kappa = 2.0
theta = 0.04
rho = -0.7

model_name = input("Quel modèle voulez-vous utiliser ? (par exemple : Net(0), StockNetwork(1), etc.) : ").strip()
 
try:
    # Importation dynamique de la classe de modèle
    if model_name == "1":
        model = StockNetwork(goal)
    if model_name == "0":
        model = Net()

except KeyError:
    raise ValueError(f"Le modèle '{model_name}' n'est pas reconnu. Assurez-vous que le nom du modèle est correct.")

 
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
 
stats_batch = evaluate_single_episode(model, S0, V0, mu, kappa, theta, sigma, rho, days, goal, batch_size=2)

print("Batch 0 Stats:")
print(f"Total Spent: {stats_batch[0]}")
print(f"Total Stocks: {stats_batch[1]}")
print(f"A_n: {stats_batch[2]}")
print(f"Payoff: {stats_batch[3]}")
print(f"Final Day: {stats_batch[4]}")
print(f"Actions: {stats_batch[5]}")

# Simulation d'un épisode et exportation des résultats
# Run a simulation
S_n, A_n, q_n, X_n, v_n, somme_density, bell_n, episode_payoff = simulate_episode(
    model, S0, V0, mu, kappa, theta, sigma, rho, days, goal, train=False, batch_size=2
)


# Exportation des résultats du premier batch
#export_csv(actions[:, 0], episode_payoff[0], S_n[:,0], A_n[:, 0], q_n[:, 0], total_spent[:, 0], "episode_sans_heston.csv")
# tracé de l'épisode
plot_episode(S_n[:, 0], A_n[:, 0], q_n, bell_n, days, jour_cloche, goal)