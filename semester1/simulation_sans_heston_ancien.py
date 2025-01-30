import numpy as np
import torch
from RL_agent_ancien import StockNetwork
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from nn import Net

def simulate_price(X, sigma, S0):
    # Simulation des prix avec PyTorch
    S_n = torch.zeros((X.shape[0] + 1, X.shape[1]), dtype=X.dtype, device=X.device)
    S_n[0, :] = S0
    S_n[1:] = S0 + torch.cumsum(sigma * X, dim=0)
    return S_n

def payoff(A_n, total_spent):
    return Q * A_n - total_spent

def calculate_condition1(bell_signals, q_n, t, jour_cloche, Q, N, batch_index):
    return ((bell_signals[t, batch_index] >= 0.5) & (t >= jour_cloche-1) & (q_n[t, batch_index] >= Q)) | (t+1 >= Q) 

def calculate_condition(bell_signals, q_n, t, jour_cloche, Q, N):
    return ((bell_signals[t, :] >= 0.5) & (t >= jour_cloche-1) & (q_n[t, :] >= Q)) | (t+1 >= N) 

def expected_payoff(A_n, total_spent, bell_signals, Q, t):
    
    uniform_random = torch.rand_like(bell_signals)
    # dim de p chapeau t+1,2 pour nos 2 batchs
    p_hat = (uniform_random < bell_signals).float()
    payoff_values = torch.zeros_like(bell_signals)
    
    payoff_values[1, :] = payoff(A_n[1, :], total_spent[1, :])
    
    for step in range(2, t + 1):
        
        payoff_current = payoff(A_n[step, :], total_spent[step, :])
        payoff_values[step, :] = p_hat[step, :] * payoff_current + (1 - p_hat[step, :]) * payoff_values[step - 1, :]

    return payoff_values[t,:]
"""
def simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, N, Q, train, batch_size=2):
    q_n = torch.zeros((N+1, batch_size), dtype=torch.float32)
    q_n[N, :] = Q 
    A_n = torch.zeros((N+1, batch_size), dtype=torch.float32)
    actions = torch.zeros((N+1, batch_size), dtype=torch.float32)
    bell_signals = torch.zeros((N+1, batch_size), dtype=torch.float32)
    total_spent = torch.zeros((N+1, batch_size), dtype=torch.float32)
    log_densities = torch.zeros((N+1, batch_size), dtype=torch.float32, requires_grad=True)
    #episode_payoff = torch.zeros(batch_size, dtype=torch.float32)
    episode_payoff = torch.full((batch_size,), float('nan'), dtype=torch.float32, requires_grad=True)

    if not train: # lorsqu'on évalue le model
        np.random.seed(0)
    X = torch.normal(0, 1, (N, batch_size), dtype=torch.float32)
    S_n = simulate_price(X, sigma, S0)
    for t in range(N+1):
        A_n[t, :] = torch.mean(S_n[1:t+1, :], axis=0) if t > 0 else S0

    for t in range(N):
        if isinstance(model, Net):
            state = model.normalize1(t+1 , S_n[t, :], A_n[t, :], q_n[t, :])  
        else: 
            state = model.normalize((t, S_n[t, :], A_n[t, :], q_n[t, :], total_spent[t, :]), N, Q, S0)

        etat = model.forward(state)
        mean = etat[:,0] 
        bell = etat[:,1]
        if train:
            std = torch.tensor((Q / N) * 0.05, dtype=torch.float32, device=total_stock_target.device)
            total_stock_target = mean + std * torch.randn_like(mean) # mu + sigma * N(0,1)
            u = np.random.uniform(0, 1, size=mean.shape)
            u = torch.tensor(u, dtype=torch.float32, device=mean.device)
            #bell_p = (u < bell).float()
            pdf_total_stock_target =  torch.exp(-0.5 * ((total_stock_target - mean) / std) ** 2) / ((np.sqrt(2*np.pi) * std))  
            density=pdf_total_stock_target 
            log_densities[t,:] =torch.log(density)
            
            # total_stock_target, x[:,1], log_density

        # MAJ des états
        q_n[t+1,:] = total_stock_target if t < N-1 else Q # (Q - q_n[t, :]) if t < N - 1 else Q
        v_n = q_n[t+1, :] - q_n[t, :]
        total_spent[t+1, :] = total_spent[t, :] + v_n * S_n[t+1, :]
        actions[t, :] = v_n
        bell_signals[t, :] = bell
        condition = calculate_condition(bell_signals, q_n, t, jour_cloche, Q, N)
        if condition.any(): # Si la condition est remplie pour au moins un batch
            bell_signals[t+1, :]=bell_signals[t+1, :]+1
            q_n[t+1:, condition] = q_n[t, condition]
            not_assigned = torch.isnan(episode_payoff)
            liste_bell = torch.zeros(N+1, batch_size, dtype=torch.float32)
            liste_bell[t] = 1           
            episode_payoff = episode_payoff.clone().requires_grad_(True)
            #episode_payoff[not_assigned] = expected_payoff(A_n, total_spent, liste_bell, Q, t)[not_assigned]
            episode_payoff[not_assigned] = payoff(A_n[t+1, not_assigned], total_spent[t+1, not_assigned])
            A_n[N, condition] = A_n[t+1, condition]
        
    condition = q_n[N, :] < Q
    if condition.any():
        final_adjustment = Q - q_n[N, condition]
        total_spent[N, condition] = total_spent[N-1,condition] + final_adjustment * S_n[N, condition]
        actions[-1, condition] = final_adjustment
        q_n[N, condition] = Q
        liste_bell=torch.zeros(N+1, batch_size, dtype=torch.float32)
        liste_bell[N]=1
        episode_payoff = episode_payoff.clone().requires_grad_(True)
        #episode_payoff = expected_payoff(A_n,total_spent,liste_bell,Q,N)
        episode_payoff = payoff(A_n[N, :], total_spent[N, :])

    return S_n, A_n, q_n, total_spent, actions, log_densities, bell_signals, episode_payoff
"""

def simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, N, Q, train, batch_size=2):
    # Initialisation des tenseurs
    q_n = torch.zeros((N + 1, batch_size), dtype=torch.float32)
    #q_n = np.zeros((N + 1, batch_size))
    q_n[0, :], q_n[-1, :]  = 0 , Q
    A_n = torch.zeros((N + 1, batch_size), dtype=torch.float32)
    actions = torch.zeros((N +1, batch_size), dtype=torch.float32)
    bell_signals = torch.zeros((N + 1, batch_size), dtype=torch.float32)
    total_spent = torch.zeros((N + 1, batch_size), dtype=torch.float32)
    #log_densities = torch.zeros((N, batch_size), dtype=torch.float32, requires_grad=True)
    #episode_payoff = torch.full((batch_size,), float('nan'), dtype=torch.float32, requires_grad=True)
    somme = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
    if not train:
        np.random.seed(0)
    
    # Simulation des prix
    X = torch.normal(0, 1, (N, batch_size), dtype=torch.float32)
    S_n = simulate_price(X, sigma, S0)
    
    for t in range(1, N + 1):
        A_n[t, :] = torch.mean(S_n[:t, :], dim=0)

    # Simulation des actions
    for t in range(N):
        if isinstance(model, Net):
            state = model.normalize(t, S_n[t, :], A_n[t, :], q_n[t, :])
        else:
            state = model.normalize((t, S_n[t, :], A_n[t, :], q_n[t, :], total_spent[t, :]), N, Q, S0)
        x = model.forward(state)
        mean, stopping_prob = x[:, 0], x[:, 1]
        
        if train:
            std = (Q / N) * 0.05
            action_sample = mean + std * torch.randn_like(mean)
            log_density = torch.exp(-0.5 * (((action_sample - mean) / std) * (action_sample - mean) / std)) / ((np.sqrt(2*np.pi) * std))
            somme += log_density
            #log_densities = log_densities.clone().requires_grad_(True)
            #log_densities[t, :] = log_density
            #u = torch.rand_like(stopping_prob)
            #bell_decision = (u < stopping_prob).float()
        else:
            action_sample = mean
            bell_decision = (stopping_prob > 0.5).float()

        v_n = action_sample.clamp(min=torch.tensor(0), max=Q - q_n[t, :])
        actions[t, :] = v_n
        q_n[t + 1, :] = q_n[t, :] + v_n
        total_spent[t + 1, :] = total_spent[t, :] + v_n * S_n[t, :]
        bell_signals[t + 1, :] = bell_signals[t, :] + bell_decision
        
        # Calcul des conditions de fin
        condition = (q_n[t + 1, :] >= Q) | (bell_signals[t + 1, :] > 0)
        if condition.any():
            episode_payoff = episode_payoff.clone().requires_grad_(True)
            episode_payoff[condition] = payoff(A_n[t + 1, condition], total_spent[t + 1, condition])
            break

    # Mise à jour finale si nécessaire
    if torch.isnan(episode_payoff).any():
        final_condition = torch.isnan(episode_payoff)
        episode_payoff = episode_payoff.clone().requires_grad_(True)
        episode_payoff[final_condition] = payoff(A_n[-1, final_condition], total_spent[-1, final_condition])

    return S_n, A_n, q_n, total_spent, actions, somme, bell_signals, episode_payoff

def train_model(model, simulate_episode, num_episodes, S0, V0, mu, kappa, theta, sigma, rho, N, Q, batch_size=2):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for episode in range(num_episodes):
        # Simulation d'un épisode
        results = simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, N, Q, train=False, batch_size=batch_size)
        _, _, _, _, _, somme, _, episode_payoff = results
        
        # Calcul de la perte
        optimizer.zero_grad()
        #log_density_sum = log_densities.sum(dim=0)
        loss = -(somme * episode_payoff).mean()
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        if episode % 10 == 0:
            print(f"Episode {episode}: Loss = {loss.item():.4f}, Average Episode Payoff {torch.mean(episode_payoff)}")

def evaluate_single_episode(model, S0, V0, mu, kappa, theta, sigma, rho, N, Q, batch_size=2):
    results = simulate_episode(model, S0, V0, mu, kappa, theta, sigma, rho, N, Q, train=False, batch_size=batch_size)
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
    episode_payoff_detached = episode_payoff.detach().numpy()
    
    # Convertir les scalaires en listes ou tableaux pour Pandas
    actions_detached = actions.detach().numpy()
    S_n_detached = S_n.detach().numpy() 
    A_n_detached = A_n.detach().numpy() 
    q_n_detached = q_n.detach().numpy() 
    total_spent_detached = total_spent.detach().numpy() 

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
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

 
def plot_episode(S_n, A_n, q_n, bell_signals, N, jour_cloche, Q):
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
    for t in range(N):
        condition0 = calculate_condition1(bell_signals, q_n, t, jour_cloche, Q, N, 0)
        if (condition0 & (bell_signals >= 1)).any():
            final_day_1 = t
            break
    for t in range(N):
        condition1 = calculate_condition1(bell_signals, q_n, t, jour_cloche, Q, N, 1)
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
N = 63
Q = 20
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
        model = StockNetwork(Q)
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
            train_model(model, simulate_episode=simulate_episode, num_episodes=num_episodes, S0=S0, V0=V0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho, N=N, Q=Q)
           
            save_path = input("Entrez le chemin pour sauvegarder le modèle : ").strip()
            model.save_model(save_path)
            print(f"Modèle sauvegardé à {save_path}")
 
    except Exception as e:
        print(f"Erreur lors du chargement ou de l'entraînement du modèle : {e}")
 
elif choice == 'e':
    # Entraînement d'un nouveau modèle
    num_episodes = int(input("Entrez le nombre d'épisodes pour l'entraînement : "))
    train_model(model, simulate_episode=simulate_episode, num_episodes=num_episodes, S0=S0, V0=V0, mu=mu, kappa=kappa, theta=theta, sigma=sigma, rho=rho, N=N, Q=Q)
   
    save_path = input("Entrez le chemin pour sauvegarder le modèle : ").strip()
    try:
        model.save_model(save_path)
        print(f"Modèle sauvegardé à {save_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")
 


stats_batch = evaluate_single_episode(model, S0, V0, mu, kappa, theta, sigma, rho, N, Q, batch_size=2)

print("Batch 0 Stats:")
print(f"Total Spent: {stats_batch[0]}")
print(f"Total Stocks: {stats_batch[1]}")
print(f"A_n: {stats_batch[2]}")
print(f"Payoff: {stats_batch[3]}")
print(f"Final Day: {stats_batch[4]}")
print(f"Actions: {stats_batch[5]}")

# Simulation d'un épisode et exportation des résultats
S_n, A_n, q_n, total_spent, actions, log_densities, bell_signals, episode_payoff = simulate_episode(
    model, S0, V0, mu, kappa, theta, sigma, rho, N, Q, train=False, batch_size=2
)

# Exportation des résultats du premier batch
export_csv(actions[:, 0], episode_payoff[0], S_n[:,0], A_n[:, 0], q_n[:, 0], total_spent[:, 0], "episode_sans_heston.csv")
# tracé de l'épisode
plot_episode(S_n[:, 0], A_n[:, 0], q_n, bell_signals, N, jour_cloche, Q)