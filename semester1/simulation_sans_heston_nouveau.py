import numpy as np
import torch
from RL_agent import StockNetwork
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from nn import Net

# Paramètres du modèle
S0 = 45
sigma = 0.6
days = 63
Q = 20
batch = 2

def simulate_price(X, sigma, S0):
    S_n = np.zeros((X.shape[0] + 1, X.shape[1]))
    S_n[0, :] = S0
    S_n[1:] = S0 + np.cumsum(sigma * X, axis=0)
    return S_n

def payoff(A_n, total_spent):
    return Q * A_n - total_spent

def calculate_condition(bell_signals, q_n, t, jour_cloche, Q, days):
    return ((bell_signals[t, :] >= 0.5) & (t >= jour_cloche-1) & (q_n[t, :] >= Q)) 

def train_model(model, days, batch, sigma, S0, Q):
    q_n = torch.zeros((days+1, batch), dtype=torch.float32)
    q_n[days, :] = Q 
    A_n = torch.zeros((days+1, batch), dtype=torch.float32)
    actions = torch.zeros((days+1, batch), dtype=torch.float32)
    bell_signals = torch.zeros((days+1, batch), dtype=torch.float32)
    total_spent = torch.zeros((days+1, batch), dtype=torch.float32)
    log_densities = torch.zeros((days+1, batch), dtype=torch.float32)
    episode_payoff = torch.full((batch,), float('nan'), dtype=torch.float32)

    X = np.random.normal(0, 1, (days, batch))
    S_n = simulate_price(X, sigma, S0)
    S_n = torch.tensor(S_n, dtype=torch.float32)
    
    for t in range(days+1):
        A_n[t, :] = torch.mean(S_n[1:t+1, :], axis=0) if t > 0 else S0
        
    for t in range(days):
        if isinstance(model, Net):
            state = model.normalize(t+1 , S_n[t, :], A_n[t, :], q_n[t, :])  
        else: 
            state = model.normalize((t, S_n[t, :], A_n[t, :], q_n[t, :], total_spent[t, :]), days, goal, S0)
        #state_tensor = torch.tensor(state, dtype=torch.float32)

        log_density = None
        #prob =0

    # Ajoutez ici le code pour l'entraînement du modèle

def evaluate_model(model, days, batch, sigma, S0, Q):
    q_n = torch.zeros((days+1, batch), dtype=torch.float32)
    q_n[days, :] = Q 
    A_n = torch.zeros((days+1, batch), dtype=torch.float32)
    actions = torch.zeros((days+1, batch), dtype=torch.float32)
    bell_signals = torch.zeros((days+1, batch), dtype=torch.float32)
    total_spent = torch.zeros((days+1, batch), dtype=torch.float32)
    log_densities = torch.zeros((days+1, batch), dtype=torch.float32)
    episode_payoff = torch.full((batch,), float('nan'), dtype=torch.float32)

    np.random.seed(0)
    X = np.random.normal(0, 1, (days, batch))
    S_n = simulate_price(X, sigma, S0)
    S_n = torch.tensor(S_n, dtype=torch.float32)
    
    for t in range(days+1):
        A_n[t, :] = torch.mean(S_n[1:t+1, :], axis=0) if t > 0 else S0
        
    for t in range(days):
        if isinstance(model, Net):
            state = model.normalize(t+1 , S_n[t, :], A_n[t, :], q_n[t, :])  
        else: 
            state = model.normalize((t, S_n[t, :], A_n[t, :], q_n[t, :], total_spent[t, :]), days, Q, S0)
        #state_tensor = torch.tensor(state, dtype=torch.float32)

        log_density = None
        #prob =0 


if training:
    train_model(model, days, batch, sigma, S0, Q)
else:
    evaluate_model(model, days, batch, sigma, S0, Q)

def simulate_episode(model, S0, sigma, days, Q, training, batch_size=batch):
    q_n = torch.zeros((days+1, batch), dtype=torch.float32)
    q_n[days, :] = Q 
    A_n = torch.zeros((days+1, batch), dtype=torch.float32)
    actions = torch.zeros((days+1, batch), dtype=torch.float32)
    bell_signals = torch.zeros((days+1, batch), dtype=torch.float32)
    total_spent = torch.zeros((days+1, batch), dtype=torch.float32)
    log_densities = torch.zeros((days+1, batch), dtype=torch.float32)
    episode_payoff =torch.full((batch,),float('nan'), dtype=torch.float32)

    if not training: # lorsqu'on évalue le model
        np.random.seed(0)
    X = np.random.normal(0, 1, (days, batch))
    S_n = simulate_price(X, sigma, S0)
    S_n = torch.tensor(S_n, dtype=torch.float32)
    
    for t in range(days+1):
        A_n[t, :] = torch.mean(S_n[1:t+1, :], axis=0) if t > 0 else S0
        
    for t in range(days):
        if isinstance(model, Net):
            state = model.normalize(t+1 , S_n[t, :], A_n[t, :], q_n[t, :])  
        else: 
            state = model.normalize((t, S_n[t, :], A_n[t, :], q_n[t, :], total_spent[t, :]), days, goal, S0)
        #state_tensor = torch.tensor(state, dtype=torch.float32)

        log_density = None
        #prob =0

        with torch.no_grad():
            if flag:
                total_stock_target, bell, log_density = model.sample_action(state, goal, days)
            else:
                if isinstance(model, Net):
                    etat = model.forward(state)
                    total_stock_target = etat[0]
                    bell = etat[1]
                else:
                    etat = model.forward(state)
                    total_stock_target = etat[0]
                    bell = etat[1]
                    #total_stock_target, bell = model.forward(state)
        """           
        bell_signals[t, :] = bell
        condition = calculate_condition(bell_signals, q_n, t, 22, goal, days)
        not_condition=~condition
        q_n[t+1, condition], q_n[t+1, not_condition] = goal, total_stock_target[not_condition]
        q_n[t+1,(q_n[t]==goal)]=goal
        v_n = q_n[t+1, :] - q_n[t, :]
        total_spent[t+1, :] = total_spent[t, :] + v_n * S_n[t+1, :]
        log_densities[t, :] = log_density if log_density is not None else 0
        """
        # MAJ des états
        q_n[t+1,:] = total_stock_target if t < days-1 else goal # * (goal - q_n[t, :]) if t < days - 1 else goal
        v_n = q_n[t+1, :] - q_n[t, :]
        total_spent[t+1, :] = total_spent[t, :] + v_n * S_n[t+1, :]
        log_densities[t, :] = log_density if log_density is not None else 0
        #print(prob)
        #probabilities[t, :] = prob #np.exp(-prob)
        actions[t, :] = v_n
        bell_signals[t, :] = bell
        #print(bell_signals[t, :])
        condition = calculate_condition(bell_signals, q_n, t, 22, goal, days) # Condition pour vérifier si le signal de cloche est activé, si t >= 19, et si q_n[t+1, :] est supérieur ou égal à goal
        if condition.any(): # Si la condition est remplie pour au moins un batch
            bell_signals[t, :]=bell_signals[t, :]+1
            q_n[t+1:, condition] = q_n[t, condition]
            not_assigned = torch.isnan(episode_payoff)
            episode_payoff[not_assigned] = payoff(A_n[t, not_assigned], total_spent[t, not_assigned])
            #A_n[days, condition] = torch.mean(S_n[1:days + 1, :], axis=0)[condition]
        
    condition = q_n[days, :] < goal
    if condition.any():
        #A_n[days, condition] = torch.mean(S_n[1:days + 1, :], axis=0)[condition]
        final_adjustment = goal - q_n[days, condition]
        total_spent[days, condition] += final_adjustment * S_n[days, condition]
        actions[-1, condition] += final_adjustment
        q_n[days, condition] = goal
        episode_payoff = payoff(A_n[days, :], total_spent[days, :])
    bell_signals[days,:]=1
    #episode_payoff = expected_payoff(A_n, total_spent, bell_signals, q_n, days)
    return S_n, A_n, q_n, total_spent, actions, log_densities, bell_signals, episode_payoff


def train_model(model, simulate_episode, num_episodes, S0, sigma, days, goal, batch_size=Batch):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for episode in tqdm(range(num_episodes)):
        results = simulate_episode(model, S0, sigma, days, goal, flag=True, batch_size=batch_size)
        S_n, A_n, q_n, total_spent, actions, log_densities, bell_signals, episode_payoff = results

        optimizer.zero_grad()
        loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        loss = loss - torch.sum(log_densities)
        loss *= torch.mean(episode_payoff)
        print(loss)
        if episode % 50 == 0:
            print(f"Episode {episode}: Average Episode Payoff {torch.mean(episode_payoff)}, Loss {loss.item()}")

        loss.backward()
        optimizer.step()
"""
def evaluate_single_episode(model, S0, V0, mu, kappa, theta, sigma, N, Q,results, batch_size=2):
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
"""
def evaluate_policy(model, num_episodes, S0, sigma, days, goal, batch_size=Batch): 
    total_spent_list = []
    total_stocks_list = []
    A_n_list = []
    payoff_list = []
    final_day_list = []
    actions_list = []

    for _ in range(num_episodes):
        results = simulate_episode(model, S0, sigma, days, goal, flag=False, batch_size=batch_size)
        S_n, A_n, q_n, total_spent, actions, log_densities, bell_signals, episode_payoff = results
        final_day = (bell_signals > 1).nonzero(as_tuple=True)[0][0]
        total_spent_single = total_spent[final_day, 0]
        total_stocks = q_n[final_day, 0]
        total_spent_list.append(total_spent_single)
        total_stocks_list.append(total_stocks)
        A_n_list.append(A_n[final_day, 0]) 
        payoff_list.append(episode_payoff[0])
        final_day_list.append(final_day)
        actions_list.append(actions[:, 0])
        
 
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

def export_csv(actions, episode_payoff, S_n, A_n, q_n, total_spent, filename):
    actions_len = len(actions)

    if len(A_n) != actions_len or len(q_n) != actions_len or len(total_spent) != actions_len:
        raise ValueError("Les longueurs des listes ne correspondent pas.")

    payoff_list = [0] * (actions_len - 1) + [episode_payoff]
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
    verification=True
    for i, cloche_value in enumerate(cloche_n): 
        
        if float(cloche_value.item()) >= 1 and verification: 
            ax1.axvline(x=i, color="purple", linestyle='--', label="cloche_n = 1" if i == 0 else "")
            verification=False
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

#V0 = 0.04  # Volatilité initiale
#mu = 0.1   # Rendement attendu
#kappa = 2.0
#theta = 0.04
#rho = -0.7

model_name = input("Quel modèle voulez-vous utiliser ? (par exemple : Net (0), StockNetwork(1)) : ").strip()
 
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
            train_model(model, simulate_episode=simulate_episode, num_episodes=num_episodes, S0=S0, sigma=sigma, days=days, goal=Q,batch_size=Batch)
           
            save_path = input("Entrez le chemin pour sauvegarder le modèle : ").strip()
            model.save_model(save_path)
            print(f"Modèle sauvegardé à {save_path}")
 
    except Exception as e:
        print(f"Erreur lors du chargement ou de l'entraînement du modèle : {e}")
 
elif choice == 'e':
    # Entraînement d'un nouveau modèle
    num_episodes = int(input("Entrez le nombre d'épisodes pour l'entraînement : "))
    train_model(model, simulate_episode=simulate_episode, num_episodes=num_episodes, S0=S0, sigma=sigma, days=days, goal=Q,batch_size=Batch)
   
    save_path = input("Entrez le chemin pour sauvegarder le modèle : ").strip()
    try:
        model.save_model(save_path)
        print(f"Modèle sauvegardé à {save_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")
 
 
 
# Évaluation de la politique
num_episodes = 100
avg_total_spent, avg_total_stocks, avg_A_n, avg_payoff, avg_final_day, avg_actions = evaluate_policy(model, num_episodes, S0, sigma, days, Q, batch_size=Batch)
print(f"\nRésultats de l'évaluation de la politique sur {num_episodes} épisodes :")
print(f"Total dépensé en moyenne: {avg_total_spent}")
print(f"Total d'actions en moyenne: {avg_total_stocks}")
print(f"Prix moyen des actions: {avg_A_n}")
print(f"Payoff moyen: {avg_payoff}")
print(f"Jour final moyen: {avg_final_day}")
print(f"Actions moyennes par jour: {avg_actions}")
 
# Simulation d'un épisode et exportation des résultats
S_n, A_n, q_n, total_spent, actions, log_densities, cloche_n, episode_payoff = simulate_episode(
    model, S0, sigma, days, Q, flag=False, batch_size=Batch
)

# Exportation des résultats du premier batch
export_csv(actions[:, 0], episode_payoff[0], S_n[:,0], A_n[:, 0], q_n[:, 0], total_spent[:, 0], "episode_sans_heston.csv")
 
# Tracé des résultats pour le premier batch
plot_episode(S_n[:, 0], A_n[:, 0], q_n[:, 0], cloche_n[:, 0])