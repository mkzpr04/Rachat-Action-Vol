import numpy as np
import torch
from RL_agent import StockNetwork
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from nn import Net

Batch=64
S0 = 45
sigma = 0.6
N = 63  #days 
Q = 20  #goal
jour_cloche = 22

def simulate_price(X, sigma, S0):
    # Simulation des prix
    S_n = np.zeros((X.shape[0] + 1, X.shape[1]))
    S_n[0, :] = S0
    S_n[1:] = S0 + np.cumsum(sigma * X, axis=0)
    return S_n

def payoff(A_n, total_spent):
    return 20 * A_n - total_spent

"""
def payoff(q_n, A_n, total_spent,n):
    return q_n[n,:] * A_n[n,:] - total_spent[n,:]
"""

def calculate_condition(bell_signals, q_n, t, jour_cloche, Q, N):
    return ((bell_signals[t, :] >= 0.5) & (t >= jour_cloche-1) & (q_n[t, :] >= Q)) 

def expected_payoff(A_n, total_spent, bell_signals, q_n, N):

    payoff_values = torch.zeros(bell_signals.shape[1], dtype=torch.float32, device=bell_signals.device)

    for n in range(1, N+1):
        # Produit des termes (1 - p_k) pour k = 1 à n-1
        product = torch.ones(bell_signals.shape[1], dtype=torch.float32, device=bell_signals.device)
        for k in range(1, n):
            product *= (1 - bell_signals[k, :])  # Produit de (1 - p_k)

        # Contribution du terme courant (p_n * PnL_n)
        p_n = bell_signals[n, :]
        pnl_n = payoff(q_n, A_n, total_spent, n)
        # Calcul du payoff à l'étape n

        # Mise à jour des valeurs de payoff
        payoff_values += product * p_n * pnl_n

    return payoff_values

def sample_action(model, state, Q, N):
    """
    Fonction pour échantillonner une action à partir du modèle.
    """
    if isinstance(model, StockNetwork):
        # Si le modèle est StockNetwork
        return model.sample_action(state, Q, N)
    else:
        # Si le modèle est Net -> méthode générique
        output = model.forward(state)
        mean = output[0]
        bell = output[1]
        
        std = (Q / N) * 0.05
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.tensor(0.0, requires_grad=True)
        
        total_stock_target = mean + std * torch.randn_like(mean)  # mu + sigma * N(0,1)
        log_density = -0.5 * torch.log(2 * torch.tensor(np.pi) * (std * std)) - ((total_stock_target - mean) * (total_stock_target - mean)) / (2 * (std * std))
        log_density = log_density.detach().requires_grad_(True)
        return total_stock_target, bell, log_density


def simulate_episode(model, S0, sigma, N, Q, flag, batch_size=Batch):

    q_n = torch.zeros((N+1, batch_size), dtype=torch.float32)
    A_n = torch.zeros((N+1, batch_size), dtype=torch.float32)
    actions = torch.zeros((N+1, batch_size), dtype=torch.float32)
    bell_signals = torch.zeros((N+1, batch_size), dtype=torch.float32)
    total_spent = torch.zeros((N+1, batch_size), dtype=torch.float32)
    log_densities = torch.zeros((N+1, batch_size), dtype=torch.float32, requires_grad=True)
    episode_payoff =torch.full((batch_size,),float('nan'), dtype=torch.float32)
    new_log_densities = torch.zeros_like(log_densities)
    payoff_calculated = torch.zeros(batch_size, dtype=torch.bool)

    if not flag: # lorsqu'on évalue le model
        np.random.seed(0)
    X = np.random.normal(0, 1, (N, batch_size))
    S_n = simulate_price(X, sigma, S0)
    S_n = torch.tensor(S_n, dtype=torch.float32)
    
    for t in range(N+1):
        A_n[t, :] = torch.mean(S_n[1:t+1, :], axis=0) if t > 0 else S0
        
    for t in range(N):
        if isinstance(model, Net):
            state = model.normalize(t+1 , S_n[t, :], A_n[t, :], q_n[t, :])  
        else: 
            state = model.normalize((t, S_n[t, :], A_n[t, :], q_n[t, :], total_spent[t, :]), N, Q, S0)

        log_density = None

        if flag:
            total_stock_target, bell, log_density = sample_action(model, state, Q, N)
        else:
            if isinstance(model, Net):
                etat = model.forward(state)
                total_stock_target = etat[0]
                bell = etat[1]
            else:
                etat = model.forward(state)
                total_stock_target = etat[0]
                bell = etat[1]


        # MAJ des états
        q_n[t+1,:] = total_stock_target if t < N-1 else Q # * (Q - q_n[t, :]) if t < N - 1 else Q
        v_n = q_n[t+1, :] - q_n[t, :]
        total_spent[t+1, :] = total_spent[t, :] + v_n * S_n[t+1, :]
        #log_densities[t, :] = log_density if log_density is not None else 0
        new_log_densities[t, :] = log_density if log_density is not None else 0
        actions[t, :] = v_n
        bell_signals[t, :] = bell
        condition = calculate_condition(bell_signals, q_n, t, jour_cloche, Q, N) # Condition pour vérifier si le signal de cloche est activé, si t >= 22, et si q_n[t+1, :] est supérieur ou égal à Q
        
        if condition.any(): # Si la condition est remplie pour au moins un batch
            bell_signals[t, condition]=bell_signals[t, condition]+1
            q_n[t+1:, condition] = q_n[t, condition]
            new_payoff_condition = condition & ~payoff_calculated 
            episode_payoff[new_payoff_condition] = payoff(A_n[t, new_payoff_condition], total_spent[t, new_payoff_condition])
            payoff_calculated[new_payoff_condition] = True
            #A_n[N, condition] = torch.mean(S_n[1:N + 1, :], axis=0)[condition]
        
    condition = q_n[N, :] < Q
    if condition.any():
        #A_n[N, condition] = torch.mean(S_n[1:N + 1, :], axis=0)[condition]
        final_adjustment = Q - q_n[N, condition]
        total_spent[N, condition] += final_adjustment * S_n[N, condition]
        actions[-1, condition] += final_adjustment
        q_n[N, condition] = Q
        episode_payoff[condition] = payoff(A_n[N, condition], total_spent[N, condition])
    bell_signals[N,:]=1
    #episode_payoff = expected_payoff(A_n, total_spent, bell_signals, q_n, N)
    log_densities = new_log_densities
    return S_n, A_n, q_n, total_spent, actions, log_densities, bell_signals, episode_payoff

def train_model(model, simulate_episode, num_episodes, S0, sigma, N, Q, batch_size=Batch):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for episode in tqdm(range(num_episodes)):
        results = simulate_episode(model, S0, sigma, N, Q, flag=True, batch_size=batch_size)
        S_n, A_n, q_n, total_spent, actions, log_densities, bell_signals, episode_payoff = results

        episode_payoff = torch.tensor(episode_payoff, dtype=torch.float32)
        episode_payoff_normalized = (episode_payoff - torch.mean(episode_payoff)) / torch.std(episode_payoff)

        loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
        #loss = loss - torch.sum(log_densities)
        #loss *= torch.mean(episode_payoff)
        loss = -torch.mean(log_densities * episode_payoff_normalized)
        if episode % 50 == 0:
            print(f"Episode {episode}: Average Episode Payoff {torch.mean(episode_payoff_normalized)}, Loss {loss.item()}")

        optimizer.zero_grad()
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

def evaluate_policy(model, num_episodes, S0, sigma, N, Q, batch_size=Batch): 
    total_spent_list = []
    total_stocks_list = []
    A_n_list = []
    payoff_list = []
    final_day_list = []
    actions_list = []

    for _ in range(num_episodes):
        results = simulate_episode(model, S0, sigma, N, Q, flag=True, batch_size=batch_size)
        S_n, A_n, q_n, total_spent, actions, log_densities, bell_signals, episode_payoff = results
        final_day = (bell_signals >= 1).nonzero(as_tuple=True)[0][0]
        total_spent_single = total_spent[final_day, 0]
        total_stocks = q_n[final_day, 0]
        total_spent_list.append(total_spent_single.detach().cpu().numpy())
        total_stocks_list.append(total_stocks.detach().cpu().numpy())
        A_n_list.append(A_n[final_day, 0].detach().cpu().numpy())
        payoff_list.append(episode_payoff[0].detach().cpu().numpy())
        final_day_list.append(final_day)
        actions_list.append(actions[:, 0].detach().cpu().numpy())
        
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
    ax1.plot(S_n.detach().cpu().numpy(), label="S_n (Prix de l'action au jour n)", color="blue")
    ax1.plot(A_n.detach().cpu().numpy(), label="A_n (Prix moyen des actions aux jours n)", color="green")
    ax1.tick_params(axis='y', labelcolor='black')
    verification=True
    for i, cloche_value in enumerate(cloche_n): 
        
        if float(cloche_value.item()) >= 1 and verification: 
            ax1.axvline(x=i, color="purple", linestyle='--', label="cloche_n = 1" if i == 0 else "")
            verification=False
    ax2 = ax1.twinx()  
    ax2.set_ylabel('q_n en valeur réelle', color='red')
    ax2.plot(q_n.detach().cpu().numpy(), label="q_n (Quantité totale d'actions au jour n)", color="red", linestyle='-')
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

model_name = input("Quel modèle voulez-vous utiliser ? (par exemple : Net(0), StockNetwork(1), etc.) : ").strip()
try:
    if model_name == "1":
        model = StockNetwork()
    if model_name == "0":
        model = Net()

except KeyError:
    raise ValueError(f"Le modèle '{model_name}' n'est pas reconnu. Assurez-vous que le nom du modèle est correct.")
 

# Choix de l'utilisateur pour charger ou entraîner le modèle
choice = get_user_choice("Voulez-vous charger un modèle existant (c) ou entraîner un nouveau modèle (e) ? (c/e) : ", ['c', 'e'])
 
if choice == 'c':
    model_path = input("Entrez le chemin du modèle à charger (0) trained_model.pt (1) trained_model_.pt: ").strip()
    try:
        if model_path == "0":
            model.load_model("trained_model.pt")
            print(f"Modèle chargé à partir de trained_model.pt")
        if model_path == "1":
            model.load_model("trained_model_.pt")
            print(f"Modèle chargé à partir de trained_model_.pt")
        else: 
            model.load_model(model_path)
            print(f"Modèle chargé à partir de {model_path}")
        
        continue_training = get_user_choice("Souhaitez-vous continuer l'entraînement du modèle ? (o/n) : ", ['o', 'n'])
        if continue_training == 'o':
            num_episodes = int(input("Entrez le nombre d'épisodes supplémentaires pour l'entraînement : "))
            train_model(model, simulate_episode=simulate_episode, num_episodes=num_episodes, S0=S0, sigma=sigma, N=N, Q=Q,batch_size=Batch)
            
            results = simulate_episode(model, S0, sigma, N, Q, flag=True, batch_size=Batch)
            S_n, A_n, q_n, total_spent, actions, log_densities, bell_signals, episode_payoff = results
            
            avg_total_spent, avg_total_stocks, avg_A_n, avg_payoff, avg_final_day, avg_actions = evaluate_policy(model, num_episodes, S0, sigma, N, Q, batch_size=Batch)
            plot_episode(S_n[:, 0], A_n[:, 0], q_n[:, 0], bell_signals[:, 0])
            
            save_path = input("Entrez le chemin pour sauvegarder le modèle : ").strip()
            model.save_model(save_path)
            print(f"Modèle sauvegardé à {save_path}")
        if continue_training == 'n':
            results = simulate_episode(model, S0, sigma, N, Q, flag=False, batch_size=Batch)
            S_n, A_n, q_n, total_spent, actions, log_densities, bell_signals, episode_payoff = results
            
            num_episodes=100
            avg_total_spent, avg_total_stocks, avg_A_n, avg_payoff, avg_final_day, avg_actions = evaluate_policy(model, num_episodes, S0, sigma, N, Q, batch_size=Batch)
            plot_episode(S_n[:, 0], A_n[:, 0], q_n[:, 0], bell_signals[:, 0])
 
    except Exception as e:
        print(f"Erreur lors du chargement ou de l'entraînement du modèle : {e}")
 
elif choice == 'e':
    # Entraînement d'un nouveau modèle
    num_episodes = int(input("Entrez le nombre d'épisodes pour l'entraînement : "))
    train_model(model, simulate_episode=simulate_episode, num_episodes=num_episodes, S0=S0, sigma=sigma, N=N, Q=Q,batch_size=Batch)
    
    results = simulate_episode(model, S0, sigma, N, Q, flag=True, batch_size=Batch)
    S_n, A_n, q_n, total_spent, actions, log_densities, bell_signals, episode_payoff = results
    

    avg_total_spent, avg_total_stocks, avg_A_n, avg_payoff, avg_final_day, avg_actions = evaluate_policy(model, num_episodes, S0, sigma, N, Q, batch_size=Batch)
    plot_episode(S_n[:, 0], A_n[:, 0], q_n[:, 0], bell_signals[:, 0])
    
    save_path = input("Entrez le chemin pour sauvegarder le modèle : ").strip()
    try:
        model.save_model(save_path)
        print(f"Modèle sauvegardé à {save_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du modèle : {e}")


print(f"\nRésultats de l'évaluation de la politique sur {num_episodes} épisodes :")
print(f"Total dépensé en moyenne: {avg_total_spent}")
print(f"Total d'actions en moyenne: {avg_total_stocks}")
print(f"Prix moyen des actions: {avg_A_n}")
print(f"Payoff moyen: {avg_payoff}")
print(f"Jour final moyen: {avg_final_day}")
print(f"Actions moyennes par jour: {avg_actions}")
 


# Exportation des résultats du premier batch
#export_csv(actions[:, 0], episode_payoff[0], S_n[:,0], A_n[:, 0], q_n[:, 0], total_spent[:, 0], "episode_sans_heston.csv")
 
