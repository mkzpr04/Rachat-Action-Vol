import torch 

'''
def payoff(goal, A_n, total_spent,n):
    return goal * A_n[n,:] - total_spent[n,:]

def expected_payoff(A_n, total_spent, bell_n, goal, N):
    payoff_values = torch.zeros(bell_n.shape[1], dtype=torch.float32, device=bell_n.device)
    for n in range(1,N+1):
        # Compute the product of (1 - bell_signals) for all i < n
        product=torch.ones(bell_n.shape[1], dtype=torch.float32, device=bell_n.device)
        for i in range(1,n):
            #print(bell_n[n,:])
            #print(bell_n[i,:])
            product*=bell_n[n,:]*(1-bell_n[i,:])*payoff(goal, A_n, total_spent, n)
            #print(payoff(goal, A_n, total_spent, n))

        # Add the contribution for step n
        payoff_values += product 

    return payoff_values


# Test simple
A_n = torch.tensor([[10, 20], [15, 25], [20, 30], [25, 35], [30, 40]], dtype=torch.float32)
total_spent = torch.tensor([[5, 10], [7, 12], [10, 15], [12, 18], [15, 20]], dtype=torch.float32)
bell_n = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
goal = 50
days = 4

payoff_values = expected_payoff(A_n, total_spent, bell_n, goal, days)
print(payoff_values)  # Vérifiez si la sortie est correcte
'''
def expected_payoff(A_n, total_spent, bell_n, goal, N):
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

def payoff(goal, A_n, total_spent, n):
    """
    Calcule le payoff à l'étape n en utilisant la formule :
    PnL_n = goal * A_n[n] - total_spent[n]

    Arguments :
    - goal : objectif d'actions à atteindre
    - A_n : moyenne des prix des actions (Tensor)
    - total_spent : montant total dépensé jusqu'à l'étape n (Tensor)
    - n : étape courante

    Retourne :
    - pnl_n : le payoff à l'étape n
    """
    return goal * A_n[n, :] - total_spent[n, :]


def expected_payoff(A_n, total_spent, bell_n, goal, N):
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
        pnl_n = payoff(goal, A_n, total_spent, n)  # Calcul du payoff à l'étape n

        # Mise à jour des valeurs de payoff
        payoff_values += product * p_n * pnl_n

    return payoff_values
"""
# Données fictives
A_n = torch.tensor([[10, 20], [15, 25], [20, 30], [25, 35], [30, 40]], dtype=torch.float32)
total_spent = torch.tensor([[5, 10], [7, 12], [10, 15], [12, 18], [15, 20]], dtype=torch.float32)
bell_n = torch.tensor([[0, 0], [0.2, 0.5], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]], dtype=torch.float32)
goal = 50
days = 4

# Calcul du payoff attendu
payoff_values = expected_payoff(A_n, total_spent, bell_n, goal, days)
print("Payoff attendu :", payoff_values)
"""



import torch

def payoff(Q, A_n, X_n, q_n, S_n, L, n):
    """
    Calcule le profit et la perte au jour n en utilisant la formule mise à jour.

    Arguments :
    - Q : Nombre total d'actions à acheter.
    - A_n : Valeur moyenne des actions détenues jusqu'au jour n.
    - X_n : Cash dépensé jusqu'au jour n.
    - q_n : Nombre d'actions achetées jusqu'au jour n.
    - S_n : Prix des actions au jour n.
    - L : Fonction convexe des coûts de transaction, modélisant `ell`.
    - n : Jour actuel.

    Retour :
    - PnL_n : Profit et perte calculé pour le jour n.
    """
    # Quantité restante à acheter
    remaining_shares = Q - q_n

    # Calcul du coût d'exécution pour les parts restantes
    execution_cost = L(remaining_shares)

    # Profit et perte
    pnl = Q * A_n[n] - X_n[n] - remaining_shares * S_n[n] - execution_cost
    return pnl

# Exemple d'utilisation

# Paramètres simulés
N = 25  # Nombre de jours
Q = 50  # Objectif total d'actions à acheter
A_n = torch.linspace(40, 50, N + 1)  # Valeur moyenne simulée
X_n = torch.cumsum(torch.linspace(10, 20, N + 1), dim=0)  # Cash dépensé cumulé
q_n = torch.cumsum(torch.ones(N + 1), dim=0)  # Actions achetées (1 action par jour)
S_n = torch.linspace(45, 55, N + 1)  # Prix simulés des actions

# Fonction convexe L modélisant les coûts d'exécution
def L(x):
    return x**2 + 0.1 * x  # Ex. de fonction convexe : quadratique avec un terme linéaire

# Calcul du payoff pour un jour donné (e.g., jour 10)
day = 10
result = payoff(Q, A_n, X_n, q_n, S_n, L, day)
print(f"Payoff pour le jour {day} :", result.item())
