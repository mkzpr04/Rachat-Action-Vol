import torch
import numpy as np

def simulate_price(X, sigma, S0):
    # Simulation des prix
    S_n = np.zeros((X.shape[0] + 1, X.shape[1]))
    S_n[0, :] = S0
    S_n[1:] = S0 + np.cumsum(sigma * X, axis=0)
    return S_n

np.random.seed(0)
X = np.random.normal(0, 1, (60, 2))

prix =simulate_price(X,2, 100) 

def normalize(state, days, goal, S0):
    t, S_n, A_n, q_n, total_spent = state  # tenseurs
    return torch.vstack([
        torch.full_like(S_n, t / days),  # t normalisé
        S_n / S0,                        # Prix S_n normalisé
        A_n / S0,                        # A_n normalisé
        q_n / goal,                      # Nombre d'actions normalisé
        total_spent / (S0 * goal)        # Total dépensé normalisé
    ]).T

initial_state = (0, torch.tensor([100, 105], dtype=torch.float32), torch.tensor([100, 101], dtype=torch.float32),
                 torch.tensor([0, 0], dtype=torch.float32), torch.tensor([0, 0], dtype=torch.float32))

# Normalisez l'état initial
normalized_state = normalize(initial_state, 60, 100, 100)

days = 60
goal = 100
S0 = 100

# Affichage de l'état normalisé
print(normalized_state)


a =torch.randn_like(torch.tensor([1, 2, 3], dtype=torch.float32))
print(a)
u = np.random.uniform(0, 1,10)

#%%
import torch
import numpy as np

class RLAgent:
    def __init__(self):
        # Initialisation des couches et des fonctions d'activation
        self.hidden1 = torch.nn.Linear(10, 10)
        self.hidden2 = torch.nn.Linear(10, 10)
        self.hidden3 = torch.nn.Linear(10, 10)
        self.mean_output = torch.nn.Linear(10, 1)
        self.bell_output = torch.nn.Linear(10, 1)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        self.act3 = torch.nn.ReLU()
        self.act_output = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        mean = self.mean_output(x)
        bell = self.act_output(self.bell_output(x))
        return mean, bell

    def sample_action(self, state, goal, days):
        mean, bell = self.forward(state)
        std = (goal / days) * 0.05

        total_stock_target = mean + std * torch.randn_like(mean)  # mu + sigma * N(0,1)
        u = torch.tensor(np.random.uniform(0, 1, size=bell.shape), dtype=torch.float32)  # Générer un tenseur de scalaires uniformes
        bell = (u < bell)  # Comparaison élément par élément, résultat est un tenseur booléen
        
        # Calcul de la vraisemblance de la première action
        log_density = -0.5 * torch.log(2 * torch.tensor(np.pi) * (std * std)) - ((total_stock_target - mean) ** 2) / (2 * (std * std))

        # Calcul de la probabilité cumulative
        prob = 0.5 * (1 + torch.erf((total_stock_target - mean) / (std * torch.sqrt(torch.tensor(2.0)))))

        # Calcul de la vraisemblance d'avoir sonné la cloche
        bell_prob = bell.float() * prob + (1 - bell.float()) * (1 - prob)
        log_density += torch.log(bell_prob)
        
        return total_stock_target, bell, log_density

# Exemple d'utilisation
agent = RLAgent()
state = torch.randn(5, 10)  # Exemple d'état avec batch size 5
goal = 100
days = 20
total_stock_target, bell, log_density = agent.sample_action(state, goal, days)
print(total_stock_target)
print(bell)
print(log_density)

bell= torch.tensor([0.1, 0.2])
u = np.random.uniform(0, 1, size=bell.shape)
u = torch.tensor(u, dtype=torch.float32)
bell = (u < bell).float()
print(bell)