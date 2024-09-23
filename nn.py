import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim


enable_cuda = True
n_neurons_per_layer = 512
n_layers = 5


# decay_a = 2e-6
# decay_b = .004
# decay_c = .6


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.n_layers = 5
        self.n_neurons_per_layer = 512

        self.fc1 = nn.Linear(4, self.n_neurons_per_layer)
        for _ in range(self.n_layers - 1):
            self.__setattr__(
                "fc" + str(_ + 2),
                nn.Linear(self.n_neurons_per_layer, self.n_neurons_per_layer),
            )

        self.__setattr__("fc" + str(_ + 3), nn.Linear(self.n_neurons_per_layer, 2))
        self.scale = nn.Parameter(torch.tensor(0.0, requires_grad=True))

        self.Q = 20e0
        self.N = 63
        self.xi_0 = 0.6**2
        self.S0 = 45.0
        

    def forward(self, x):
        input = x
        for _ in range(self.n_layers):
            x = nn.functional.relu(self.__getattr__("fc" + str(_ + 1))(x))
        x = self.__getattr__("fc" + str(_ + 2))(x)
        # x[:, 1] = torch.sigmoid(self.scale * (x[:, 1] - input[:, 3]))
        x[:, 1] = torch.sigmoid(x[:, 1])
        x[:, 0] = torch.minimum(            
                torch.maximum((x[:, 0] + input[:, 0] + 0.5) * self.Q, torch.tensor(0)),
                torch.tensor(self.Q)
        )
        return x.T

    def normalize(self, n_plus_one, S_tensor_step, A_tensor_step, q_previous):
        return torch.vstack(
            [
                torch.full_like(S_tensor_step, (n_plus_one) / self.N - 0.5),
                (S_tensor_step - self.S0) / np.sqrt((n_plus_one) * self.xi_0),
                (A_tensor_step - S_tensor_step) / np.sqrt((n_plus_one) * self.xi_0),
                q_previous / self.Q - 0.5,
            ]
        ).T

    def load_model(self, path, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
model = Net()
#weights_before = {name: param.clone() for name, param in model.named_parameters()}

model.load_model("trained_model.pt")

#weights_after = {name: param.clone() for name, param in model.named_parameters()}


print(model.fc1.weight)

state = torch.tensor(((23.0), (41.0),( 43.0), (19.0)), dtype=torch.float32)
print(state)
state=model.normalize(23.0, torch.tensor(41.0),torch.tensor( 43.0), torch.tensor(19.0))
actions = model.forward(state)
print(actions)
    
#weights_equal = {name: torch.equal(weights_before[name], weights_after[name]) for name in weights_before}
#print(weights_equal)"""