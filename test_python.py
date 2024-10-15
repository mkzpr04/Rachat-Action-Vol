import torch

tens =torch.tensor([[0., 1., 1., 1., 0.],
        [0., 1., 1., 1., 0.]])

print(tens.shape)
print(3*(10*tens[:,3]))

torch.tensor([20., 20.]) * torch.tensor([2.,2.])
