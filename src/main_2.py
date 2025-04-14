import torch

x = torch.tensor(2.0,requires_grad=True)
y = x**2 + 2*x + 1

print(y)
print(x)

