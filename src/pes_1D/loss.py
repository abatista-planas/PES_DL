# import torch
# import torch.nn as nn


# class Loss(nn.Module):
#     def __init__(self):
#         super(Loss, self).__init__()

#         def forward(self, y_pred, y_true):
#             """
#             Compute the loss between the predicted and true values.

#             Parameters:
#             y (torch.Tensor): True values.
#             y_pred (torch.Tensor): Predicted values.

#             Returns:
#             torch.Tensor: Computed loss.
#             """
#             loss = torch.mean((y_true - y_pred) ** 2)

#             return loss

import matplotlib.pyplot as plt
import numpy as np

from pes_1D.utils import NoiseFunctions

r0 = np.random.uniform(3.0, 10.0)
A = np.random.uniform(1, 2)
n = int(np.random.uniform(1, 25))

r = np.linspace(1.0, 10, 1000, dtype=np.float64)

oscillation_function, oscillation_derivative = NoiseFunctions.oscillation(r, r0, A, n)

# labels,oscillation_function,deriv = NoiseFunctions.piecewise_random(r,4.0)


plt.plot(r, oscillation_function)
plt.show()
