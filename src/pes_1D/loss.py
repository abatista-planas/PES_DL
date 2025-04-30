import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

        def forward(self, y_pred, y_true):
            """
            Compute the loss between the predicted and true values.

            Parameters:
            y (torch.Tensor): True values.
            y_pred (torch.Tensor): Predicted values.

            Returns:
            torch.Tensor: Computed loss.
            """
            loss = torch.mean((y_true - y_pred) ** 2)

            return loss
