"""Training and evaluation functions for a toy model."""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
) -> Tuple[list[float], torch.Tensor]:
    """Trains the model on the training set and returns the trained model and losses."""
    
        # initialize losses
    losses = torch.zeros(num_epochs)
    trainAcc = []
    

    # loop over epochs
    for epochi in range(num_epochs):
        # switch on training mode
        model.train()

        # loop over training data batches
        batchAcc = []
        batchLoss = []
        for X, y in train_loader:
            
            # forward pass and loss
            
            y_pred = model.forward(X)
            loss = criterion(y_pred, y)
            
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss from this batch
            batchLoss.append(loss.item())

            # compute training accuracy for this batch
            # batchAcc.append( 100*torch.mean(((yHat>0) == y).float()).item() )
            batchAcc.append(
                100 * torch.mean(((y_pred>0) == y).float()).item()
            )

        # end of batch loop...

        # now that we've trained through the batches, get their average training accuracy
        trainAcc.append(float(np.mean(batchAcc)))

        # and get average losses across the batches
        losses[epochi] = np.mean(batchLoss)

    return trainAcc, losses


def test_model(
    test_loader: DataLoader,
    model: nn.Module,
    device: str = "cpu",
) -> float:
    """Evaluates the model on the test set and returns the loss and accuracy."""
 
    model.eval()
    X, y = next(iter(test_loader))  # extract X,y from test dataloader
    X, y = X.to(device), y.to(device)
    with torch.no_grad():  # deactivates autograd
        y_eval = model.forward(X)


    return float(100 * torch.mean(((y_eval>0) == y).float()).item())
