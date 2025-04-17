"""Training and evaluation functions for a toy model."""

from typing import Tuple

import torch
import torch.nn as nn


def train_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    verbose: bool = False,
) -> list[float]:
    """Trains the model on the training set and returns the trained model and losses."""
    losses: list[float] = []

    for epoch in range(epochs):
        y_pred = model.forward(X_train)
        model.train()

        loss = criterion(y_pred, y_train)
        losses.append(loss.item())
        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses


def test_model(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    model: nn.Module,
    criterion: nn.Module,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Evaluates the model on the test set and returns the loss and accuracy."""
    with torch.no_grad():
        y_eval = model.forward(X_test)
        loss = criterion(y_eval, y_test)
        test_loss = loss.item()
        accuracy = (
            100 * torch.sum(torch.argmax(y_eval, dim=1) == y_test).item() / len(y_test)
        )
        if verbose:
            print("Evaluation:")
            print(f"Test Loss: {test_loss}")
            print("Accuracy (%): ", accuracy)

        return test_loss, accuracy
