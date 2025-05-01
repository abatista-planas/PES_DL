from typing import Tuple

import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary  # type: ignore


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layers = nn.ModuleDict()
        pass

    def forward_profiler(self, x, mask):
        with profiler.record_function("LINEAR PASS"):
            x = self.forward(x)

        with profiler.record_function("MASK INDICES"):
            threshold = x.sum(axis=1).mean()
            hi_idx = (mask > threshold).nonzero(as_tuple=True)

        return x, hi_idx

    def forward(self, x):
        pass

    def reset(self):
        for layer in self.layers:
            if hasattr(self.layers[layer], "reset_parameters"):
                self.layers[layer].reset_parameters()

    def generic_summary(self, model_name, input_size):
        print("Model Summary: ")
        print("Model architecture: ", model_name)
        return summary(self, input_size)

    def count_parameters(self) -> int:
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def test_model(
        self,
        test_loader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Evaluates the model on the test set and returns the loss and accuracy."""

        self.eval()
        running_loss = 0.0
        total_samples = 0

        with torch.no_grad():  # deactivates autograd
            for X, y in test_loader:
                batch_size = X.shape[0]
                y_pred = self.forward(X)
                loss = criterion(y_pred, y)

                running_loss += loss.item() * batch_size
                total_samples += batch_size

            test_avg_loss = running_loss / total_samples

            return (
                test_avg_loss,
                y_pred,
                y,
            )

    def train_model(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
    ) -> Tuple[list[float], float]:
        """Trains the model on the training set and returns the trained model and losses."""

        # initialize losses
        trainAcc = []
        loss_arr = []

        # loop over epochs
        for epochi in range(num_epochs):
            # switch on training mode
            self.train()

            # loop over training data batches
            running_loss = 0.0
            total_samples = 0

            for X, y in train_loader:
                # forward pass and loss
                batch_size = X.shape[0]
                y_pred = self.forward(X)

                loss = criterion(y_pred, y)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_size
                total_samples += batch_size
            # end of batch loop...

            # now that we've trained through the batches, get their average training accuracy
            avg_loss = running_loss / total_samples
            loss_arr.append(avg_loss)
            trainAcc.append(avg_loss)

        return loss_arr, trainAcc[-1]


class SuperResolution1D(Generator):
    def __init__(self, upscale_factor=4, base_channels=64):
        super().__init__()
        # 1) Learnable upsampling: ConvTranspose1d takes (C_in, C_out, kernel, stride, padding)
        #    Here we go from 1→base_channels, upsampling by factor=4.
        self.upconv = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=base_channels,
            kernel_size=upscale_factor,
            stride=upscale_factor,
            padding=0,
            output_padding=0,
        )
        # 2) Refinement layers
        self.refine = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_channels // 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        x: (batch, 1, n)
        returns: (batch, 1, 4*n)
        """
        x = self.upconv(x)
        x = self.refine(x)
        return x
