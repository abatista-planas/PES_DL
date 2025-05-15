import sys
from math import floor
from typing import Tuple

import numpy as np
import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary  # type: ignore

from pes_1D.data_generator import (
    generate_discriminator_training_set,
)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

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
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """Evaluates the model on the test set and returns the loss and accuracy."""

        self.eval()
        X, y = next(iter(test_loader))
        with torch.no_grad():  # deactivates autograd
            y_eval = self.forward(X)

        return (
            float(100 * torch.mean(((y_eval > 0) == y).float()).item()),
            y_eval,
            y,
        )

    def train_model(
        self,
        train_loader: DataLoader,
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
            self.train()

            # loop over training data batches
            batchAcc = []
            batchLoss = []

            # print out a status message
            if (epochi + 1) % 50 == 0:
                msg = f"Finished epoch {epochi + 1}/{num_epochs}"
                sys.stdout.write("\r" + msg)

            for X, y in train_loader:
                # forward pass and loss

                y_pred = self.forward(X)
                loss = criterion(y_pred, y)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # loss from this batch
                batchLoss.append(loss.item())

                # compute training accuracy for this batch
                batchAcc.append(100 * torch.mean(((y_pred > 0) == y).float()).item())

            # end of batch loop...

            # now that we've trained through the batches, get their average training accuracy
            trainAcc.append(float(np.mean(batchAcc)))

            # and get average losses across the batches
            losses[epochi] = np.mean(batchLoss)

        return trainAcc, losses

    def pre_train(self, criterion, n_samples=5000, num_epochs=300):
        """Pre-trains the model on the training set and returns the pre-trained model and losses."""

        grid_size = self.params["grid_size"]
        batch_size = 50
        test_split = 0.5
        pes_name_list = ["lennard_jones"]  # PES names to generate
        deformation_list = np.array(
            [
                "outliers",
                "oscillation",
                "pulse_random_fn",
                "piecewise_random",
                "random_functions",
            ]
        )  # Types of deformation to generate

        probability_deformation = np.array(
            [0.25, 0.35, 0.3, 0.05, 0.05]
        )  # Probability of deformation to generate

        properties_list = [
            "energy",
            "derivative",
            "inverse_derivative",
        ]  # List of properties to use for training

        properties_format = "table_1D"  # Format [concatenated array or table] of properties to use for training

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self = self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        train_loader, test_loader, _, _ = generate_discriminator_training_set(
            n_samples=n_samples,
            batch_size=batch_size,
            grid_size=grid_size,
            pes_name_list=pes_name_list,
            properties_list=properties_list,
            deformation_list=deformation_list,
            probability_deformation=probability_deformation,
            properties_format=properties_format,
            test_split=test_split,
            device=device,
        )

        # Train the model
        print("Pre-Training the Discriminator...")
        trainAcc, _ = self.train_model(
            train_loader,
            criterion,
            optimizer,
            num_epochs,
        )

        test_results = self.test_model(test_loader)
        testAcc = test_results[0]  # Extract the first tensor as test accuracy

        print(f"Training Accuracy:{trainAcc[-1]}")
        print(f"Test Accuracy:{testAcc}")


class AnnDiscriminator(Discriminator):
    def __init__(self, model_paramaters):
        super(AnnDiscriminator, self).__init__()

        self.model_paramaters = model_paramaters
        # input layer
        self.layers["input"] = nn.Linear(
            model_paramaters["in_features"], model_paramaters["hidden_layers"][0]
        )

        # hidden layers
        for i in range(len(model_paramaters["hidden_layers"]) - 1):
            self.layers[f"hidden_{i}"] = nn.Linear(
                model_paramaters["hidden_layers"][i],
                model_paramaters["hidden_layers"][i + 1],
            )

        self.layers["output"] = nn.Linear(
            model_paramaters["hidden_layers"][-1], model_paramaters["out_features"]
        )

    def forward(self, x):
        # input layer
        x = F.relu(self.layers["input"](x))

        # hidden layers
        for i in range(len(self.layers) - 2):
            x = F.relu(self.layers[f"hidden_{i}"](x))

        # output layer
        x = self.layers["output"](x)
        return x

    def summary(self):
        return self.generic_summary(
            "AnnDiscriminator", (1, self.model_paramaters["in_features"])
        )


class CnnDiscriminator(Discriminator):
    def __init__(self, model_paramaters):
        """_summary_

        Args:
            model_paramaters (_type_): "in_features"

        """
        super(CnnDiscriminator, self).__init__()
        self.params = model_paramaters
        sz = self.params["grid_size"]
        self.layers["cv_0"] = nn.Conv1d(
            self.params["in_channels"],
            self.params["hidden_channels"][0],
            kernel_size=self.params["kernel_size"][0],
            stride=1,
            padding=0,
        )

        self.layers["cv_1"] = nn.Conv1d(
            self.params["hidden_channels"][0],
            self.params["hidden_channels"][1],
            kernel_size=self.params["kernel_size"][1],
            stride=1,
            padding=0,
        )

        for i in range(len(self.params["pool_size"])):
            sz = floor(
                (sz - self.params["kernel_size"][i] + 1) / self.params["pool_size"][i]
            )

        self.layers["fc_0"] = nn.Linear(sz * self.params["hidden_channels"][-1], 128)
        self.layers["output"] = nn.Linear(128, 1)

    def forward(self, x):

        x = F.relu(self.layers["cv_0"](x))
        x = F.avg_pool1d(x, kernel_size=self.params["pool_size"][0])
        x = F.relu(self.layers["cv_1"](x))
        x = F.avg_pool1d(x, kernel_size=self.params["pool_size"][1])

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.layers["fc_0"](x))
        # output layer
        x = self.layers["output"](x)
        return x

    def summary(self):
        print("CnnDiscriminator")
        return self.generic_summary(
            "CnnDiscriminator", (self.params["in_channels"], self.params["grid_size"])
        )


class SRDiscriminator(Discriminator):
    def __init__(self, model_paramaters):
        """_summary_

        Args:
            model_paramaters (_type_): "in_features"

        """
        super(CnnDiscriminator, self).__init__()
        self.params = model_paramaters
        sz = self.params["grid_size"]
        self.layers["cv_0"] = nn.Conv1d(
            self.params["in_channels"],
            self.params["hidden_channels"][0],
            kernel_size=self.params["kernel_size"][0],
            stride=1,
            padding=0,
        )

        self.layers["cv_1"] = nn.Conv1d(
            self.params["hidden_channels"][0],
            self.params["hidden_channels"][1],
            kernel_size=self.params["kernel_size"][1],
            stride=1,
            padding=0,
        )

        for i in range(len(self.params["pool_size"])):
            sz = floor(
                (sz - self.params["kernel_size"][i] + 1) / self.params["pool_size"][i]
            )

        self.layers["fc_0"] = nn.Linear(sz * self.params["hidden_channels"][-1], 128)
        self.layers["output"] = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.layers["cv_0"](x))
        x = F.avg_pool1d(x, kernel_size=self.params["pool_size"][0])
        x = F.relu(self.layers["cv_1"](x))
        x = F.avg_pool1d(x, kernel_size=self.params["pool_size"][1])

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.layers["fc_0"](x))
        # output layer
        x = self.layers["output"](x)
        return x

    def summary(self):
        print("CnnDiscriminator")
        return self.generic_summary(
            "CnnDiscriminator", (self.params["in_channels"], self.params["grid_size"])
        )
