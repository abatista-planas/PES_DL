import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pes_1D.data_generator import (
    generate_disciminator_training_set_from_G,
    generate_true_pes_samples,
)
from pes_1D.discriminator import CnnDiscriminator  # type: ignore

n_samples = [1000, 1000]
batch_size = 50
num_epochs = 1000
pes_name_list = ["lennard_jones", "morse"]
upscale_factor = 100
lr_grid_size = 8
hr_grid_size = lr_grid_size * upscale_factor
test_split = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

properties_format = (
    "table_1D"  # Format [concatenated array or table] of properties to use for training
)


df_high_res = generate_true_pes_samples(
    pes_name_list,
    n_samples,
    hr_grid_size,
)


model_parameters = {
    "in_channels": len(properties_list),
    "grid_size": hr_grid_size,
    "hidden_channels": [16, 32],
    "kernel_size": [3, 3],
    "pool_size": [2, 2],
}


D = CnnDiscriminator(model_parameters).to(device)
lr_D = 2e-4
bce_loss = nn.BCEWithLogitsLoss()
mse_loss = nn.MSELoss()
optD = optim.Adam(D.parameters(), lr=lr_D)

D.pre_train(bce_loss, n_samples=[1000], num_epochs=100)

G = torch.load(
    "/home/albplanas/Desktop/Programming/PES_DL/PES_DL/saved_models/"
    + "Dev_Generator.pth",
    weights_only=False,
).to(device)


train_loader, test_loader, _, _ = generate_disciminator_training_set_from_G(
    df_high_res,
    G,
    parallel=False,
    world_size=2,
    rank=0,
    batch_size=batch_size,
    up_scale=upscale_factor,
    properties_list=["r", "energy", "derivative", "inverse_derivative"],
    properties_format="table_1D",
    test_split=0,
    device=device,
)


print("Training beginning...")
# -- Training loop --
for epoch in range(num_epochs):
    # 1) Sample real high-res sine: batch of 150-point sine waves
    #    then take the first 10 as generator input
    realAcc = []
    fakeAcc = []
    initial_params = [param.clone() for param in D.parameters()]

    for real_hr, fake_hr in train_loader:

        # Labels
        real_labels = torch.ones((batch_size, 1), device=device)
        fake_labels = torch.zeros((batch_size, 1), device=device)

        # -- Train Discriminator on real --
        D.zero_grad()

        real_pred = D(real_hr[:, 1:, :])  # [B]
        loss_real = bce_loss(real_pred, real_labels)
        realAcc.append(
            100 * torch.mean(((real_pred > 0) == real_labels).float()).item()
        )

        # # -- Train Discriminator on fake --
        fake_pred = D(fake_hr[:, 1:, :])  # [B]
        loss_fake = bce_loss(fake_pred, fake_labels)
        fakeAcc.append(
            100 * torch.mean(((fake_pred > 0) == fake_labels).float()).item()
        )

        D_loss = (loss_real + loss_fake) / 2
        D_loss.backward()

        optD.step()

    # trained_params = [param.clone() for param in D.parameters()]

    # for initial, trained in zip(initial_params, trained_params):
    #     if torch.equal(initial, trained):
    #         print("D Model has not been trained.")

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch} | D_loss: {D_loss.item():.2e}, realAcc: {realAcc[-1]:.2f} | fakeAcc: {fakeAcc[-1]:.2f}"
        )
