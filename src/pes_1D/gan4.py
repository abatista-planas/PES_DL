import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from pes_1D.data_generator import (
    generate_generator_training_set_from_df,
    generate_true_pes_samples,
    get_generator_hr,
)
from pes_1D.discriminator import CnnDiscriminator  # type: ignore
from pes_1D.generator import ResNetUpscaler_B

num_epochs = 200
batch_size = 25
pes_name_list = ["lennard_jones", "morse"]
n_samples = [10000, 10000]
upscale_factor = 25
lr_grid_size = 8
hr_grid_size = lr_grid_size * upscale_factor
test_split = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_high_res = generate_true_pes_samples(pes_name_list, n_samples, hr_grid_size, seed=33)


train_loader, _, _, _, _ = generate_generator_training_set_from_df(
    df_high_res,
    batch_size=batch_size,
    up_scale=upscale_factor,
    properties_list=["r", "energy", "derivative", "inverse_derivative"],
    properties_format="table_1D",
    test_split=0,
    device=device,
)

df_high_test = generate_true_pes_samples(
    pes_name_list, [1000, 1000], hr_grid_size, seed=45
)

test_loader, _, _, _, _ = generate_generator_training_set_from_df(
    df_high_test,
    batch_size=len(df_high_test),
    up_scale=upscale_factor,
    properties_list=["r", "energy", "derivative", "inverse_derivative"],
    properties_format="table_1D",
    test_split=0,
    device=device,
)


# -- Hyperparameters --
lr_D, lr_G = 1e-4, 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Models, losses, optimizers --
G = ResNetUpscaler_B(upscale_factor=upscale_factor, num_channels=16, num_blocks=2).to(
    device
)
D = CnnDiscriminator(
    {
        "in_channels": 3,
        "grid_size": hr_grid_size,
        "hidden_channels": [16, 32],
        "kernel_size": [3, 3],
        "pool_size": [2, 2],
    }
).to(device)

bce_loss = nn.BCEWithLogitsLoss()
optD = optim.Adam(D.parameters(), lr=lr_D)
mse_loss = nn.MSELoss()
optG = optim.Adam(G.parameters(), lr=lr_G)


plt.ion()

fig, axes = plt.subplots(2, 1)
ax1 = axes[0]
ax2 = axes[1]

setup_graph = False


def plot_sample(lr, hr, fake_hr=[], idx=0):
    global setup_graph
    global fig, ax1, line1, ax2, line2

    hr_ = hr.clone().detach().cpu().numpy()

    if not setup_graph:

        lr_ = lr.clone().detach().cpu().numpy()

        setup_graph = True

        ax1.plot(hr_[idx, 0, :], hr_[idx, 1, :], label="hr-grid", color="blue")
        ax1.scatter(lr_[idx, 0, :], lr_[idx, 1, :], label="lr-grid", color="g")

        (line1,) = ax1.plot(
            hr_[idx, 0, :],
            np.zeros_like(hr_[idx, 0, :]),
            "r.--",
            label="model",
            markersize=1,
        )  # Returns a tuple of line objects, thus the comma

        ax1.set_xlabel("r")
        ax1.set_ylabel("Energy")

        ax2.plot(hr_[idx, 0, :], hr_[idx, 2, :], label="hr-grid", color="blue")
        ax2.scatter(lr_[idx, 0, :], lr_[idx, 2, :], label="lr-grid", color="g")

        (line2,) = ax2.plot(
            hr_[idx, 0, :],
            np.zeros_like(hr_[idx, 0, :]),
            "r.--",
            label="model",
            markersize=1,
        )  # Returns a tuple of line objects, thus the comma

        ax2.set_xlabel("r")
        ax2.set_ylabel("1st Derivative")

        ax1.legend()
    else:
        fhr_ = fake_hr.clone().detach().cpu().numpy()

        ax1.title.set_text(
            f"Energy RMSE : {np.sqrt(np.mean((fhr_[idx,1,:] -hr_[idx,1,:])**2)):.2e} | Derivative RMSE: {np.sqrt(np.mean((fhr_[idx,2,:] -hr_[idx,2,:])**2)):.2e}"
        )
        line1.set_ydata(fhr_[idx, 1, :])
        line2.set_ydata(fhr_[idx, 2, :])

        fig.canvas.draw()
        fig.canvas.flush_events()


def eval_models():
    G.eval()
    D.eval()

    with torch.no_grad():
        for test_lr, test_hr in test_loader:

            fake_hr = get_generator_hr(G, test_lr, test_hr, device)
            mse = mse_loss(fake_hr[:, 1, :], test_hr[:, 1, :])
            test_rmse = np.sqrt(mse.item())

            plot_sample(test_lr, test_hr, fake_hr, idx=0)
    return test_rmse


test_rmse = eval_models()
print(f"Test RMSE: {test_rmse:.2e}")


# print("Training beginning...")
# # -- Training loop --


# Labels
real_labels = torch.ones((batch_size, 1), device=device)
fake_labels = torch.zeros((batch_size, 1), device=device)

D.pre_train(bce_loss, n_samples=[10000], num_epochs=500)


for epoch in range(num_epochs):
    # 1) Sample real high-res sine: batch of 150-point sine waves
    #    then take the first 10 as generator input

    initial_params_G = [param.clone() for param in G.parameters()]
    initial_params_D = [param.clone() for param in D.parameters()]
    realAcc = []
    fakeAcc = []

    for lr, hr in train_loader:
        G.train()
        D.train()

        fake_hr = get_generator_hr(G, lr, hr, device)
        loss_adv = bce_loss(D(fake_hr[:, 1:, :]), real_labels)

        if loss_adv.item() < 1e-3:
            # -- Train Discriminator on real --
            print("Training Discriminator")
            D.zero_grad()

            real_pred = D(hr[:, 1:, :])  # [B]
            loss_real = bce_loss(real_pred, real_labels)
            realAcc.append(
                100 * torch.mean(((real_pred > 0) == real_labels).float()).item()
            )

            # # -- Train Discriminator on fake --
            fake_pred = D(fake_hr[:, 1:, :])
            loss_fake = bce_loss(fake_pred, fake_labels)
            fakeAcc.append(
                100 * torch.mean(((fake_pred > 0) == fake_labels).float()).item()
            )

            D_loss = (loss_real + loss_fake) / 2
            D_loss.backward()

            optD.step()

            trained_params_D = [param.clone() for param in D.parameters()]

            for initial, trained in zip(initial_params_D, trained_params_D):
                if torch.equal(initial, trained):
                    print("D Model has not been trained.")

        # ---- End of Discriminator training ----
        else:
            # -- Train Generator --

            G.zero_grad()

            mse = mse_loss(fake_hr[:, 1, :], hr[:, 1, :])
            derivative_mse = mse_loss(fake_hr[:, 2, :], hr[:, 2, :])
            variation_loss_1d = torch.mean(
                torch.abs(fake_hr[:, 1, :-1] - fake_hr[:, 1, 1:])
            )
            laplacian_loss_2d = torch.mean(
                torch.square(
                    fake_hr[:, 1, 2:] - 2 * fake_hr[:, 1, 1:-1] + fake_hr[:, 1, :-2]
                )
            )
            smooth_weight = 0.05
            D_weight = 10**-4
            v_weight = 0.01
            l_weight = 100
            mse_weight = 1
            lossG = (
                mse_weight * mse
                + smooth_weight * derivative_mse
                + D_weight * loss_adv
                + v_weight * variation_loss_1d
                + l_weight * laplacian_loss_2d
            )  # combine

            lossG.backward()
            optG.step()

            trained_params_G = [param.clone() for param in G.parameters()]

            for initial, trained in zip(initial_params_G, trained_params_G):
                if torch.equal(initial, trained):
                    print("G Model has not been trained.")

    if epoch % 50 == 0:
        test_mse = eval_models()

        if len(realAcc) > 0:
            print(
                f"Epoch {epoch} , test_rmse: {test_mse:.2e} | loss_adv: {loss_adv.item():.2e} | realAcc: {realAcc[-1]:.2f} | fakeAcc: {fakeAcc[-1]:.2f}"
            )
        else:
            print(
                f"Epoch {epoch} , rmse: {np.sqrt(mse.item()): .2e} | test_rmse: {test_mse:.2e} | mse: {mse.item():.2e} | deriv : {derivative_mse.item():.2e} | loss_adv: {loss_adv.item():.2e} | lap_loss: {laplacian_loss_2d.item():.2e}"
            )


plt.show()
plt.pause(0)
