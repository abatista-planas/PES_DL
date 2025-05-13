import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.interpolate import CubicSpline  # type: ignore

from pes_1D.data_generator import (
    generate_generator_training_set_from_df,
    generate_true_pes_samples,
)
from pes_1D.generator import ResNetUpscaler
from pes_1D.utils import Normalizers

num_epochs = 2000
batch_size = 50
pes_name_list = ["lennard_jones", "morse"]
n_samples = [2500, 2500]
upscale_factor = 100
lr_grid_size = 8
hr_grid_size = lr_grid_size * upscale_factor
test_split = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_high_res = generate_true_pes_samples(
    pes_name_list,
    n_samples,
    hr_grid_size,
)

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
    pes_name_list,
    n_samples,
    hr_grid_size,
)

test_loader, _, _, _, _ = generate_generator_training_set_from_df(
    df_high_test,
    batch_size=len(df_high_test),
    up_scale=upscale_factor,
    properties_list=["energy"],
    properties_format="array",
    test_split=0,
    device=device,
)


# -- Hyperparameters --
lr_D, lr_G = 1e-3, 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Models, losses, optimizers --
G = ResNetUpscaler(upscale_factor=upscale_factor, num_channels=16, num_blocks=2).to(
    device
)


mse_loss = nn.MSELoss()
optG = optim.Adam(G.parameters(), lr=lr_G)


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

r_hr = df_high_test.iloc[0].pes["r"].to_numpy()
energy_hr = df_high_test.iloc[0].pes["energy"].to_numpy()
indices = np.arange(0, energy_hr.shape[0] - 1, upscale_factor)

energy_lr = energy_hr[indices]
r_lr = r_hr[indices]

plt.plot(r_hr, energy_hr, label="hr-grid", color="blue")
plt.scatter(r_lr, energy_lr, label="lr-grid", color="g")


(line1,) = ax.plot(
    r_hr, np.zeros_like(r_hr), "r.--", label="model", markersize=1
)  # Returns a tuple of line objects, thus the comma

plt.xlabel("r")
plt.ylabel("Energy")
plt.title("Sample TEST PES")
plt.legend()


def eval_models():
    G.eval()

    energy_pred = G(
        torch.from_numpy(energy_lr).float().unsqueeze(0).unsqueeze(1).to(device)
    )

    fake_hr = energy_pred.detach().cpu()  # [B,150]

    fake_hr_np = fake_hr.squeeze(0).squeeze(0).numpy()

    test_mse = 0
    test_mse_fig = 0

    with torch.no_grad():  # deactivates autograd
        for X, y in test_loader:
            y_pred = G.forward(X)
            loss = mse_loss(y_pred, y)
            test_mse += loss.item()

            test_mse_fig = mse_loss(
                G.forward(X[0, :, :].unsqueeze(0)), y[0, :, :].unsqueeze(0)
            ).item()

    test_rmse = np.sqrt(test_mse)
    ax.set_title(f"RMSE: {np.sqrt(test_mse_fig):.2e}")
    line1.set_ydata(fake_hr_np)
    fig.canvas.draw()
    fig.canvas.flush_events()

    return test_rmse


test_rmse = eval_models()
print(f"Test RMSE: {test_rmse:.2e}")


print("Training beginning...")
# -- Training loop --


def df(r_input, v_input):

    cs = CubicSpline(r_input, v_input, extrapolate=True)
    deriv = cs.derivative(1)(r_input)
    # Normalize the derivative
    return Normalizers.normalize(deriv).reshape(deriv.shape)


def prepare_discriminator_data(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Prepare discriminator data for training.
    """

    output_tensor = torch.zeros((input_tensor.shape[0], 3, input_tensor.shape[-1]))
    output_tensor[:, 0, :] = input_tensor[:, 1, :]
    in_tensor = input_tensor.detach().cpu().numpy()

    for i in range(in_tensor.shape[0]):
        r_input = in_tensor[i, 0, :]
        v_input = in_tensor[i, 1, :]

        deriv = df(r_input, v_input)
        inverse_deriv = Normalizers.normalize(1 / np.array(deriv))
        output_tensor[i, 1, :] = torch.tensor(deriv)
        output_tensor[:, 2, :] = torch.tensor(inverse_deriv)
    return output_tensor


for epoch in range(num_epochs):
    # 1) Sample real high-res sine: batch of 150-point sine waves
    #    then take the first 10 as generator input

    initial_params = [param.clone() for param in G.parameters()]

    for lr, hr in train_loader:
        G.train()

        input_tensor = hr[:, :2, :].clone().detach().cpu()  # [B,150]

        # -- Train Generator --
        G.zero_grad()
        gen_hr = G(lr[:, 1, :].unsqueeze(1))  # [B,150]
        input_tensor[:, 1, :] = gen_hr[:, 0, :]
        fake_pes = prepare_discriminator_data(input_tensor).to(device)

        mse = mse_loss(gen_hr, hr[:, 1, :].unsqueeze(1))
        derivative_mse = mse_loss(fake_pes[:, 1, :], hr[:, 2, :])

        smooth_weight = 10**-2
        lossG = mse + smooth_weight * derivative_mse  # combine

        lossG.backward()
        optG.step()

    trained_params = [param.clone() for param in G.parameters()]

    for initial, trained in zip(initial_params, trained_params):
        if torch.equal(initial, trained):
            print("Model has not been trained.")

    if epoch % 50 == 0:
        test_mse = eval_models()

        print(
            f"Epoch {epoch} , rmse: {np.sqrt(mse.item()): .2e} | test_rmse: {test_mse:.2e}"
        )
